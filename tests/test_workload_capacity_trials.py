"""End-to-end tests for the read-only repeated-workload capacity trial.

POST /datasets/{dataset_id}/workload-capacity-trials
"""
from __future__ import annotations

import itertools
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-workload-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "workload.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()
_rev_counter = itertools.count()

TOL = 1e-12


def _new_dataset_id(name: str) -> str:
    return f"dswt-{name}-{next(_counter)}"


def _new_reversal_id() -> str:
    return f"rev-wt-{next(_rev_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


async def _trial(client, dataset_id: str, items: list[dict], max_runs: int = 1000):
    return await client.post(
        f"/datasets/{dataset_id}/workload-capacity-trials",
        json={"items": items, "max_runs": max_runs},
    )


async def _grant(client, ds: str, request_id: str, epsilon: float,
                 delta: float = 1e-7, query_type: str = "q"):
    return await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": request_id, "query_type": query_type,
              "epsilon": epsilon, "delta": delta},
    )


async def _backdate(dataset_id: str, reservation_id: str,
                    expires_at: datetime) -> None:
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn,
            dataset_id=dataset_id,
            reservation_id=reservation_id,
            expires_at=expires_at.isoformat(),
        )
        await conn.commit()


def _brute_force_max(
    committed: list[tuple[float, float]],
    run: list[tuple[float, float]],
    cap: float,
    target_delta: float,
    max_runs: int,
) -> int:
    """Largest k in 0..max_runs with committed + k*run under the cap."""
    for k in range(max_runs, -1, -1):
        _, eps, _ = rdp.evaluate_entries(committed + run * k, target_delta)
        if max(0.0, eps) <= cap + TOL:
            return k
    return 0


def _assert_projection(body_projection: dict, repeats: int,
                       expected_eps: float, cap: float) -> None:
    assert body_projection["complete_runs"] == repeats
    assert body_projection["projected_epsilon"] == pytest.approx(
        expected_eps, abs=1e-12
    )
    assert body_projection["epsilon_remaining"] == pytest.approx(
        max(0.0, cap - expected_eps), abs=1e-12
    )
    assert body_projection["fits_cap"] == (expected_eps <= cap + TOL)


# ----------------------------------------------------------------- behaviour --


@pytest.mark.asyncio
async def test_pure_epsilon_repetitions_use_exact_addition(client):
    ds = _new_dataset_id("pure")
    await _register(client, ds, eps=1.0, delta=1e-6)
    r = await _trial(client, ds, [{"epsilon": 0.15, "delta": 0.0}], max_runs=100)
    assert r.status_code == 200
    body = r.json()
    assert body["max_complete_runs"] == 6  # 6 * 0.15 = 0.9; 7 * 0.15 = 1.05
    assert body["epsilon_committed"] == pytest.approx(0.0)
    at_max = body["projection_at_max"]
    _assert_projection(at_max, 6, 0.9, 1.0)
    assert at_max["rdp_best_order"] == rdp.RDP_ORDERS[0]
    one_more = body["projection_one_more_run"]
    _assert_projection(one_more, 7, 1.05, 1.0)
    assert one_more["fits_cap"] is False


@pytest.mark.asyncio
async def test_pure_epsilon_group_sums_items_per_run(client):
    ds = _new_dataset_id("puregroup")
    # Each run composes 0.1 + 0.05 = 0.15 exactly.
    await _register(client, ds, eps=0.5, delta=1e-6)
    items = [
        {"epsilon": 0.1, "delta": 0.0},
        {"epsilon": 0.05, "delta": 0.0},
    ]
    body = (await _trial(client, ds, items, max_runs=50)).json()
    assert body["max_complete_runs"] == 3  # 3 * 0.15 = 0.45; 4 * 0.15 = 0.6
    _assert_projection(body["projection_at_max"], 3, 0.45, 0.5)
    _assert_projection(body["projection_one_more_run"], 4, 0.6, 0.5)


@pytest.mark.asyncio
async def test_approx_dp_repetitions_match_rdp_accounting(client):
    ds = _new_dataset_id("approx")
    await _register(client, ds, eps=1.0, delta=1e-6)
    body = (await _trial(
        client, ds, [{"epsilon": 0.1, "delta": 1e-8}], max_runs=100
    )).json()
    # Independent rdp.py accounting: 9 runs (0.921927) fit, 10 (1.024176) do not.
    assert body["max_complete_runs"] == 9
    _, eps9, order9 = rdp.evaluate_entries([(0.1, 1e-8)] * 9, 1e-6)
    _, eps10, _ = rdp.evaluate_entries([(0.1, 1e-8)] * 10, 1e-6)
    _assert_projection(body["projection_at_max"], 9, max(0.0, eps9), 1.0)
    _assert_projection(body["projection_one_more_run"], 10, max(0.0, eps10), 1.0)
    assert body["projection_at_max"]["rdp_best_order"] == order9


@pytest.mark.asyncio
async def test_mixed_pure_and_approx_items_in_group(client):
    ds = _new_dataset_id("mixed")
    cap, target_delta = 1.0, 1e-6
    await _register(client, ds, eps=cap, delta=target_delta)
    run = [(0.1, 1e-8), (0.02, 0.0)]
    body = (await _trial(
        client, ds,
        [{"epsilon": 0.1, "delta": 1e-8}, {"epsilon": 0.02, "delta": 0.0}],
        max_runs=100,
    )).json()
    expected = _brute_force_max([], run, cap, target_delta, 100)
    assert body["max_complete_runs"] == expected
    _, eps_k, _ = rdp.evaluate_entries(run * expected, target_delta)
    _, eps_k1, _ = rdp.evaluate_entries(run * (expected + 1), target_delta)
    _assert_projection(
        body["projection_at_max"], expected, max(0.0, eps_k), cap
    )
    _assert_projection(
        body["projection_one_more_run"], expected + 1, max(0.0, eps_k1), cap
    )
    assert body["projection_one_more_run"]["fits_cap"] is False


@pytest.mark.asyncio
async def test_zero_runs_feasible_is_reported(client):
    ds = _new_dataset_id("zero")
    # One run alone already exceeds the cap: answer is zero, but the zero-run
    # projection is still returned.
    await _register(client, ds, eps=0.1, delta=1e-6)
    body = (await _trial(
        client, ds, [{"epsilon": 0.5, "delta": 1e-8}], max_runs=10
    )).json()
    assert body["max_complete_runs"] == 0
    at_zero = body["projection_at_max"]
    _assert_projection(at_zero, 0, 0.0, 0.1)
    assert at_zero["fits_cap"] is True
    one_more = body["projection_one_more_run"]
    assert one_more["complete_runs"] == 1
    assert one_more["fits_cap"] is False
    assert one_more["projected_epsilon"] > 0.1


@pytest.mark.asyncio
async def test_zero_runs_with_existing_committed_occupancy(client):
    ds = _new_dataset_id("zerocommitted")
    await _register(client, ds, eps=0.2, delta=1e-6)
    await _grant(client, ds, "base", 0.15, delta=0.0)
    # Only 0.05 epsilon headroom; a 0.1 pure run never fits even once.
    body = (await _trial(
        client, ds, [{"epsilon": 0.1, "delta": 0.0}], max_runs=5
    )).json()
    assert body["max_complete_runs"] == 0
    assert body["epsilon_committed"] == pytest.approx(0.15)
    _assert_projection(body["projection_at_max"], 0, 0.15, 0.2)
    _assert_projection(body["projection_one_more_run"], 1, 0.25, 0.2)


@pytest.mark.asyncio
async def test_committed_granted_requests_reduce_capacity(client):
    ds = _new_dataset_id("committed")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "base", 0.4, delta=0.0)
    body = (await _trial(
        client, ds, [{"epsilon": 0.1, "delta": 0.0}], max_runs=100
    )).json()
    # 0.4 + k * 0.1 <= 1.0  ->  k <= 6.
    assert body["max_complete_runs"] == 6
    assert body["epsilon_committed"] == pytest.approx(0.4)
    _assert_projection(body["projection_at_max"], 6, 1.0, 1.0)
    _assert_projection(body["projection_one_more_run"], 7, 1.1, 1.0)


@pytest.mark.asyncio
async def test_active_reservation_reduces_capacity(client):
    ds = _new_dataset_id("reserved")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "r1", "epsilon": 0.5, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    body = (await _trial(
        client, ds, [{"epsilon": 0.1, "delta": 0.0}], max_runs=100
    )).json()
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    committed = budget["epsilon_consumed"] + budget["epsilon_reserved"]
    assert body["epsilon_committed"] == pytest.approx(committed, abs=1e-12)
    # The reservation (approximate DP) plus k pure runs: cross-check with the
    # independent rdp.py composition.
    expected = _brute_force_max([(0.5, 1e-8)], [(0.1, 0.0)], 1.0, 1e-6, 100)
    assert body["max_complete_runs"] == expected


@pytest.mark.asyncio
async def test_reversed_requests_do_not_count(client):
    ds = _new_dataset_id("reversed")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "q1", 0.2, delta=0.0)
    await client.post(
        f"/datasets/{ds}/requests/q1/reversals",
        json={"reversal_id": _new_reversal_id(),
              "reason": "trial setup cleanup"},
    )
    body = (await _trial(
        client, ds, [{"epsilon": 0.25, "delta": 0.0}], max_runs=10
    )).json()
    assert body["epsilon_committed"] == pytest.approx(0.0, abs=1e-12)
    # 4 * 0.25 = 1.0 fits exactly.
    assert body["max_complete_runs"] == 4


@pytest.mark.asyncio
async def test_past_ttl_reservations_excluded_without_being_swept(client):
    ds = _new_dataset_id("expired")
    await _register(client, ds, eps=1.0, delta=1e-6)
    # Create the reservation then backdate it; no budget-touching endpoint
    # runs in between, so the lazy sweep never gets the chance to expire it.
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "exp", "epsilon": 0.4, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    await _backdate(
        ds, "exp", datetime.now(timezone.utc) - timedelta(seconds=5)
    )
    body = (await _trial(
        client, ds, [{"epsilon": 0.25, "delta": 0.0}], max_runs=10
    )).json()
    # The past-TTL reservation is treated as inactive...
    assert body["epsilon_committed"] == pytest.approx(0.0, abs=1e-12)
    assert body["max_complete_runs"] == 4
    # ...but the read-only trial must not sweep it itself.
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "exp")
        assert row["status"] == "granted"
        assert row["expired_at"] is None


@pytest.mark.asyncio
async def test_reaching_max_runs_omits_one_more_run(client):
    ds = _new_dataset_id("capped")
    await _register(client, ds, eps=100.0, delta=1e-6)
    body = (await _trial(
        client, ds, [{"epsilon": 0.1, "delta": 0.0}], max_runs=10
    )).json()
    assert body["max_complete_runs"] == 10
    assert body["projection_one_more_run"] is None
    at_max = body["projection_at_max"]
    assert at_max["complete_runs"] == 10
    assert at_max["fits_cap"] is True


@pytest.mark.asyncio
async def test_capacity_limited_by_max_runs_even_when_more_would_fit(client):
    ds = _new_dataset_id("runbound")
    await _register(client, ds, eps=1.0, delta=1e-6)
    # 100000 runs of 0.001 would fit exactly; with max_runs=3 the answer is 3.
    body = (await _trial(
        client, ds, [{"epsilon": 0.001, "delta": 0.0}], max_runs=3
    )).json()
    assert body["max_complete_runs"] == 3
    assert body["projection_one_more_run"] is None
    _assert_projection(body["projection_at_max"], 3, 0.003, 1.0)


@pytest.mark.asyncio
async def test_large_max_runs_pure_epsilon(client):
    ds = _new_dataset_id("largepure")
    await _register(client, ds, eps=1.0, delta=1e-6)
    body = (await _trial(
        client, ds, [{"epsilon": 0.00002, "delta": 0.0}], max_runs=100_000
    )).json()
    assert body["max_complete_runs"] == 50_000  # 50000 * 2e-5 = 1.0
    _assert_projection(body["projection_at_max"], 50_000, 1.0, 1.0)
    _assert_projection(body["projection_one_more_run"], 50_001, 1.00002, 1.0)


@pytest.mark.asyncio
async def test_closed_form_matches_brute_force_on_mixed_scenarios(client):
    ds = _new_dataset_id("brute")
    cap, target_delta = 1.0, 1e-6
    await _register(client, ds, eps=cap, delta=target_delta)
    # Pre-existing occupancy mixing pure and approximate DP.
    await _grant(client, ds, "base1", 0.08, delta=1e-9)
    await _grant(client, ds, "base2", 0.03, delta=0.0)
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "r1", "epsilon": 0.05, "delta": 1e-10,
              "ttl_seconds": 3600},
    )
    run = [(0.07, 1e-8), (0.02, 1e-8), (0.01, 0.0)]
    committed = [(0.08, 1e-9), (0.03, 0.0), (0.05, 1e-10)]
    for max_runs in (1, 5, 50):
        body = (await _trial(client, ds, [
            {"epsilon": 0.07, "delta": 1e-8},
            {"epsilon": 0.02, "delta": 1e-8},
            {"epsilon": 0.01, "delta": 0.0},
        ], max_runs=max_runs)).json()
        expected = _brute_force_max(
            committed, run, cap, target_delta, max_runs
        )
        assert body["max_complete_runs"] == expected
        k = expected
        _, eps_k, _ = rdp.evaluate_entries(
            committed + run * k, target_delta
        )
        assert body["projection_at_max"]["projected_epsilon"] == pytest.approx(
            max(0.0, eps_k), abs=1e-12
        )
        if k < max_runs:
            _, eps_k1, _ = rdp.evaluate_entries(
                committed + run * (k + 1), target_delta
            )
            assert body["projection_one_more_run"]["projected_epsilon"] == \
                pytest.approx(max(0.0, eps_k1), abs=1e-12)
            assert body["projection_one_more_run"]["fits_cap"] is False
        else:
            assert body["projection_one_more_run"] is None


@pytest.mark.asyncio
async def test_trial_is_strictly_read_only(client):
    ds = _new_dataset_id("readonly")
    await _register(client, ds, eps=0.5, delta=1e-6)
    # An expired reservation exists but must not be swept by the trial.
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "exp", "epsilon": 0.4, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    await _backdate(
        ds, "exp", datetime.now(timezone.utc) - timedelta(seconds=5)
    )
    items = [{"epsilon": 0.3, "delta": 1e-8}]

    first = (await _trial(client, ds, items, max_runs=10)).json()
    second = (await _trial(client, ds, items, max_runs=10)).json()
    # generated_at differs; everything else must be identical.
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second

    # Still granted / unswept.
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "exp")
        assert row["status"] == "granted"
        assert row["expired_at"] is None
    # No records of any kind were created.
    assert (await client.get(f"/datasets/{ds}/requests")).json()["count"] == 0
    assert (
        await client.get(f"/datasets/{ds}/reservations")
    ).json()["count"] == 1
    assert (await client.get(f"/datasets/{ds}/reversals")).json()["count"] == 0


@pytest.mark.asyncio
async def test_trial_recomputes_as_ledger_changes(client):
    ds = _new_dataset_id("fresh")
    await _register(client, ds, eps=0.5, delta=1e-6)
    items = [{"epsilon": 0.2, "delta": 0.0}]

    before = (await _trial(client, ds, items, max_runs=10)).json()
    assert before["max_complete_runs"] == 2  # 2 * 0.2 = 0.4
    await _grant(client, ds, "holder", 0.2, delta=0.0)
    after = (await _trial(client, ds, items, max_runs=10)).json()
    # 0.2 committed leaves room for one more 0.2 run only.
    assert after["epsilon_committed"] == pytest.approx(0.2)
    assert after["max_complete_runs"] == 1


# ---------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_trial_missing_dataset_404(client):
    r = await _trial(
        client, "no-such-dataset", [{"epsilon": 0.1, "delta": 1e-8}]
    )
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_trial_item_count_boundaries(client):
    ds = _new_dataset_id("bounds")
    await _register(client, ds, eps=10.0, delta=1e-6)
    one = [{"epsilon": 0.01, "delta": 0.0}]
    five = [{"epsilon": 0.01, "delta": 0.0} for _ in range(5)]
    six = [{"epsilon": 0.01, "delta": 0.0} for _ in range(6)]
    assert (await _trial(client, ds, one)).status_code == 200
    assert (await _trial(client, ds, five)).status_code == 200
    assert (await _trial(client, ds, [])).status_code == 422
    assert (await _trial(client, ds, six)).status_code == 422


@pytest.mark.asyncio
async def test_trial_max_runs_boundaries(client):
    ds = _new_dataset_id("runbounds")
    await _register(client, ds, eps=1000.0, delta=1e-6)
    items = [{"epsilon": 0.001, "delta": 0.0}]
    assert (await _trial(client, ds, items, max_runs=1)).status_code == 200
    assert (
        await _trial(client, ds, items, max_runs=100_000)
    ).status_code == 200
    assert (await _trial(client, ds, items, max_runs=0)).status_code == 422
    assert (
        await _trial(client, ds, items, max_runs=100_001)
    ).status_code == 422


@pytest.mark.asyncio
async def test_trial_invalid_epsilon_delta_422(client):
    ds = _new_dataset_id("badefd")
    await _register(client, ds)
    assert (
        await _trial(client, ds, [{"epsilon": 0.0, "delta": 0.0}])
    ).status_code == 422
    assert (
        await _trial(client, ds, [{"epsilon": 0.1, "delta": 1.0}])
    ).status_code == 422
    assert (
        await _trial(client, ds, [{"epsilon": -0.1, "delta": 0.0}])
    ).status_code == 422


@pytest.mark.asyncio
async def test_trial_does_not_create_verdict_records(client):
    ds = _new_dataset_id("norecords")
    await _register(client, ds, eps=0.01, delta=1e-6)
    r = await _trial(
        client,
        ds,
        [{"epsilon": 5.0, "delta": 1e-8}, {"epsilon": 0.001, "delta": 0.0}],
        max_runs=10,
    )
    assert r.status_code == 200
    # The over-budget workload leaves nothing behind.
    assert (await client.get(f"/datasets/{ds}/requests")).json()["count"] == 0
