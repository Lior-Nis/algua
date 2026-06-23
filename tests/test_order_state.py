from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution.order_state import (
    clear_all_peaks,
    clear_peak_equity,
    client_order_id,
    count_orders,
    derive_positions,
    get_peak_equity,
    latest_tick_snapshot,
    persist_run,
    recent_orders,
    record_submitted_order,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.execution.sim_broker import Fill
from algua.live.paper_loop import OrderRecord, PaperRunResult
from algua.registry.db import connect, migrate

T0 = datetime(2023, 1, 2, tzinfo=UTC)
T1 = datetime(2023, 1, 3, tzinfo=UTC)


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


@pytest.fixture()
def conn(tmp_path):
    return _conn(tmp_path)


def test_persist_run_writes_orders_and_fills(tmp_path):
    conn = _conn(tmp_path)
    result = PaperRunResult(
        strategy="s",
        orders=[OrderRecord(OrderIntent("AAA", Side.BUY, 1.0, T0), "sim-1")],
        fills=[Fill("AAA", 50.0, 100.0, T0, T1, "sim-1")],
        final_positions={"AAA": 50.0}, final_cash=5000.0,
        final_equity=10000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 1
    assert derive_positions(conn, "s") == {"AAA": 50.0}


def test_persist_run_reads_broker_id_from_record_not_position(tmp_path):
    # broker ids are NOT sim-1/sim-2 in list order; persistence must use the recorded ids so the
    # fill (keyed on "alp-zzz") still links to its order.
    conn = _conn(tmp_path)
    result = PaperRunResult(
        strategy="s",
        orders=[
            OrderRecord(OrderIntent("AAA", Side.BUY, 0.5, T0), "alp-zzz"),
            OrderRecord(OrderIntent("BBB", Side.SELL, 0.0, T0), "alp-aaa"),
        ],
        fills=[Fill("AAA", 10.0, 100.0, T0, T1, "alp-zzz")],  # only the first order executed
        final_positions={"AAA": 10.0}, final_cash=0.0,
        final_equity=1000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    rows = conn.execute(
        "SELECT symbol, status, broker_order_id FROM paper_orders ORDER BY id"
    ).fetchall()
    assert [(r["symbol"], r["status"], r["broker_order_id"]) for r in rows] == [
        ("AAA", "filled", "alp-zzz"),
        ("BBB", "rejected", "alp-aaa"),
    ]
    assert derive_positions(conn, "s") == {"AAA": 10.0}


def test_client_order_id_deterministic_and_sanitised():
    a = client_order_id("mom strat", T0, "BRK.B")
    b = client_order_id("mom strat", T0, "BRK.B")
    assert a == b  # identical inputs -> identical id (idempotent retries)
    assert client_order_id("mom strat", T0, "AAA") != a  # symbol changes the id
    assert all(c.isalnum() or c in "-_" for c in a)  # only Alpaca-safe chars
    assert len(a) <= 128


def test_record_submitted_order_persists_immediately(tmp_path):
    conn = _conn(tmp_path)
    record_submitted_order(conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1", strategy_id=1)
    row = conn.execute(
        "SELECT strategy, symbol, status, broker_order_id FROM paper_orders"
    ).fetchone()
    assert (row["strategy"], row["symbol"], row["status"], row["broker_order_id"]) == (
        "s", "AAA", "submitted", "alp-1")


def test_count_orders_scoped_to_strategy(tmp_path):
    conn = _conn(tmp_path)
    assert count_orders(conn, "s") == 0
    record_submitted_order(conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1", strategy_id=1)
    record_submitted_order(conn, "s", "BBB", "buy", 1.0, T0.isoformat(), "alp-2", strategy_id=1)
    record_submitted_order(conn, "other", "CCC", "buy", 1.0, T0.isoformat(), "alp-3", strategy_id=2)
    assert count_orders(conn, "s") == 2
    assert count_orders(conn, "other") == 1


def test_record_submitted_order_idempotent_on_duplicate(tmp_path):
    # #18: a crash/retry or duplicate Alpaca client_order_id path can return the SAME broker order
    # again. Re-recording it must NOT create a duplicate paper_orders row.
    conn = _conn(tmp_path)
    record_submitted_order(conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1", strategy_id=1)
    record_submitted_order(  # retry — idempotent
        conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1", strategy_id=1
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE strategy='s' AND broker_order_id='alp-1'"
    ).fetchone()[0] == 1


def test_peak_equity_ratchets_up(tmp_path):
    conn = _conn(tmp_path)
    assert get_peak_equity(conn, "s") is None
    assert update_peak_equity(conn, "s", 100.0) == 100.0
    assert update_peak_equity(conn, "s", 120.0) == 120.0  # new high
    assert update_peak_equity(conn, "s", 90.0) == 120.0   # below peak -> peak unchanged
    assert get_peak_equity(conn, "s") == 120.0


def test_clear_peak_equity_rebases(tmp_path):
    conn = _conn(tmp_path)
    update_peak_equity(conn, "s", 200.0)
    clear_peak_equity(conn, "s")
    assert get_peak_equity(conn, "s") is None  # re-based: next tick starts a fresh high-water mark
    clear_peak_equity(conn, "s")  # idempotent: clearing an absent row is a no-op


def test_persist_run_partial_fill_reaches_partial_status(tmp_path):
    """SimBroker buy clamped to available cash produces Fill.status='partial'; that must be
    persisted as paper_orders.status='partial', not collapsed to 'filled'."""
    from algua.execution.sim_broker import SimBroker

    conn = _conn(tmp_path)
    # equity=$1150 (1 BBB@$1000 + $150 cash).  target_weight=0.5 for AAA@$100 → wants 5 shares,
    # but only $150 cash on hand → affordable=1.  SimBroker clamps qty to 1 → partial fill.
    broker = SimBroker(cash=150.0)
    broker.positions["BBB"] = 1.0
    intent = OrderIntent("AAA", Side.BUY, 0.5, T0)
    broker_order_id = broker.submit(intent)
    fills = broker.fill_pending(pd.Series({"AAA": 100.0, "BBB": 1000.0}), fill_ts=T1)
    assert len(fills) == 1 and fills[0].status == "partial", "SimBroker must emit partial"

    result = PaperRunResult(
        strategy="s",
        orders=[OrderRecord(intent, broker_order_id)],
        fills=fills,
        final_positions=dict(broker.get_positions()),
        final_cash=broker.cash,
        final_equity=broker.equity(pd.Series({"AAA": 100.0})),
        reconcile_ok=True,
    )
    persist_run(conn, result)

    row = conn.execute("SELECT status FROM paper_orders WHERE strategy='s'").fetchone()
    assert row["status"] == "partial"


def test_persist_run_replaces_prior_paper_state_not_accumulates(tmp_path):
    conn = _conn(tmp_path)
    result = PaperRunResult(
        strategy="s",
        orders=[OrderRecord(OrderIntent("AAA", Side.BUY, 1.0, T0), "sim-1")],
        fills=[Fill("AAA", 50.0, 100.0, T0, T1, "sim-1")],
        final_positions={"AAA": 50.0}, final_cash=5000.0,
        final_equity=10000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    persist_run(conn, result)  # re-running the same replay must REPLACE, not double
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 1
    assert derive_positions(conn, "s") == {"AAA": 50.0}


_SNAP_PROV = dict(
    lane="paper", strategy_id=1, code_hash="c", config_hash="g",
    dependency_hash="d", account_id="acct", cash=1.0, clock_source="broker",
)


def test_tick_snapshot_roundtrip_latest_wins(tmp_path):
    conn = _conn(tmp_path)
    assert latest_tick_snapshot(conn, "s") is None
    record_tick_snapshot(conn, "s", tick_ts="2023-06-01T00:00:00+00:00",
                         decision_ts="2023-05-31T00:00:00+00:00", equity=100.0, peak_equity=100.0,
                         positions={"AAA": 10.0}, n_submitted=1, reconcile_ok=True, **_SNAP_PROV)
    record_tick_snapshot(conn, "s", tick_ts="2023-06-02T00:00:00+00:00",
                         decision_ts="2023-06-01T00:00:00+00:00", equity=120.0, peak_equity=120.0,
                         positions={"AAA": 12.0}, n_submitted=0, reconcile_ok=False, **_SNAP_PROV)
    latest = latest_tick_snapshot(conn, "s")
    assert latest["equity"] == 120.0 and latest["positions"] == {"AAA": 12.0}
    assert latest["reconcile_ok"] is False and latest["n_submitted"] == 0


def test_recent_orders_newest_first_and_limit(tmp_path):
    conn = _conn(tmp_path)
    for i in range(3):
        record_submitted_order(conn, "s", f"SYM{i}", "buy", 1.0, "2023-06-01T00:00:00+00:00",
                               f"o-{i}", strategy_id=1)
    rows = recent_orders(conn, "s", limit=2)
    assert [r["broker_order_id"] for r in rows] == ["o-2", "o-1"]  # newest first, limited
    assert rows[0]["symbol"] == "SYM2" and rows[0]["side"] == "buy"


def test_clear_all_peaks_wipes_every_strategy(tmp_path):
    conn = _conn(tmp_path)
    update_peak_equity(conn, "a", 100.0)
    update_peak_equity(conn, "b", 200.0)
    clear_all_peaks(conn)
    assert get_peak_equity(conn, "a") is None and get_peak_equity(conn, "b") is None


def test_nav_peak_ratchets_and_clears(tmp_path):
    from algua.execution.order_state import clear_nav_peak, get_nav_peak, update_nav_peak
    conn = _conn(tmp_path)
    assert get_nav_peak(conn, "s1") is None
    assert update_nav_peak(conn, "s1", 10_000.0) == 10_000.0
    assert update_nav_peak(conn, "s1", 9_000.0) == 10_000.0   # only ratchets up
    assert update_nav_peak(conn, "s1", 11_000.0) == 11_000.0
    assert get_nav_peak(conn, "s1") == 11_000.0
    clear_nav_peak(conn, "s1")
    assert get_nav_peak(conn, "s1") is None


# ---------------------------------------------------------------------------
# Task 5 (#124): stamped writers — provenance columns
# ---------------------------------------------------------------------------

def test_record_tick_snapshot_stamps_provenance(conn):
    record_tick_snapshot(
        conn, "s", tick_ts="2026-06-11T14:00:00+00:00", decision_ts=None,
        equity=1.0, peak_equity=None, positions={}, n_submitted=0,
        reconcile_ok=True, lane="paper", strategy_id=7, code_hash="c",
        config_hash="g", dependency_hash="d", account_id="acct", cash=1.0,
        clock_source="broker",
    )
    row = conn.execute("SELECT * FROM tick_snapshots").fetchone()
    assert row["lane"] == "paper" and row["strategy_id"] == 7 and row["clock_source"] == "broker"
    assert row["code_hash"] == "c" and row["config_hash"] == "g" and row["dependency_hash"] == "d"
    assert row["account_id"] == "acct" and row["cash"] == 1.0
    assert row["recorded_at"]


def test_record_tick_snapshot_invalid_lane_raises(conn):
    with pytest.raises(ValueError, match="lane"):
        record_tick_snapshot(
            conn, "s", tick_ts="2026-06-11T14:00:00+00:00", decision_ts=None,
            equity=1.0, peak_equity=None, positions={}, n_submitted=0,
            reconcile_ok=True, lane="bad_lane", strategy_id=7, code_hash="c",
            config_hash="g", dependency_hash="d", account_id="acct", cash=1.0,
            clock_source="broker",
        )


def test_record_tick_snapshot_invalid_clock_source_raises(conn):
    with pytest.raises(ValueError, match="clock_source"):
        record_tick_snapshot(
            conn, "s", tick_ts="2026-06-11T14:00:00+00:00", decision_ts=None,
            equity=1.0, peak_equity=None, positions={}, n_submitted=0,
            reconcile_ok=True, lane="paper", strategy_id=7, code_hash="c",
            config_hash="g", dependency_hash="d", account_id="acct", cash=1.0,
            clock_source="bad_source",
        )


def test_record_submitted_order_persists_strategy_id(conn):
    record_submitted_order(
        conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1", strategy_id=42,
    )
    row = conn.execute("SELECT strategy_id FROM paper_orders").fetchone()
    assert row["strategy_id"] == 42
