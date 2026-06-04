from datetime import UTC, datetime

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.order_state import (
    client_order_id,
    count_orders,
    derive_positions,
    get_peak_equity,
    persist_run,
    reconcile,
    record_submitted_order,
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


def test_reconcile_true_on_match_false_on_mismatch():
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 50.0})) is True
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 49.0})) is False
    assert reconcile({}, pd.Series(dtype="float64")) is True


def test_client_order_id_deterministic_and_sanitised():
    a = client_order_id("mom strat", T0, "BRK.B")
    b = client_order_id("mom strat", T0, "BRK.B")
    assert a == b  # identical inputs -> identical id (idempotent retries)
    assert client_order_id("mom strat", T0, "AAA") != a  # symbol changes the id
    assert all(c.isalnum() or c in "-_" for c in a)  # only Alpaca-safe chars
    assert len(a) <= 128


def test_record_submitted_order_persists_immediately(tmp_path):
    conn = _conn(tmp_path)
    record_submitted_order(conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1")
    row = conn.execute(
        "SELECT strategy, symbol, status, broker_order_id FROM paper_orders"
    ).fetchone()
    assert (row["strategy"], row["symbol"], row["status"], row["broker_order_id"]) == (
        "s", "AAA", "submitted", "alp-1")


def test_count_orders_scoped_to_strategy(tmp_path):
    conn = _conn(tmp_path)
    assert count_orders(conn, "s") == 0
    record_submitted_order(conn, "s", "AAA", "buy", 1.0, T0.isoformat(), "alp-1")
    record_submitted_order(conn, "s", "BBB", "buy", 1.0, T0.isoformat(), "alp-2")
    record_submitted_order(conn, "other", "CCC", "buy", 1.0, T0.isoformat(), "alp-3")
    assert count_orders(conn, "s") == 2
    assert count_orders(conn, "other") == 1


def test_peak_equity_ratchets_up(tmp_path):
    conn = _conn(tmp_path)
    assert get_peak_equity(conn, "s") is None
    assert update_peak_equity(conn, "s", 100.0) == 100.0
    assert update_peak_equity(conn, "s", 120.0) == 120.0  # new high
    assert update_peak_equity(conn, "s", 90.0) == 120.0   # below peak -> peak unchanged
    assert get_peak_equity(conn, "s") == 120.0


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
