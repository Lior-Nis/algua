from datetime import UTC, datetime

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.order_state import derive_positions, persist_run, reconcile
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
