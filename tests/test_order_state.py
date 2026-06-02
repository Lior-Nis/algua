from datetime import UTC, datetime

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.order_state import derive_positions, persist_run, reconcile
from algua.execution.sim_broker import Fill
from algua.live.paper_loop import PaperRunResult
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
        orders=[OrderIntent("AAA", Side.BUY, 1.0, T0)],
        fills=[Fill("AAA", 50.0, 100.0, T0, T1, "sim-1")],
        final_positions={"AAA": 50.0}, final_cash=5000.0,
        final_equity=10000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 1
    assert derive_positions(conn, "s") == {"AAA": 50.0}


def test_reconcile_true_on_match_false_on_mismatch():
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 50.0})) is True
    assert reconcile({"AAA": 50.0}, pd.Series({"AAA": 49.0})) is False
    assert reconcile({}, pd.Series(dtype="float64")) is True


def test_persist_run_replaces_prior_paper_state_not_accumulates(tmp_path):
    conn = _conn(tmp_path)
    result = PaperRunResult(
        strategy="s",
        orders=[OrderIntent("AAA", Side.BUY, 1.0, T0)],
        fills=[Fill("AAA", 50.0, 100.0, T0, T1, "sim-1")],
        final_positions={"AAA": 50.0}, final_cash=5000.0,
        final_equity=10000.0, reconcile_ok=True,
    )
    persist_run(conn, result)
    persist_run(conn, result)  # re-running the same replay must REPLACE, not double
    assert conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 1
    assert derive_positions(conn, "s") == {"AAA": 50.0}
