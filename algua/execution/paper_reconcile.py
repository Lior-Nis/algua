"""Account-level reconcile for the PAPER lane (#313). Mirrors live_reconcile but gates on
`attributed_paper_net`: an account holding NO current-paper strategy owns leaves a residual and
fails closed — the multi-tenant safety semantics. The grace window (in reconcile_core) absorbs a
just-ingested fill whose order is not yet broker-id-backfilled (briefly orphan -> pending, not
halt)."""
from __future__ import annotations

import sqlite3

from algua.execution.reconcile_core import (
    DEFAULT_GRACE_CYCLES,
    DEFAULT_TOLERANCE,
    ReconcileResult,
    reconcile_account,
)
from algua.execution.reconcile_core import next_cycle as _core_next_cycle

__all__ = [
    "attributed_paper_net",
    "next_cycle",
    "paper_account_expected_net",
    "reconcile",
]


def paper_account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """Σ ALL paper_venue_fills.qty (signed) per symbol — every recorded paper fill, attributed or
    not. Zero nets omitted. (Diagnostic/account analog of live account_expected_net.)"""
    rows = conn.execute(
        "SELECT symbol, SUM(qty) AS q FROM paper_venue_fills GROUP BY symbol"
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def attributed_paper_net(conn: sqlite3.Connection) -> dict[str, float]:
    """Σ paper_venue_fills counting ONLY fills attributed to a CURRENTLY-PAPER strategy. Orphan
    (strategy IS NULL) and non-paper fills are EXCLUDED so they can never 'explain' a broker
    position. Zero nets omitted."""
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS q FROM paper_venue_fills f "
        "JOIN strategies s ON s.name = f.strategy AND s.stage = 'paper' "
        "GROUP BY f.symbol"
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    return _core_next_cycle(conn, table="paper_cycle")


def reconcile(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    cycle: int,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    return reconcile_account(
        conn,
        broker_net,
        attributed_paper_net(conn),
        cycle,
        state_table="paper_reconcile_state",
        tolerance=tolerance,
        grace_cycles=grace_cycles,
    )
