"""Account-level reconcile for the LIVE lane: the books' expected net (Σ all live_fills) must match
the broker's netted book per symbol. The grace-window + persisted-state algorithm lives in
`reconcile_core`; this module supplies the live `expected` net and the live table names."""
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
    "ReconcileResult",
    "account_expected_net",
    "attributed_live_net",
    "next_cycle",
    "reconcile",
]


def account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' belief of the account net per symbol = Σ all live_fills.qty (signed), across every
    strategy (the account is shared). Zero nets are omitted."""
    rows = conn.execute("SELECT symbol, SUM(qty) AS q FROM live_fills GROUP BY symbol").fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def attributed_live_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' expected net per symbol counting ONLY fills attributed to a CURRENTLY-LIVE
    strategy. Orphan fills (strategy IS NULL) and non-live fills are EXCLUDED, so they can never
    'explain' a broker position. Used by the resume reconcile: an unattributed broker holding leaves
    an UNexplained residual and fails closed (resume refuses) rather than being silently cancelled
    out by an orphan fill of the same symbol. Zero nets are omitted."""
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS q FROM live_fills f "
        "JOIN strategies s ON s.name = f.strategy AND s.stage = 'live' "
        "GROUP BY f.symbol"
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    return _core_next_cycle(conn, table="live_cycle")


def reconcile(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    cycle: int,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    return reconcile_account(
        conn, broker_net, account_expected_net(conn), cycle,
        state_table="live_reconcile_state", tolerance=tolerance, grace_cycles=grace_cycles,
    )
