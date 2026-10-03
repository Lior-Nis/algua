"""Paper-ledger residual rules shared by the flatten loop and the registry's bench check.

Kept in its own leaf module (stdlib plus two execution constants) so the registry's bench flatness
check and the flatten loop share one rule without the registry importing the live lane.
"""
from __future__ import annotations

import sqlite3

from algua.execution.reconcile_core import DEFAULT_TOLERANCE
from algua.execution.sizing import MIN_NOTIONAL


def paper_dust(conn: sqlite3.Connection, symbol: str, qty: float) -> bool:
    """True when a paper residual is within the reconcile tolerance, or worth less than the
    venue's minimum order (``MIN_NOTIONAL``) at the symbol's latest recorded paper fill.
    Notional-sized fills leave such cross-tenant residuals (#677) and the venue refuses to trade
    them; with no recorded price there is no proof, so the residual is not dust."""
    if abs(qty) <= DEFAULT_TOLERANCE:
        return True
    row = conn.execute(
        "SELECT price FROM paper_venue_fills WHERE symbol = ? ORDER BY fill_ts DESC, id DESC"
        " LIMIT 1", (symbol,)).fetchone()
    return row is not None and abs(qty) * float(row[0]) < MIN_NOTIONAL


def paper_ledger_net(conn: sqlite3.Connection, symbol: str) -> float:
    """What the paper ledger says the shared account holds in ``symbol``: the sum of every recorded
    paper fill, whichever tenant it is attributed to."""
    row = conn.execute(
        "SELECT COALESCE(SUM(qty), 0.0) FROM paper_venue_fills WHERE symbol = ?", (symbol,)
    ).fetchone()
    return float(row[0])
