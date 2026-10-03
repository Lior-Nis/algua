"""Paper-ledger residual rules shared by the flatten loop, the registry's bench check and the paper
account reconcile.

A stdlib-only leaf with its own constants: the registry imports it without reaching the live lane,
and changing a sizing or reconcile setting elsewhere cannot widen what the protected bench check
accepts as flat.
"""
from __future__ import annotations

import sqlite3

#: A believed residual at or below this many shares is float noise.
DUST_SHARES = 1e-6
#: Alpaca refuses any order worth less than this, so a smaller residual can never be offset.
VENUE_MIN_ORDER_NOTIONAL = 1.0


def paper_dust(conn: sqlite3.Connection, symbol: str, qty: float) -> bool:
    """True when a paper residual is float noise, or worth less than the venue's minimum order at
    the symbol's latest recorded paper fill. Notional-sized fills leave such cross-tenant residuals
    (#677) and the venue refuses to trade them; with no recorded price there is no proof, so the
    residual is not dust."""
    if abs(qty) <= DUST_SHARES:
        return True
    row = conn.execute(
        "SELECT price FROM paper_venue_fills WHERE symbol = ? ORDER BY fill_ts DESC, id DESC"
        " LIMIT 1", (symbol,)).fetchone()
    return row is not None and abs(qty) * float(row[0]) < VENUE_MIN_ORDER_NOTIONAL


def paper_ledger_net(conn: sqlite3.Connection, symbol: str) -> float:
    """What the paper ledger says the shared account holds in ``symbol``: the sum of every recorded
    paper fill, whichever tenant it is attributed to."""
    row = conn.execute(
        "SELECT COALESCE(SUM(qty), 0.0) FROM paper_venue_fills WHERE symbol = ?", (symbol,)
    ).fetchone()
    return float(row[0])
