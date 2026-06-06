"""Append-only audit of buying-power trims/skips: when the shared per-cycle pool can't fully fund a
strategy's intended buy, the shortfall is recorded so a starved strategy is visible (not a silent
no-op)."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def record_reservation(conn: sqlite3.Connection, cycle: int, strategy: str, symbol: str,
                       intended: float, permitted: float) -> None:
    """Record that an intended buy of `intended` was funded to `permitted` (reason 'skipped' when
    permitted == 0, else 'trimmed'). Only call when permitted < intended (a full fund is silent)."""
    reason = "skipped" if permitted <= 0.0 else "trimmed"
    conn.execute(
        "INSERT INTO live_reservations"
        "(cycle, strategy, symbol, intended_notional, permitted_notional, reason, ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (cycle, strategy, symbol, intended, permitted, reason, datetime.now(UTC).isoformat()),
    )
    conn.commit()
