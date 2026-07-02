"""Persistence for the account-wide high-water mark the book-level drawdown breaker measures
against (#390). The aggregate analog of `execution.order_state`'s per-strategy NAV peak: one row
(`book_equity_peak`, id=1) that ratchets up with account equity and is cleared on resume-all.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime


def get_book_peak(conn: sqlite3.Connection) -> float | None:
    row = conn.execute("SELECT peak FROM book_equity_peak WHERE id = 1").fetchone()
    return float(row["peak"]) if row is not None else None


def update_book_peak(conn: sqlite3.Connection, equity: float) -> float:
    """Ratchet the account high-water mark up to `equity` and return the new peak.

    Rejects a non-finite / non-positive `equity` with ValueError: the peak is the drawdown
    denominator, so a bad read must never corrupt it (the caller validates equity and fails closed
    BEFORE reaching here; this is defense-in-depth)."""
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError(
            f"book equity {equity!r} is not a usable (positive, finite) high-water value"
        )
    prior = get_book_peak(conn)
    peak = equity if prior is None else max(prior, equity)
    conn.execute(
        "INSERT INTO book_equity_peak(id, peak, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET peak = excluded.peak, updated_at = excluded.updated_at",
        (peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_book_peak(conn: sqlite3.Connection) -> None:
    """Wipe the account high-water mark — used by resume-all after the whole account is flattened
    to cash, so the book re-bases its drawdown denominator on the next cycle."""
    conn.execute("DELETE FROM book_equity_peak")
    conn.commit()
