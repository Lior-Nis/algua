"""Persistence for the account-wide high-water mark the book-level drawdown breaker measures
against (#390). The aggregate analog of `execution.order_state`'s per-strategy NAV peak: one row
(`book_equity_peak`, id=1) that ratchets up with account equity and is cleared on resume-all.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime

# One high-water row PER LANE: the paper and live accounts are different brokerage accounts with
# different equity, so a shared peak would trip one lane against the other's scale (#647).
LANE_PEAK_TABLES = {"live": "book_equity_peak", "paper": "paper_book_equity_peak"}


def _table(lane: str) -> str:
    try:
        return LANE_PEAK_TABLES[lane]
    except KeyError:
        raise ValueError(f"unknown book-peak lane {lane!r}; expected one of "
                         f"{sorted(LANE_PEAK_TABLES)}") from None


def get_book_peak(conn: sqlite3.Connection, lane: str = "live") -> float | None:
    row = conn.execute(f"SELECT peak FROM {_table(lane)} WHERE id = 1").fetchone()
    return float(row["peak"]) if row is not None else None


def update_book_peak(conn: sqlite3.Connection, equity: float, lane: str = "live") -> float:
    """Ratchet the account high-water mark up to `equity` and return the new peak.

    Rejects a non-finite / non-positive `equity` with ValueError: the peak is the drawdown
    denominator, so a bad read must never corrupt it (the caller validates equity and fails closed
    BEFORE reaching here; this is defense-in-depth)."""
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError(
            f"book equity {equity!r} is not a usable (positive, finite) high-water value"
        )
    prior = get_book_peak(conn, lane)
    peak = equity if prior is None else max(prior, equity)
    conn.execute(
        f"INSERT INTO {_table(lane)}(id, peak, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET peak = excluded.peak, updated_at = excluded.updated_at",
        (peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_book_peak(conn: sqlite3.Connection, lane: str = "live") -> None:
    """Wipe the account high-water mark — used by resume-all after the whole account is flattened
    to cash, so the book re-bases its drawdown denominator on the next cycle."""
    conn.execute(f"DELETE FROM {_table(lane)}")
    conn.commit()
