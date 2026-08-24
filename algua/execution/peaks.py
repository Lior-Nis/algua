"""Drawdown high-water-mark REBASE policy — what to clear when a halt is lifted.

Two operator commands lift a halt (`paper resume` for one strategy, `paper resume-all` for the
account), and each must re-base the drawdown peaks the breachers read. Getting the SET of tables
wrong is silent and only shows up as a strategy that re-trips immediately on resume, so the choice
is named once here instead of being spelled out at each call site.

Why rebasing is required at all: after a drawdown trip flattens a strategy to cash, its equity is
below the pre-loss high-water mark, so every subsequent tick re-trips against that stale peak (#27).

Which table, per stage: a LIVE strategy's drawdown breaker reads the NAV peak (`live_nav_peaks`),
NOT the paper account-equity peak. Clearing the wrong one leaves a resumed live strategy re-tripping
on a stale pre-breach NAV peak (codex C1 review) — which is exactly the bug this split prevents.

WHAT THIS MODULE DOES NOT COVER: the ordering invariant. Peaks must be re-based FIRST and the
un-halt (`kill_switch.reset` / `global_halt.clear`) written LAST, so that any earlier failure leaves
the strategy safely halted and the operation stays retryable (#109). A function cannot enforce the
order of a write it does not perform, so that ordering stays at the call sites and is commented
there. Do not assume calling these covers it.

WHY THIS LIVES IN `execution` AND NOT `risk` (the spec said `risk/peaks.py`): the per-strategy
peak tables it clears live in `algua.execution.order_state`, and that module imports
`algua.live.paper_loop`. Placed under `risk/`, this policy therefore made
`risk -> execution -> live` reachable — measured with a real import-linter probe, not guessed —
which inverts the layering the rest of this stage went to some trouble to preserve (`risk` sits
BELOW `live`; see `algua/risk/book_cycle.py` for the body that stayed in `risk` precisely because
it needed nothing from the live lane). Sitting beside the state it mutates costs no new edge
direction: `algua.execution` already imports `algua.risk` (`fleet_health.py`), so the single
`risk.book_equity` import below runs with the existing grain rather than against it.
"""

from __future__ import annotations

import sqlite3

from algua.contracts.lifecycle import Stage
from algua.execution.order_state import (
    clear_all_nav_peaks,
    clear_all_peaks,
    clear_nav_peak,
    clear_peak_equity,
)
from algua.risk.book_equity import clear_book_peak


def rebase_strategy_peak(conn: sqlite3.Connection, name: str, stage: Stage) -> None:
    """Re-base ONE strategy's drawdown peak, picking the table its breaker actually reads."""
    if stage is Stage.LIVE:
        clear_nav_peak(conn, name)
    else:
        clear_peak_equity(conn, name)


def rebase_all_peaks(conn: sqlite3.Connection) -> None:
    """Re-base every drawdown peak the account-wide resume must clear.

    All three tables, because a resumed account can re-trip through any of them: the paper
    (account-equity) peaks, the live (NAV) peaks, and the ACCOUNT-WIDE book high-water mark (#390) —
    after a flatten-to-cash the book breaker must not re-trip against the pre-loss peak. The
    book's daily-loss baseline needs no clearing; it auto-re-bases next session from the broker's
    prior-session close.
    """
    clear_all_peaks(conn)
    clear_all_nav_peaks(conn)
    clear_book_peak(conn)
