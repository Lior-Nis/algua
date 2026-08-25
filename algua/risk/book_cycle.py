"""Per-cycle evaluation of the account-wide book-level loss/drawdown circuit breaker (#390).

Belongs in `algua.risk`, not `algua.live`: `risk` sits BELOW `live` in the layering
(`algua/live/*` imports `algua.risk.limits`; `algua/risk/*` imports nothing from `algua.live`), and
this body needs none of the live-lane marks wall (`assert_marks_usable`, `_latest_bar_ts`,
`_latest_marks` in `algua/live/live_loop.py`) — only the account-snapshot + peak-ratchet machinery
that already lives in `algua.risk`. Filing it under `live` would put it above where its
dependencies (`algua.risk.book_breaker`, `algua.risk.book_equity`) actually sit for no reason;
filing the OTHER book body (`build_book_exposure`, which does need the live-lane marks wall) here
would invert the layering and create a package cycle — see `algua.live.book_exposure` for that one.
"""

from __future__ import annotations

import math

from algua.config.settings import get_settings
from algua.execution.errors import BrokerError
from algua.risk.book_breaker import BookBreach, BookBreakerLimits, evaluate_book_breaker
from algua.risk.book_equity import update_book_peak


def evaluate_book_loss_breaker(conn, broker):
    """Evaluate the book-level loss/drawdown circuit breaker (#390) for the whole account.

    Reads ONE ``broker.account()`` snapshot (equity + prior-session close). Returns a ``BookBreach``
    to halt+flatten, or None to proceed. Equity is validated BEFORE the high-water mark is ratcheted
    so a non-finite/non-positive read can never corrupt the peak (GATE-1): an unusable equity
    short-circuits to a breach without touching the peak. Otherwise the peak ratchets to include
    this cycle (a fresh all-time high => zero drawdown), and the daily-loss baseline is the broker's
    prior trading-session close (``account.last_equity``).

    A BrokerError reading / parsing the account (missing or malformed equity / last_equity) is
    itself a fail-closed breach: without a trustworthy account snapshot the book is unvaluable, so
    it must engage the persistent halt rather than fall through to a retryable JSON error (GATE-2).
    """
    try:
        account = broker.account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
    except BrokerError as exc:
        return BookBreach(
            "book_account_read_failed",
            f"could not read a trustworthy account snapshot for the book breaker ({exc}) — "
            "refusing to trade the shared book blind",
        )
    limits = BookBreakerLimits(
        max_drawdown=get_settings().book_max_drawdown,
        max_daily_loss=get_settings().book_max_daily_loss,
    )
    if not math.isfinite(equity) or equity <= 0.0:
        # Do NOT ratchet the peak on an unusable read; evaluate_book_breaker returns the
        # book_equity_unusable breach (peak value is irrelevant on this branch).
        return evaluate_book_breaker(equity, 0.0, last_equity, limits)
    peak = update_book_peak(conn, equity)
    return evaluate_book_breaker(equity, peak, last_equity, limits)
