"""Per-cycle build of the account-level book-exposure accumulator (#389).

Belongs in `algua.live`, not `algua.risk`: `risk` sits BELOW `live` in the layering
(`algua/live/*` imports `algua.risk.limits`; `algua/risk/*` imports nothing from `algua.live`), and
this body NEEDS the live-lane marks wall — `assert_marks_usable`, `_latest_bar_ts`, `_latest_marks`
from `algua.live.live_loop` — to value the reconciled account positions. A body needing the
live-lane marks wall cannot live in `risk` without inverting that layering and creating a package
cycle. (The OTHER book body, `evaluate_book_loss_breaker`, needs none of the marks wall and lives
in `algua.risk.book_cycle` instead, one layer down.)
"""

from __future__ import annotations

from datetime import UTC, datetime

from algua.config.settings import get_settings
from algua.live.live_loop import _latest_bar_ts as _book_latest_bar_ts
from algua.live.live_loop import _latest_marks as _book_latest_marks
from algua.live.live_loop import assert_marks_usable
from algua.primitives.timeparse import utc
from algua.risk.book_limits import BookExposure, BookRiskLimits


def build_book_exposure(
    broker, provider, net_positions: dict[str, float], start: str, end: str,
    now: datetime | None = None,
) -> tuple[BookExposure | None, str | None]:
    """Build the account-level book-exposure accumulator (#389) that caps aggregate gross / net /
    single-name concentration across ALL strategies sharing the live account. Seeds from the
    RECONCILED broker net positions (whole-account truth, incl. non-ticked/dormant/orphan
    residuals) valued at latest closed-bar marks, against ACCOUNT equity (not a subaccount).

    Data-integrity failures (a stale / absent / future-dated / non-finite mark on a held name) go
    through the SHARED mark-freshness wall (`assert_marks_usable`, #452 HIGH#2) so the account book
    and the per-strategy valuation apply the EXACT SAME staleness math + thresholds. That wall
    raises `RiskBreach('stale_marks' | 'unvaluable_marks')`, which the run-all caller routes to a
    HALT-WITHOUT-FLATTEN (a dark feed, broker still alive) — NOT a benign defer.

    Returns (BookExposure, None) on success, or (None, reason) to BENIGNLY DEFER (caller skips
    trading this cycle) only for policy/economic states — NOT data failures:
      - any reconciled nonzero position with qty < 0 (a short — long-only precondition);
      - an account book that ALREADY breaches a book-level cap at reconcile time.
    `now` is injected (default `datetime.now(UTC)`) so the freshness wall is testable."""
    now = now or datetime.now(UTC)
    nonzero = {s: float(q) for s, q in net_positions.items() if float(q) != 0.0}
    shorts = sorted(s for s, q in nonzero.items() if q < 0.0)
    if shorts:
        return None, f"account holds short position(s) {shorts} — book-risk precondition is " \
                     "long-only; refusing to trade this cycle"
    # Marks for EVERY held symbol (the book is valued on the reconciled account positions; a
    # strategy's not-yet-held universe symbols carry no book notional, so they need no mark here).
    fetch = sorted(nonzero)
    bars = (provider.get_bars(fetch, utc(start), utc(end), "1d").sort_index()
            if fetch else None)
    # Null-preserving latest-row selection so the wall and the valuation read the SAME atomic row
    # and a NaN-latest close is not masked (shared with the per-strategy loop, #452).
    latest_ts = _book_latest_bar_ts(bars) if bars is not None else {}
    latest_close = _book_latest_marks(bars) if bars is not None else {}
    # Data-integrity wall BEFORE building notionals: a stale / absent / future-dated / non-finite
    # mark raises RiskBreach (dark feed) — the caller HALTS, never flattens.
    assert_marks_usable(sorted(nonzero), latest_ts, latest_close, now)
    book_notionals = {sym: qty * latest_close[sym] for sym, qty in nonzero.items()}
    equity = float(broker.account().equity)
    s = get_settings()
    limits = BookRiskLimits(
        max_gross=s.book_max_gross,
        max_net=s.book_max_net,
        max_symbol_concentration=s.book_max_symbol_concentration,
        max_symbol_notional=s.book_max_symbol_notional,
    )
    book = BookExposure(equity, book_notionals, limits)
    # An account book that ALREADY breaches a cap at reconcile time is an anomaly: the per-buy
    # monotone headroom guarantees no-worse but cannot heal an already-over OTHER symbol via a buy,
    # so trading through it is unsound (Codex #389 GATE-2). Fail closed — skip the whole cycle.
    breaches = book.seed_breaches()
    if breaches:
        return None, f"account book already breaches book-level cap(s) {breaches} at reconcile " \
                     "— refusing to trade this cycle"
    return book, None
