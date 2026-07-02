"""Book-level aggregate loss / drawdown circuit breaker — a pure risk helper (no I/O, stdlib).

Where `algua.risk.book_limits` caps the aggregate EXPOSURE (gross/net/concentration) of the shared
account at order-submit time, this module is the LOSS breaker: given the whole-account equity, an
account high-water mark, and the prior trading-session's closing equity, it decides whether the
book has drawn down or lost enough on the day to HALT and flatten the entire account (#390).

Institutional/prop practice hard-codes a daily-loss circuit breaker (~2-5%) that liquidates and
halts the whole book, precisely because per-strategy stops are insufficient under correlated
stress. This is that breaker, at the account level, composing with the existing `global_halt`.

Pure: the caller (`live run-all`) engages the halt and flattens on a returned `BookBreach`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(x: float) -> bool:
    return math.isfinite(x)


@dataclass(frozen=True)
class BookBreakerLimits:
    """Account-level loss caps as fractions of the reference equity (1.0 == 100%).

    - `max_drawdown`: fraction below the account high-water mark that trips the drawdown halt.
    - `max_daily_loss`: fraction below the prior trading-session close that trips the daily halt.

    Both are hard limits and must be finite and in (0, 1] — a non-positive or >1 cap is a
    misconfiguration that would silently disable the breaker, so it fails fast at construction."""

    max_drawdown: float
    max_daily_loss: float

    def __post_init__(self) -> None:
        for name, value in (("max_drawdown", self.max_drawdown),
                            ("max_daily_loss", self.max_daily_loss)):
            if not _finite(value) or not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name} must be a finite fraction in (0, 1]; got {value!r}"
                )


@dataclass(frozen=True)
class BookBreach:
    """A tripped book-level circuit breaker. `kind` is a stable machine tag; `detail` is the
    human-readable reason recorded in the audit row and the global-halt reason."""

    kind: str
    detail: str


def evaluate_book_breaker(
    equity: float, peak: float, last_equity: float, limits: BookBreakerLimits
) -> BookBreach | None:
    """Return a `BookBreach` if the whole-account book has breached a loss/drawdown cap, else None.

    FAIL CLOSED on unusable inputs: a non-finite / non-positive account `equity` means the book is
    unvaluable this cycle — refuse to trade rather than trade blind. Likewise a non-finite /
    non-positive `last_equity` means the daily-loss baseline (the broker's prior-session close)
    cannot be established, so the daily breaker cannot be evaluated — fail closed.

    - `equity`: current whole-account equity.
    - `peak`: the account high-water mark (already ratcheted to include this cycle by the caller);
      a fresh all-time high has `equity == peak` and therefore zero drawdown.
    - `last_equity`: the account equity at the PRIOR trading-session close (broker-supplied) — the
      exchange-session-correct start-of-day baseline that captures overnight/pre-market gaps.
    """
    if not _finite(equity) or equity <= 0.0:
        return BookBreach(
            "book_equity_unusable",
            f"account equity {equity!r} is not a usable (positive, finite) value — refusing to "
            "trade the shared book blind",
        )
    # Drawdown vs the account high-water mark.
    if peak > 0.0 and equity < peak * (1.0 - limits.max_drawdown):
        dd = 1.0 - (equity / peak)
        return BookBreach(
            "book_drawdown",
            f"book drawdown {dd:.4f} exceeds max_drawdown {limits.max_drawdown:.4f} "
            f"(equity {equity:.2f}, peak {peak:.2f})",
        )
    # Daily loss vs the prior trading-session close (start-of-day baseline).
    if not _finite(last_equity) or last_equity <= 0.0:
        return BookBreach(
            "book_baseline_unusable",
            f"prior-session close equity {last_equity!r} is not usable — cannot establish the "
            "daily-loss baseline; refusing to trade the shared book",
        )
    if equity < last_equity * (1.0 - limits.max_daily_loss):
        loss = 1.0 - (equity / last_equity)
        return BookBreach(
            "book_daily_loss",
            f"book daily loss {loss:.4f} exceeds max_daily_loss {limits.max_daily_loss:.4f} "
            f"(equity {equity:.2f}, prior-session close {last_equity:.2f})",
        )
    return None
