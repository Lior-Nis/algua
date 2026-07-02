"""Unit tests for the pure book-level loss/drawdown circuit breaker (#390)."""

from __future__ import annotations

import math

import pytest

from algua.risk.book_breaker import (
    BookBreach,
    BookBreakerLimits,
    evaluate_book_breaker,
)

LIMITS = BookBreakerLimits(max_drawdown=0.15, max_daily_loss=0.05)


def test_limits_validation_rejects_out_of_range() -> None:
    for bad in (0.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BookBreakerLimits(max_drawdown=bad, max_daily_loss=0.05)
        with pytest.raises(ValueError):
            BookBreakerLimits(max_drawdown=0.15, max_daily_loss=bad)


def test_limits_accept_boundary_one() -> None:
    lim = BookBreakerLimits(max_drawdown=1.0, max_daily_loss=1.0)
    assert lim.max_drawdown == 1.0


def test_clean_book_returns_none() -> None:
    # equity above both thresholds, below peak by less than max_drawdown.
    assert evaluate_book_breaker(96_000.0, 100_000.0, 98_000.0, LIMITS) is None


def test_fresh_all_time_high_has_zero_drawdown() -> None:
    # equity == peak: dd == 0, and equity == last_equity: no daily loss.
    assert evaluate_book_breaker(100_000.0, 100_000.0, 100_000.0, LIMITS) is None


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), 0.0, -1.0])
def test_unusable_equity_fails_closed(equity: float) -> None:
    breach = evaluate_book_breaker(equity, 100_000.0, 100_000.0, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_equity_unusable"


def test_drawdown_just_under_limit_passes() -> None:
    # exactly at the drawdown threshold is NOT a breach (strict <). last_equity == equity so the
    # daily-loss branch does not trip and we isolate the drawdown boundary.
    peak = 100_000.0
    equity = peak * (1.0 - LIMITS.max_drawdown)  # == 85_000
    assert evaluate_book_breaker(equity, peak, equity, LIMITS) is None


def test_drawdown_over_limit_trips() -> None:
    peak = 100_000.0
    equity = peak * (1.0 - LIMITS.max_drawdown) - 1.0  # just past
    # last_equity == equity so only the drawdown branch can trip.
    breach = evaluate_book_breaker(equity, peak, equity, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_drawdown"


def test_daily_loss_over_limit_trips() -> None:
    # peak high enough that drawdown does not trip first; daily baseline is today's open.
    last_equity = 100_000.0
    equity = last_equity * (1.0 - LIMITS.max_daily_loss) - 1.0  # just past 5% down
    breach = evaluate_book_breaker(equity, equity, last_equity, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_daily_loss"


def test_daily_loss_just_under_limit_passes() -> None:
    last_equity = 100_000.0
    equity = last_equity * (1.0 - LIMITS.max_daily_loss)  # exactly 5% down
    assert evaluate_book_breaker(equity, equity, last_equity, LIMITS) is None


@pytest.mark.parametrize("last_equity", [float("nan"), float("inf"), 0.0, -1.0])
def test_unusable_baseline_fails_closed(last_equity: float) -> None:
    # equity usable, drawdown clean, but the daily baseline can't be established.
    breach = evaluate_book_breaker(100_000.0, 100_000.0, last_equity, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_baseline_unusable"


def test_drawdown_checked_before_daily_loss() -> None:
    # both would trip; drawdown is the first check and wins.
    peak = 200_000.0
    last_equity = 100_000.0
    equity = 80_000.0  # 60% off peak AND 20% off prior close
    breach = evaluate_book_breaker(equity, peak, last_equity, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_drawdown"


def test_zero_peak_skips_drawdown_but_still_checks_daily() -> None:
    # peak == 0 (no high-water yet): drawdown branch skipped; daily-loss still evaluated.
    last_equity = 100_000.0
    equity = last_equity * (1.0 - LIMITS.max_daily_loss) - 1.0
    breach = evaluate_book_breaker(equity, 0.0, last_equity, LIMITS)
    assert isinstance(breach, BookBreach)
    assert breach.kind == "book_daily_loss"


def test_detail_is_populated() -> None:
    breach = evaluate_book_breaker(10.0, 100.0, 100.0, LIMITS)
    assert breach is not None
    assert math.isclose  # sanity import use
    assert "drawdown" in breach.detail
