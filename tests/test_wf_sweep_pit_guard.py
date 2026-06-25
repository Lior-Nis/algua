"""Task 9 (#132): fail-closed guard rejecting PIT-sidecar strategies at the entry of
walk_forward and sweep (provider threading deferred), plus _override copying news_signal_fn."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.sweep import _override, sweep
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _news_strategy() -> LoadedStrategy:
    return LoadedStrategy(
        config=StrategyConfig(
            name="n",
            universe=["AAPL"],
            execution=ExecutionContract(rebalance_frequency="1d"),
            params={"window_days": 5},
            construction="equal_weight_positive",
            needs_news=True,
        ),
        news_signal_fn=lambda v, p, news: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy("equal_weight_positive"),
    )


def _fund_strategy() -> LoadedStrategy:
    return LoadedStrategy(
        config=StrategyConfig(
            name="f",
            universe=["AAPL"],
            execution=ExecutionContract(rebalance_frequency="1d"),
            params={"window_days": 5},
            construction="equal_weight_positive",
            needs_fundamentals=True,
        ),
        fundamentals_signal_fn=lambda v, p, f: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy("equal_weight_positive"),
    )


def test_walk_forward_without_provider_fails_closed_needs_news():
    with pytest.raises(BacktestError, match="needs_news"):
        walk_forward(_news_strategy(), SyntheticProvider(seed=3), START, END)


def test_walk_forward_without_provider_fails_closed_needs_fundamentals():
    with pytest.raises(BacktestError, match="needs_fundamentals"):
        walk_forward(_fund_strategy(), SyntheticProvider(seed=3), START, END)


def test_sweep_needs_news_without_provider_fails_closed():
    # Single-combo grid runs inline (no worker pickling), so the deep build_portfolio fail-closed
    # is the error that surfaces — not the parallel pickle preflight.
    with pytest.raises(BacktestError, match="needs_news"):
        sweep(_news_strategy(), SyntheticProvider(seed=3), START, END, grid={"window_days": [5]})


def test_sweep_needs_fundamentals_without_provider_fails_closed():
    with pytest.raises(BacktestError, match="needs_fundamentals"):
        sweep(_fund_strategy(), SyntheticProvider(seed=3), START, END,
              grid={"window_days": [5]})


def test_override_preserves_news_signal_fn():
    s = _news_strategy()
    out = _override(s, {})
    assert out.news_signal_fn is s.news_signal_fn


def test_override_preserves_fundamentals_signal_fn():
    s = _fund_strategy()
    out = _override(s, {})
    assert out.fundamentals_signal_fn is s.fundamentals_signal_fn
