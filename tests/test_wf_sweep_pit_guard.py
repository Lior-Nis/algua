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


def test_sweep_rejects_needs_news():
    with pytest.raises(BacktestError, match="not supported in sweep"):
        sweep(_news_strategy(), SyntheticProvider(seed=3), START, END, grid={"x": [1]})


def test_sweep_rejects_needs_fundamentals():
    with pytest.raises(BacktestError, match="not supported in sweep"):
        sweep(_fund_strategy(), SyntheticProvider(seed=3), START, END, grid={"x": [1]})


def test_override_preserves_news_signal_fn():
    s = _news_strategy()
    out = _override(s, {})
    assert out.news_signal_fn is s.news_signal_fn
