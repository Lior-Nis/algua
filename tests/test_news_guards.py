import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
    assert_tradable_without_news,
)
from algua.strategies.loader import load_strategy


def test_helper_allows_plain_strategy():
    assert_tradable_without_news(load_strategy("cross_sectional_momentum"))  # no raise


def test_helper_blocks_news_strategy():
    s = LoadedStrategy(
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
    with pytest.raises(ValueError, match="needs_news"):
        assert_tradable_without_news(s)
