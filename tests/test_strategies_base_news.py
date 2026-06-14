import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
    assert_tradable_without_news,
    config_hash,
)


def _cfg(**kw):
    base = dict(
        name="n", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="equal_weight_positive",
    )
    base.update(kw)
    return StrategyConfig(**base)


def _news_fn(view, params, news):
    return pd.Series(dtype="float64")


def _pol():
    return get_construction_policy("equal_weight_positive")


def test_needs_news_requires_news_signal_fn():
    with pytest.raises(ValueError, match="news_signal_fn"):
        LoadedStrategy(config=_cfg(needs_news=True), construct_fn=_pol())


def test_needs_news_and_needs_fundamentals_together_rejected():
    with pytest.raises(ValueError, match="both"):
        LoadedStrategy(config=_cfg(needs_news=True, needs_fundamentals=True),
                       news_signal_fn=_news_fn, construct_fn=_pol())


def test_stray_news_signal_fn_without_flag_rejected():
    with pytest.raises(ValueError):
        LoadedStrategy(config=_cfg(), news_signal_fn=_news_fn, construct_fn=_pol())


def test_signal_routes_news_and_fails_closed_without_it():
    s = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn, construct_fn=_pol())
    with pytest.raises(ValueError, match="news"):
        s.signal(pd.DataFrame())  # no news frame
    assert s.signal(pd.DataFrame(), news=pd.DataFrame()).empty


def test_news_strategy_rejects_fundamentals_sidecar():
    s = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn, construct_fn=_pol())
    with pytest.raises(ValueError):
        s.signal(pd.DataFrame(), fundamentals=pd.DataFrame())


def test_config_hash_changes_with_needs_news():
    a = LoadedStrategy(config=_cfg(), signal_fn=lambda v, p: pd.Series(dtype="float64"),
                       construct_fn=_pol())
    b = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn, construct_fn=_pol())
    assert config_hash(a) != config_hash(b)


def test_assert_tradable_without_news_raises():
    s = LoadedStrategy(config=_cfg(needs_news=True), news_signal_fn=_news_fn, construct_fn=_pol())
    with pytest.raises(ValueError, match="needs_news"):
        assert_tradable_without_news(s)


def test_fundamentals_strategy_rejects_news_sidecar():
    def _fund_fn(view, params, fundamentals):
        return pd.Series(dtype="float64")
    s = LoadedStrategy(config=_cfg(needs_fundamentals=True), fundamentals_signal_fn=_fund_fn,
                       construct_fn=_pol())
    with pytest.raises(ValueError):
        s.signal(pd.DataFrame(), news=pd.DataFrame())


def test_plain_strategy_rejects_any_sidecar():
    s = LoadedStrategy(config=_cfg(), signal_fn=lambda v, p: pd.Series(dtype="float64"),
                       construct_fn=_pol())
    with pytest.raises(ValueError):
        s.signal(pd.DataFrame(), news=pd.DataFrame())
    with pytest.raises(ValueError):
        s.signal(pd.DataFrame(), fundamentals=pd.DataFrame())
