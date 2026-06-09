import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _cfg(needs):
    return StrategyConfig(
        name="x", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, needs_fundamentals=needs,
    )


def test_config_defaults_no_fundamentals():
    assert _cfg(False).needs_fundamentals is False


def test_dispatch_passes_fundamentals_when_declared():
    seen = {}

    def fund_fn(view, params, fundamentals):
        seen["f"] = fundamentals
        return pd.Series(dtype="float64")

    ls = LoadedStrategy(config=_cfg(True), fundamentals_fn=fund_fn)
    frame = pd.DataFrame({"k": [1]})
    ls.target_weights(pd.DataFrame(), frame)
    assert seen["f"] is frame


def test_dispatch_plain_when_not_declared():
    def plain(view, params):
        return pd.Series([1.0], index=["AAPL"])

    ls = LoadedStrategy(config=_cfg(False), fn=plain)
    out = ls.target_weights(pd.DataFrame())
    assert out["AAPL"] == 1.0


def test_post_init_requires_matching_fn():
    with pytest.raises(ValueError, match="needs_fundamentals"):
        LoadedStrategy(config=_cfg(True), fn=lambda v, p: None)  # missing fundamentals_fn


def test_config_hash_includes_needs_fundamentals():
    a = LoadedStrategy(config=_cfg(False), fn=lambda v, p: None)
    b = LoadedStrategy(config=_cfg(True), fundamentals_fn=lambda v, p, f: None)
    assert config_hash(a) != config_hash(b)
