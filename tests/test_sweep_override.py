import pandas as pd

from algua.backtest.sweep import _override
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _base():
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60, "top_k": 3},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(dtype="float64"))


def test_override_merges_over_defaults():
    base = _base()
    out = _override(base, {"lookback": 20})
    assert out.config.params == {"lookback": 20, "top_k": 3}
    assert out.fn is base.fn
    assert out.name == "m"


def test_override_does_not_mutate_base():
    base = _base()
    _override(base, {"lookback": 20, "top_k": 1})
    assert base.config.params == {"lookback": 60, "top_k": 3}  # unchanged
