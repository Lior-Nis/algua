"""Changing a declared ExecutionContract rail changes config_hash, so existing live
authorizations correctly invalidate and must re-sign (#135)."""
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _strategy(**execution_kw) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="s",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", **execution_kw),
        params={},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(dtype="float64"))


def test_max_weight_per_symbol_changes_config_hash():
    base = config_hash(_strategy())
    tighter = config_hash(_strategy(max_weight_per_symbol=0.2))
    assert base != tighter


def test_allow_short_changes_config_hash():
    base = config_hash(_strategy())
    shortable = config_hash(_strategy(allow_short=True))
    assert base != shortable
