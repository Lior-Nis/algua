"""Changing a declared ExecutionContract rail changes config_hash, so existing live
authorizations correctly invalidate and must re-sign (#135)."""
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _strategy(**execution_kw) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="s",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", **execution_kw),
        params={},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_max_weight_per_symbol_changes_config_hash():
    base = config_hash(_strategy())
    tighter = config_hash(_strategy(max_weight_per_symbol=0.2))
    assert base != tighter


def test_allow_short_changes_config_hash():
    base = config_hash(_strategy())
    shortable = config_hash(_strategy(allow_short=True))
    assert base != shortable


def _cfg(**over):
    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import StrategyConfig
    base = dict(
        name="s", universe=["A", "B"], execution=ExecutionContract(rebalance_frequency="1d"),
        params={"lookback": 10}, construction="top_k_equal_weight",
        construction_params={"top_k": 2},
    )
    base.update(over)
    return StrategyConfig(**base)


def test_config_hash_changes_with_construction_id():
    from algua.strategies.base import config_hash
    from algua.strategies.loader import _loaded_for_test  # see note below
    a = _loaded_for_test(_cfg())
    b = _loaded_for_test(_cfg(construction="score_proportional_long", construction_params={}))
    assert config_hash(a) != config_hash(b)


def test_config_hash_changes_with_construction_params():
    from algua.strategies.base import config_hash
    from algua.strategies.loader import _loaded_for_test
    a = _loaded_for_test(_cfg(construction_params={"top_k": 2}))
    b = _loaded_for_test(_cfg(construction_params={"top_k": 3}))
    assert config_hash(a) != config_hash(b)


def test_config_hash_changes_with_feature_lookback():
    # #345: feature_lookback sizes the walk-forward embargo (carves different windows), so two runs
    # that differ only in declared lookback must not collide on config_hash / artifact identity.
    from algua.strategies.base import config_hash
    from algua.strategies.loader import _loaded_for_test
    undeclared = config_hash(_loaded_for_test(_cfg()))                       # feature_lookback=None
    declared_0 = config_hash(_loaded_for_test(_cfg(feature_lookback=0)))
    declared_60 = config_hash(_loaded_for_test(_cfg(feature_lookback=60)))
    assert undeclared != declared_0 != declared_60
    assert undeclared != declared_60
