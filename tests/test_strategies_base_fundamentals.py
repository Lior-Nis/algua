import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import equal_weight_positive
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _cfg(needs):
    return StrategyConfig(
        name="x", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="equal_weight_positive", needs_fundamentals=needs,
    )


def test_config_defaults_no_fundamentals():
    assert _cfg(False).needs_fundamentals is False


def test_dispatch_passes_fundamentals_when_declared():
    seen = {}

    def fund_fn(view, params, fundamentals):
        seen["f"] = fundamentals
        return pd.Series(dtype="float64")

    ls = LoadedStrategy(
        config=_cfg(True), fundamentals_signal_fn=fund_fn, construct_fn=equal_weight_positive
    )
    frame = pd.DataFrame({"k": [1]})
    ls.target_weights(pd.DataFrame(), frame)
    assert seen["f"] is frame


def test_dispatch_plain_when_not_declared():
    def plain(view, params):
        return pd.Series([1.0], index=["AAPL"])

    ls = LoadedStrategy(config=_cfg(False), signal_fn=plain, construct_fn=equal_weight_positive)
    out = ls.target_weights(pd.DataFrame())
    assert out["AAPL"] == 1.0


def test_post_init_requires_matching_fn():
    with pytest.raises(ValueError, match="needs_fundamentals"):
        # missing fundamentals_signal_fn
        LoadedStrategy(
            config=_cfg(True), signal_fn=lambda v, p: None, construct_fn=equal_weight_positive
        )


def test_config_hash_includes_needs_fundamentals():
    a = LoadedStrategy(
        config=_cfg(False), signal_fn=lambda v, p: None, construct_fn=equal_weight_positive
    )
    b = LoadedStrategy(
        config=_cfg(True), fundamentals_signal_fn=lambda v, p, f: None,
        construct_fn=equal_weight_positive,
    )
    assert config_hash(a) != config_hash(b)


def test_target_weights_fundamentals_lane_composes():
    import pandas as pd

    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import equal_weight_positive
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params, fundamentals):
        return pd.Series({"A": 1.0, "B": -1.0})

    cfg = StrategyConfig(
        name="f", universe=["A", "B"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="equal_weight_positive", needs_fundamentals=True,
    )
    loaded = LoadedStrategy(
        config=cfg, fundamentals_signal_fn=signal, construct_fn=equal_weight_positive
    )
    w = loaded.target_weights(pd.DataFrame(), pd.DataFrame())
    assert w.to_dict() == {"A": 1.0}
    with pytest.raises(ValueError):
        loaded.target_weights(pd.DataFrame())  # needs_fundamentals but no frame -> fail closed
