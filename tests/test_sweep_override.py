import pandas as pd

from algua.backtest.sweep import _override
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import top_k_equal_weight
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _base():
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60}, construction="top_k_equal_weight",
        construction_params={"top_k": 3},
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=top_k_equal_weight,
    )


def test_override_merges_over_defaults():
    base = _base()
    out = _override(base, {"lookback": 20})
    assert out.config.params == {"lookback": 20}
    assert out.signal_fn is base.signal_fn
    assert out.name == "m"


def test_override_does_not_mutate_base():
    base = _base()
    _override(base, {"lookback": 20, "construction.top_k": 1})
    assert base.config.params == {"lookback": 60}  # unchanged
    assert base.config.construction_params == {"top_k": 3}  # unchanged


def test_override_preserves_signal_panel_fn():
    """The fast-path acceleration hook must survive a sweep combo rebuild — otherwise sweeps
    silently drop the fast path and re-incur the per-bar cost on every combo."""
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60}, construction="top_k_equal_weight",
        construction_params={"top_k": 3},
    )
    panel = lambda b, p: pd.DataFrame()  # noqa: E731
    base = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
        signal_panel_fn=panel, construct_fn=top_k_equal_weight,
    )
    out = _override(base, {"lookback": 20})
    assert out.signal_panel_fn is panel


def test_override_routes_construction_namespace():
    from algua.backtest.sweep import _override
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")  # construction top_k_equal_weight, top_k=3
    out = _override(s, {"construction.top_k": 5, "lookback": 30})
    assert out.config.construction_params["top_k"] == 5
    assert out.config.params["lookback"] == 30
    assert out.construct_fn is s.construct_fn
    assert out.signal_panel_fn is s.signal_panel_fn


def test_override_rejects_unknown_signal_key():
    import pytest

    from algua.backtest.sweep import _override
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    with pytest.raises(ValueError):
        _override(s, {"not_a_real_param": 1})  # non-prefixed key not in CONFIG.params


def test_override_revalidates_construction_params():
    import pytest

    from algua.backtest.sweep import _override
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    with pytest.raises(ValueError):
        _override(s, {"construction.top_k": 0})  # fails the policy validator
