import pandas as pd
import pytest

from algua.backtest.sweep import _override
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import top_k_equal_weight
from algua.portfolio.overlays import OverlaySpec, trailing_stop
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


_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def _with_overlay():
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60}, construction="top_k_equal_weight",
        construction_params={"top_k": 3},
        overlays=[OverlaySpec(policy="trailing_stop", params=_TS)],
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=top_k_equal_weight, overlay_fns=(trailing_stop,),
    )


def test_override_tunes_overlay_params_and_keeps_fns():
    base = _with_overlay()
    out = _override(base, {"overlay.0.stop_pct": 0.2})
    assert out.config.overlays[0].params == {**_TS, "stop_pct": 0.2}
    assert out.overlay_fns == (trailing_stop,)
    assert base.config.overlays[0].params == _TS  # base untouched


def test_override_rejects_out_of_range_overlay_index():
    with pytest.raises(ValueError, match=r"overlay\.1\.stop_pct.*declares 1 overlay"):
        _override(_with_overlay(), {"overlay.1.stop_pct": 0.2})


def test_override_rejects_malformed_overlay_key():
    with pytest.raises(ValueError, match="expected overlay.<i>.<param>"):
        _override(_with_overlay(), {"overlay.stop_pct": 0.2})


def test_override_rejects_invalid_swept_overlay_param():
    with pytest.raises(ValueError, match="swept overlay params invalid"):
        _override(_with_overlay(), {"overlay.0.stop_pct": 1.5})


def test_override_rejects_overlay_window_exceeding_declared_lookback():
    base = _with_overlay()
    base = LoadedStrategy(
        config=base.config.model_copy(update={"feature_lookback": 7}),
        signal_fn=base.signal_fn, construct_fn=base.construct_fn, overlay_fns=base.overlay_fns,
    )
    with pytest.raises(ValueError, match="feature_lookback 7 is smaller"):
        _override(base, {"overlay.0.lookback": 10})
