from datetime import UTC, datetime

import pandas as pd
import pytest

import algua.backtest.walkforward as wfmod
from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import WalkForwardResult, walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _equal_weight():
    from algua.portfolio.construction import get_construction_policy

    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
        construction="equal_weight_positive",
    )
    # A flat positive score on every symbol -> equal_weight_positive holds them all equal-weight,
    # reproducing the old direct equal-weight allocation.
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def _strategy(*, feature_lookback=None, decision_lag_bars=1):
    from algua.portfolio.construction import get_construction_policy
    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=decision_lag_bars),
        params={}, construction="equal_weight_positive", feature_lookback=feature_lookback,
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_resolve_embargo_explicit_arg_wins():
    # An explicit embargo (the exploratory --embargo override) beats any derivation.
    assert wfmod._resolve_embargo(_strategy(feature_lookback=60), 3) == 3
    assert wfmod._resolve_embargo(_strategy(feature_lookback=None), 0) == 0


def test_resolve_embargo_undeclared_is_zero():
    # Undeclared lookback -> legacy zero gap (the agent promote path refuses undeclared upstream).
    assert wfmod._resolve_embargo(_strategy(feature_lookback=None), None) == 0


def test_resolve_embargo_declared_is_max_lookback_lag():
    # Declared lookback dominates the decision lag...
    assert wfmod._resolve_embargo(_strategy(feature_lookback=60, decision_lag_bars=1), None) == 60
    # ...but a declared 0 (no rolling window) still floors at the t->t+1 decision lag.
    assert wfmod._resolve_embargo(_strategy(feature_lookback=0, decision_lag_bars=1), None) == 1
    assert wfmod._resolve_embargo(_strategy(feature_lookback=2, decision_lag_bars=5), None) == 5


def test_walk_forward_shape_and_stamps():
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    assert isinstance(res, WalkForwardResult)
    d = res.to_dict()
    assert d["windows"] == 4
    assert len(d["window_metrics"]) == 4
    assert {"start", "end", "n_bars", "sharpe", "total_return"} <= set(d["holdout_metrics"])
    stability_keys = {"mean_sharpe", "std_sharpe", "min_sharpe", "pct_positive_windows"}
    assert stability_keys == set(d["stability"])
    assert 0.0 <= d["stability"]["pct_positive_windows"] <= 1.0
    assert d["config_hash"] and d["data_source"] == "SyntheticProvider"
    assert d["code_hash"] and d["dependency_hash"]


def test_walk_forward_is_deterministic():
    a = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    b = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    assert a.to_dict() == b.to_dict()


def test_windows_and_holdout_cover_all_bars():
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    total = sum(w["n_bars"] for w in res.window_metrics) + res.holdout_metrics["n_bars"]
    assert total > 0
    assert res.holdout_metrics["n_bars"] > 0


def test_walk_forward_carries_timeframe_and_seed():
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    d = res.to_dict()
    assert d["timeframe"] == "1d"
    assert d["seed"] == 3


def test_on_peek_fires_before_holdout_eval(monkeypatch):
    # Spy on _segment_record to count holdout-metric evaluations.
    calls = []
    orig = wfmod._segment_record

    def spy(returns, s, e):
        calls.append((s, e))
        return orig(returns, s, e)

    monkeypatch.setattr(wfmod, "_segment_record", spy)

    def boom(_cfg_hash):
        raise RuntimeError("burn failed")

    with pytest.raises(RuntimeError, match="burn failed"):
        walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                     windows=4, holdout_frac=0.2, on_peek=boom)

    # on_peek raised before the holdout was evaluated: only the 4 in-sample windows were recorded,
    # NOT a 5th (holdout) evaluation. Proves the burn point is strictly before the peek.
    assert len(calls) == 4


def test_on_peek_receives_config_hash_and_completes(monkeypatch):
    seen = []
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2, on_peek=lambda cfg: seen.append(cfg))
    # Fired exactly once, with the same config_hash that lands in the result, and the run completed.
    assert seen == [res.config_hash]
    assert res.holdout_metrics  # the peek still happened on the success path


def test_walk_forward_without_on_peek_unchanged():
    # Default (on_peek=None) path is byte-identical to a second run.
    a = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    b = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END, on_peek=None)
    assert a.to_dict() == b.to_dict()
