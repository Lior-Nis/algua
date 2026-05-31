from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import WalkForwardResult, walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _equal_weight():
    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(
        1.0 / len(v["symbol"].unique()), index=sorted(v["symbol"].unique())))


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
