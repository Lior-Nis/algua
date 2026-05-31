from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import SweepResult, sweep
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _momentum():
    from algua.features.indicators import momentum

    cfg = StrategyConfig(
        name="m", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 40, "top_k": 1},
    )

    def fn(view, params):
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) <= int(params["lookback"]):
            return pd.Series(dtype="float64")
        scores = momentum(wide, lookback=int(params["lookback"])).iloc[-1].dropna()
        winners = scores.sort_values(ascending=False).head(int(params["top_k"])).index
        if len(winners) == 0:
            return pd.Series(dtype="float64")
        return pd.Series(1.0 / len(winners), index=winners)

    return LoadedStrategy(config=cfg, fn=fn)


def test_sweep_ranks_and_counts():
    res = sweep(_momentum(), SyntheticProvider(seed=3), START, END,
                grid={"lookback": [20, 40], "top_k": [1, 2]}, windows=4, holdout_frac=0.2)
    assert isinstance(res, SweepResult)
    d = res.to_dict()
    assert d["n_combos"] == 4
    assert len(d["ranked"]) == 4
    scores = [r["score"] for r in d["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert d["ranked"][0]["score"] == d["best"]["score"]
    top = d["ranked"][0]
    assert "holdout" in top and "stability" in top
    assert top["score"] == top["stability"]["mean_sharpe"]
    assert set(top["params"]) == {"lookback", "top_k"}
    assert d["code_hash"] and d["dependency_hash"]


def test_sweep_is_deterministic():
    kw = dict(grid={"lookback": [20, 40], "top_k": [1, 2]}, windows=4, holdout_frac=0.2)
    a = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    b = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    assert a.to_dict() == b.to_dict()


def test_sweep_rejects_bad_rank_by():
    with pytest.raises(ValueError):
        sweep(_momentum(), SyntheticProvider(seed=3), START, END,
              grid={"lookback": [20, 40]}, rank_by="holdout_sharpe")


def test_sweep_records_windows_and_holdout_frac():
    res = sweep(_momentum(), SyntheticProvider(seed=3), START, END,
                grid={"lookback": [20, 40]}, windows=3, holdout_frac=0.25)
    d = res.to_dict()
    assert d["windows"] == 3
    assert d["holdout_frac"] == 0.25
