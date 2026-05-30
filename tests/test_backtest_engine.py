from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError, run
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _equal_weight_strategy() -> LoadedStrategy:
    cfg = StrategyConfig(
        name="ew",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def fn(view: pd.DataFrame, params: dict) -> pd.Series:
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    return LoadedStrategy(config=cfg, fn=fn)


def test_run_produces_metrics_keys() -> None:
    res = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    for key in [
        "total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown",
        "turnover", "avg_gross_exposure", "n_rebalances",
    ]:
        assert key in res.metrics


def test_run_is_deterministic() -> None:
    a = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    b = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    assert a.metrics == b.metrics


def test_t_plus_1_blocks_same_bar_fill() -> None:
    """A 'cheating' strategy that, at bar t, puts 100% on whichever symbol rises at t
    must NOT capture bar t's move, because the engine shifts weights by decision_lag_bars."""
    cfg = StrategyConfig(
        name="cheat",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def cheat(view: pd.DataFrame, params: dict) -> pd.Series:
        wide = view.reset_index().pivot(
            index="timestamp", columns="symbol", values="adj_close"
        )
        if len(wide) < 2:
            return pd.Series(dtype="float64")
        last_ret = wide.iloc[-1] / wide.iloc[-2] - 1.0  # uses the CURRENT bar's return
        winner = last_ret.idxmax()
        return pd.Series([1.0], index=[winner])

    strat = LoadedStrategy(config=cfg, fn=cheat)
    cheating = run(strat, SyntheticProvider(seed=5), START, END)

    cfg0 = StrategyConfig(
        name="cheat0",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=0),
        params={},
    )
    leaky = run(LoadedStrategy(config=cfg0, fn=cheat), SyntheticProvider(seed=5), START, END)

    # Look-ahead (lag=0, same-bar fill) must look dramatically better than the honest t+1 version.
    assert leaky.metrics["total_return"] > cheating.metrics["total_return"] + 0.05


def test_empty_universe_data_raises() -> None:
    cfg = StrategyConfig(
        name="x",
        universe=[],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(dtype="float64"))
    with pytest.raises(BacktestError):
        run(strat, SyntheticProvider(), START, END)
