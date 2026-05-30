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
    """With the mandatory lag>=1 rule, a strategy that chases a one-time jump can only
    enter AFTER the jump bar, so it cannot capture the jump's return."""

    class JumpProvider:
        seed = 0

        def get_bars(self, symbols, start, end, timeframe):
            ts = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")
            path = [100.0] * 5 + [150.0] * 7  # flat, +50% jump at bar 5, then flat
            rows = [{"timestamp": t, "symbol": "AAA", "open": px, "high": px, "low": px,
                     "close": px, "adj_close": px, "volume": 1.0}
                    for t, px in zip(ts, path, strict=True)]
            return pd.DataFrame(rows).set_index("timestamp").sort_index()

    cfg = StrategyConfig(
        name="chaser", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def chaser(view, params):
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) < 2:
            return pd.Series(dtype="float64")
        if wide["AAA"].iloc[-1] > wide["AAA"].iloc[-2]:  # only after seeing the jump
            return pd.Series([1.0], index=["AAA"])
        return pd.Series(dtype="float64")

    res = run(LoadedStrategy(config=cfg, fn=chaser), JumpProvider(),
              datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC))
    # Chases the jump but enters at t+1 (price already 150) -> earns ~0 from it.
    assert res.metrics["total_return"] < 0.01
    # And it genuinely DID take a position (the test isn't vacuously true).
    assert res.metrics["n_rebalances"] >= 1


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
