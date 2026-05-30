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


def test_t_plus_1_blocks_same_bar_fill():
    """Guards the t->t+1 rule: a position entered (decided) at bar t must fill at t+1,
    so it cannot capture a close[t]->close[t+1] jump. The strategy decides to go long
    while price is still 100 and HOLDS; the price then jumps 100->150 from bar 5 to bar 6.
    Honest (lag=1) fills at 150 and earns ~0; a broken/removed shift would fill at 100 and
    capture +50%, failing the assertion."""

    class JumpProvider:
        seed = 0

        def get_bars(self, symbols, start, end, timeframe):
            ts = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")
            path = [100.0] * 6 + [150.0] * 6  # flat 100 through bar 5, jump to 150 at bar 6
            rows = [{"timestamp": t, "symbol": "AAA", "open": px, "high": px, "low": px,
                     "close": px, "adj_close": px, "volume": 1.0}
                    for t, px in zip(ts, path, strict=True)]
            return pd.DataFrame(rows).set_index("timestamp").sort_index()

    cfg = StrategyConfig(
        name="holder", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )

    def holder(view, params):
        # Deterministically go long once we have >=6 bars of history (decision at bar index 5,
        # price still 100), and HOLD from then on.
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) >= 6:
            return pd.Series([1.0], index=["AAA"])
        return pd.Series(dtype="float64")

    res = run(LoadedStrategy(config=cfg, fn=holder), JumpProvider(),
              datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC))
    # Entered at t+1 (price 150), held flat at 150 -> earns ~0 from the jump it sat through.
    assert res.metrics["total_return"] < 0.01
    # It genuinely held a position (not vacuous).
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
