from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import holdout_window
from algua.backtest.walkforward import walk_forward
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
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_holdout_window_matches_walk_forward_burned_tail():
    # The linchpin: the reserved interval must equal what walk_forward actually burns.
    strat, prov = _equal_weight(), SyntheticProvider(seed=3)
    for frac in (0.2, 0.35):
        wf = walk_forward(strat, prov, START, END, windows=4, holdout_frac=frac)
        hs, he = holdout_window(strat, prov, START, END, holdout_frac=frac)
        assert (hs, he) == (wf.holdout_metrics["start"], wf.holdout_metrics["end"])


def test_holdout_end_is_last_actual_bar_not_period_end():
    strat, prov = _equal_weight(), SyntheticProvider(seed=1)
    hs, he = holdout_window(strat, prov, START, END, holdout_frac=0.2)
    bars = prov.get_bars(["AAA", "BBB"], START, END, "1d")
    last_session = bars.index.max().date().isoformat()
    assert he == last_session
    assert hs < he


def test_holdout_window_empty_bars_returns_conservative_period():
    strat = _equal_weight()

    class _Empty:
        def get_bars(self, symbols, start, end, timeframe):
            return SyntheticProvider().get_bars([], start, end, timeframe)

    hs, he = holdout_window(strat, _Empty(), START, END, holdout_frac=0.2)
    assert (hs, he) == (START.date().isoformat(), END.date().isoformat())


def test_holdout_window_tiny_frac_rounds_to_zero_returns_full_grid():
    strat, prov = _equal_weight(), SyntheticProvider(seed=2)
    bars = prov.get_bars(["AAA", "BBB"], START, END, "1d")
    hs, he = holdout_window(strat, prov, START, END, holdout_frac=1e-6)
    assert hs == bars.index.min().date().isoformat()
    assert he == bars.index.max().date().isoformat()
