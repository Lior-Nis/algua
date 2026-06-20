"""Tests for #221 Slice 4 Task 1: WalkForwardResult.market_returns benchmark series."""
from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import WalkForwardResult, walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _equal_weight():
    from algua.portfolio.construction import get_construction_policy

    cfg = StrategyConfig(
        name="ew2", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_market_returns_excluded_from_to_dict():
    """market_returns must be excluded from to_dict() (bulky internal field)."""
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3},
        stability={},
        market_returns=([0.001, -0.002], ["2020-12-30", "2020-12-31"]))
    assert "market_returns" not in wf.to_dict()
    assert wf.market_returns is not None


def test_market_returns_defaults_to_none():
    """market_returns defaults to None and is excluded from to_dict()."""
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3},
        stability={},
    )
    assert wf.market_returns is None
    assert "market_returns" not in wf.to_dict()


def test_to_dict_still_excludes_holdout_returns():
    """Extending to_dict exclusion must not drop holdout_returns exclusion."""
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3},
        stability={},
        holdout_returns=([0.1], ["2020-12-31"]),
        market_returns=([0.001], ["2020-12-30"]),
    )
    d = wf.to_dict()
    assert "holdout_returns" not in d
    assert "market_returns" not in d


def test_walk_forward_populates_market_returns():
    """walk_forward populates market_returns with full-period equal-weighted daily returns."""
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=42), START, END,
                       windows=4, holdout_frac=0.2)
    assert res.market_returns is not None
    mr, md = res.market_returns
    # Same lengths
    assert len(mr) == len(md) > 0
    # All ISO date strings YYYY-MM-DD
    assert all(isinstance(d, str) and len(d) == 10 for d in md)
    # Full-period: market series spans the whole simulation (longer than holdout alone)
    assert len(md) >= res.holdout_metrics["n_bars"]
    # Dates are sorted (monotonically increasing)
    assert md == sorted(md)
    # All returns are finite floats
    import math
    assert all(math.isfinite(r) for r in mr)


def test_market_returns_not_in_to_dict_after_walk_forward():
    """walk_forward result to_dict() must not expose market_returns."""
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=7), START, END,
                       windows=4, holdout_frac=0.2)
    d = res.to_dict()
    assert "market_returns" not in d


def test_market_returns_equal_weighted_two_symbols():
    """For a 2-symbol universe, market return at each date ≈ mean of the two symbols' pct change.

    SyntheticProvider is deterministic: we can reconstruct the per-symbol adj_close series and
    verify the equal-weighted cross-section matches market_returns within float tolerance.
    """
    strategy = _equal_weight()
    provider = SyntheticProvider(seed=99)
    res = walk_forward(strategy, provider, START, END, windows=4, holdout_frac=0.2)
    assert res.market_returns is not None
    mr, md = res.market_returns

    # Reconstruct from the provider (same immutable snapshot, same seed)
    bars = provider.get_bars(["AAA", "BBB"], START, END, "1d")
    adj = (
        bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )
    xs = adj.pct_change().mean(axis=1).dropna()
    expected_dates = [str(idx.date()) for idx in xs.index]
    expected_returns = [float(x) for x in xs.to_numpy()]

    assert md == expected_dates
    assert len(mr) == len(expected_returns)
    for actual, expected in zip(mr, expected_returns, strict=True):
        assert abs(actual - expected) < 1e-10
