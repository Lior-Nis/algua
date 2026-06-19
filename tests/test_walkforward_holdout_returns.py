"""Tests for #221 Slice 1 Task 2: WalkForwardResult.holdout_returns SENSITIVE field."""
from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import WalkForwardResult, walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _minimal_wf(**over):
    base = dict(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3},
        stability={},
        holdout_returns=([0.1, -0.2, 0.05], ["2020-12-29", "2020-12-30", "2020-12-31"]),
    )
    base.update(over)
    return WalkForwardResult(**base)


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


def test_to_dict_excludes_holdout_returns():
    d = _minimal_wf().to_dict()
    assert "holdout_returns" not in d
    # the rest of the payload is unchanged
    assert d["holdout_metrics"] == {"n_bars": 3}


def test_holdout_returns_defaults_to_none():
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3}, stability={})
    assert wf.holdout_returns is None
    assert "holdout_returns" not in wf.to_dict()


def test_holdout_returns_accessible_on_dataclass():
    """holdout_returns is present on the dataclass even though to_dict excludes it."""
    wf = _minimal_wf()
    assert wf.holdout_returns is not None
    rets, dates = wf.holdout_returns
    assert rets == [0.1, -0.2, 0.05]
    assert dates == ["2020-12-29", "2020-12-30", "2020-12-31"]


def test_to_dict_does_not_include_holdout_returns_even_when_set():
    """Confirm to_dict strips holdout_returns regardless of its value."""
    wf = _minimal_wf(holdout_returns=([1.0, 2.0], ["2020-01-01", "2020-01-02"]))
    d = wf.to_dict()
    assert "holdout_returns" not in d


def test_walk_forward_populates_holdout_returns():
    """walk_forward sets holdout_returns with correct length and ISO date strings."""
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    assert res.holdout_returns is not None
    rets, dates = res.holdout_returns
    n = res.holdout_metrics["n_bars"]
    assert len(rets) == n
    assert len(dates) == n
    # Date alignment: first/last dates must match the holdout segment's start/end.
    assert dates[0] == res.holdout_metrics["start"]
    assert dates[-1] == res.holdout_metrics["end"]


def test_walk_forward_holdout_returns_dates_are_iso_strings():
    """All date strings in holdout_returns[1] must be valid ISO YYYY-MM-DD."""
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    assert res.holdout_returns is not None
    _rets, dates = res.holdout_returns
    for d in dates:
        assert isinstance(d, str)
        assert len(d) == 10
        # Must parse as a date — YYYY-MM-DD
        from datetime import date
        parsed = date.fromisoformat(d)
        assert str(parsed) == d


def test_walk_forward_holdout_returns_not_in_to_dict():
    """walk_forward result to_dict() must not expose the SENSITIVE field."""
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2)
    d = res.to_dict()
    assert "holdout_returns" not in d
