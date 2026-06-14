"""Tests for #51: to_dict uses dataclasses.asdict (no hand-listed field duplication)."""
import dataclasses

from algua.backtest.result import BacktestResult
from algua.backtest.sweep import SweepResult
from algua.backtest.walkforward import WalkForwardResult


def _backtest_result() -> BacktestResult:
    return BacktestResult(
        strategy="test",
        metrics={"sharpe": 1.2},
        config_hash="abc123",
        data_source="SyntheticProvider",
        timeframe="1d",
        period={"start": "2022-01-01", "end": "2023-12-31"},
        seed=42,
        snapshot_id=None,
        code_hash="deadbeef",
        dependency_hash="cafebabe",
    )


def _walkforward_result() -> WalkForwardResult:
    return WalkForwardResult(
        strategy="test",
        config_hash="abc123",
        data_source="SyntheticProvider",
        snapshot_id=None,
        timeframe="1d",
        seed=42,
        period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4,
        holdout_frac=0.2,
        window_metrics=[{"index": 0, "n_bars": 10, "sharpe": 1.0}],
        holdout_metrics={"n_bars": 5, "sharpe": 0.8},
        stability={"mean_sharpe": 1.0, "std_sharpe": 0.1, "min_sharpe": 0.8,
                   "pct_positive_windows": 1.0},
        code_hash="deadbeef",
        dependency_hash="cafebabe",
    )


def _sweep_result() -> SweepResult:
    return SweepResult(
        strategy="test",
        data_source="SyntheticProvider",
        snapshot_id=None,
        timeframe="1d",
        seed=42,
        period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4,
        holdout_frac=0.2,
        grid={"lookback": [20, 40]},
        n_combos=2,
        rank_by="mean_sharpe",
        ranked=[],
        best=None,
        code_hash="deadbeef",
        dependency_hash="cafebabe",
    )


def test_backtest_result_to_dict_matches_asdict():
    r = _backtest_result()
    assert r.to_dict() == dataclasses.asdict(r)


def test_walkforward_result_to_dict_matches_asdict():
    r = _walkforward_result()
    assert r.to_dict() == dataclasses.asdict(r)


def test_sweep_result_to_dict_matches_asdict():
    r = _sweep_result()
    assert r.to_dict() == dataclasses.asdict(r)


def test_backtest_result_to_dict_has_expected_keys():
    d = _backtest_result().to_dict()
    assert set(d) == {
        "strategy", "metrics", "config_hash", "data_source", "timeframe",
        "period", "seed", "snapshot_id", "code_hash", "dependency_hash",
        "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
    }


def test_walkforward_result_to_dict_has_expected_keys():
    d = _walkforward_result().to_dict()
    assert set(d) == {
        "strategy", "config_hash", "data_source", "snapshot_id", "timeframe",
        "seed", "period", "windows", "holdout_frac", "window_metrics",
        "holdout_metrics", "stability", "code_hash", "dependency_hash",
        "universe_name", "universe_snapshots",
        "fundamentals_snapshot", "news_snapshot",
    }


def test_sweep_result_to_dict_has_expected_keys():
    d = _sweep_result().to_dict()
    assert set(d) == {
        "strategy", "data_source", "snapshot_id", "timeframe", "seed",
        "period", "windows", "holdout_frac", "grid", "n_combos",
        "rank_by", "ranked", "best", "code_hash", "dependency_hash",
        "universe_name", "universe_snapshots",
        "fundamentals_snapshot", "news_snapshot",
    }
