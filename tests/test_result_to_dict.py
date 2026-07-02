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


def test_backtest_result_to_dict_excludes_returns():
    r = _backtest_result()
    d = r.to_dict()
    # to_dict() pops 'returns' because pd.Series is not JSON-serializable
    assert "returns" not in d
    expected = dataclasses.asdict(r)
    expected.pop("returns", None)
    assert d == expected


def test_walkforward_result_to_dict_matches_asdict():
    # holdout_returns is SENSITIVE and excluded from to_dict() (#221 Slice 1).
    # market_returns is bulky (full-period aggregate) and excluded from to_dict() (#221 Slice 4).
    # Strip both from the dataclasses.asdict reference so the invariant holds.
    r = _walkforward_result()
    expected = dataclasses.asdict(r)
    expected.pop("holdout_returns", None)
    expected.pop("market_returns", None)
    assert r.to_dict() == expected


def test_sweep_result_to_dict_matches_asdict():
    r = _sweep_result()
    assert r.to_dict() == dataclasses.asdict(r)


def test_backtest_result_to_dict_has_expected_keys():
    d = _backtest_result().to_dict()
    # 'returns' is excluded from to_dict() (not JSON-serializable)
    assert set(d) == {
        "strategy", "metrics", "config_hash", "data_source", "timeframe",
        "period", "seed", "snapshot_id", "code_hash", "dependency_hash",
        "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
        "model_ref", "delisting_snapshot", "forced_exits",
    }


def test_walkforward_result_to_dict_has_expected_keys():
    d = _walkforward_result().to_dict()
    assert set(d) == {
        "strategy", "config_hash", "data_source", "snapshot_id", "timeframe",
        "seed", "period", "windows", "holdout_frac", "embargo", "window_metrics",
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
        "trial_sharpe_count", "trial_sharpe_mean", "trial_sharpe_var_ann",
    }
