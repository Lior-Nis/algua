import pandas as pd
from mlflow.tracking import MlflowClient

from algua.backtest.result import BacktestResult
from algua.tracking import mlflow_tracker
from algua.tracking.mlflow_tracker import log_backtest


def _result(returns):
    return BacktestResult(
        strategy="mom_series", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-03"},
        seed=0, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def _artifact_names(uri, run_id):
    return {a.path for a in MlflowClient(tracking_uri=uri).list_artifacts(run_id)}


def test_log_backtest_logs_series_parquet(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    run_id = log_backtest(_result(pd.Series([0.01, -0.02, 0.0], index=idx)), {}, tracking_uri=uri)
    names = _artifact_names(uri, run_id)
    assert "series.parquet" in names
    local = MlflowClient(tracking_uri=uri).download_artifacts(run_id, "series.parquet")
    df = pd.read_parquet(local)
    assert list(df.columns) == ["date", "ret"] and len(df) == 3


def test_log_backtest_skips_series_when_returns_none(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    run_id = log_backtest(_result(None), {}, tracking_uri=uri)
    assert "series.parquet" not in _artifact_names(uri, run_id)


def test_only_log_backtest_emits_series_artifact():
    # Structural invariant: log_sweep and log_walk_forward must never reference series.parquet
    # (those functions contain the single-use OOS holdout tail and must never emit a series).
    assert "series.parquet" not in mlflow_tracker.log_sweep.__code__.co_consts
    assert "series.parquet" not in mlflow_tracker.log_walk_forward.__code__.co_consts
    # log_backtest must call the series emitter helper.
    assert "_log_series_artifact" in mlflow_tracker.log_backtest.__code__.co_names
