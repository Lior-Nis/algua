import math

from algua.backtest.walkforward import WalkForwardResult
from algua.tracking.mlflow_tracker import log_walk_forward


def _wf():
    return WalkForwardResult(
        strategy="ew", config_hash="abc", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2,
        window_metrics=[{"index": 0, "start": "2022-01-03", "end": "2022-06-01", "n_bars": 100,
                         "total_return": 0.1, "ann_return": 0.2, "ann_volatility": 0.15,
                         "sharpe": 1.3, "max_drawdown": -0.05}],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": 120,
                         "total_return": 0.05, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": 0.8, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.1, "std_sharpe": 0.3, "min_sharpe": 0.7,
                   "pct_positive_windows": 0.75},
    )


def test_log_walk_forward_records_metrics(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_walk_forward(_wf(), {"lookback": 60}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    m = runs[0].data.metrics
    assert abs(m["mean_sharpe"] - 1.1) < 1e-9
    # The OOS holdout is WITHHELD from walk-forward tracking (reserved for `research promote`):
    # no holdout.* metric is logged.
    assert not any(k.startswith("holdout.") for k in m)
    assert runs[0].data.tags["kind"] == "walk_forward"
    assert runs[0].data.params["windows"] == "4"


def test_walk_forward_drops_nonfinite_metrics(tmp_path):
    """NaN/inf stability or holdout metrics must not reach MLflow."""
    from mlflow.tracking import MlflowClient

    wf = WalkForwardResult(
        strategy="ew_bad", config_hash="x", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={"n_bars": 50, "sharpe": float("nan"), "total_return": 0.01,
                         "ann_return": 0.02, "ann_volatility": 0.1, "max_drawdown": -0.05},
        stability={"mean_sharpe": float("inf"), "std_sharpe": 0.2,
                   "min_sharpe": float("-inf"), "pct_positive_windows": 0.5},
    )
    uri = str(tmp_path / "mlruns")
    log_walk_forward(wf, {}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew_bad")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    for v in runs[0].data.metrics.values():
        assert math.isfinite(v), f"Non-finite metric found: {v}"
