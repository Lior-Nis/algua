from algua.backtest.result import BacktestResult
from algua.tracking.mlflow_tracker import _flatten, log_backtest


def test_flatten_nested():
    assert _flatten({"a": 1, "b": {"c": 2, "d": 3}}) == {"a": 1, "b.c": 2, "b.d": 3}


def _result():
    return BacktestResult(
        strategy="ew", metrics={"sharpe": 1.25, "cagr": 0.2, "n_rebalances": 7},
        config_hash="abc123", data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2022-01-01", "end": "2023-12-31"}, seed=0, snapshot_id=None,
    )


def test_log_backtest_records_run(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    run_id = log_backtest(_result(), {"lookback": 60, "top_k": 3}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == run_id
    assert abs(run.data.metrics["sharpe"] - 1.25) < 1e-9
    assert run.data.params["config_hash"] == "abc123"
    assert run.data.params["param.lookback"] == "60"
    assert run.data.tags["kind"] == "backtest"
