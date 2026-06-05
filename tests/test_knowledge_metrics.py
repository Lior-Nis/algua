from algua.knowledge.metrics import latest_run_metrics


def _log_run(uri, strategy, *, kind, metrics, params):
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(strategy)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"kind": kind})


def test_latest_run_metrics_none_when_no_experiment(tmp_path):
    uri = str(tmp_path / "mlruns")
    assert latest_run_metrics("ghost", tracking_uri=uri) is None


def test_latest_run_metrics_none_when_store_missing(tmp_path, monkeypatch):
    # A relative tracking uri with no store on disk (fresh checkout) must degrade to None,
    # not raise — otherwise `strategy doc` would crash before any tracked run exists.
    monkeypatch.chdir(tmp_path)
    assert latest_run_metrics("alpha", tracking_uri="mlruns") is None


def test_latest_run_metrics_reads_latest(tmp_path):
    uri = str(tmp_path / "mlruns")
    _log_run(uri, "alpha", kind="backtest",
             metrics={"sharpe": 0.4}, params={"snapshot_id": "ds_1", "seed": "7"})
    # Run includes holdout metrics (as logged before writers were sealed) plus a normal metric.
    _log_run(uri, "alpha", kind="walk_forward",
             metrics={"holdout.sharpe": 0.28, "sharpe": 0.35},
             params={"snapshot_id": "ds_2", "seed": "7"})

    out = latest_run_metrics("alpha", tracking_uri=uri)
    assert out is not None
    assert out["kind"] == "walk_forward"
    assert out["snapshot_id"] == "ds_2"
    assert out["seed"] == "7"
    # Pins the holdout seal at the knowledge surface: holdout.* keys must be withheld.
    assert "holdout.sharpe" not in out["metrics"]
    assert "holdout_metrics" not in out["metrics"]
    # Normal metrics must still be present.
    assert out["metrics"]["sharpe"] == 0.35
    assert isinstance(out["run_id"], str) and out["run_id"]
