from algua.backtest.sweep import SweepResult
from algua.tracking.mlflow_tracker import log_sweep


def _combo(lookback, score):
    return {
        "params": {"lookback": lookback, "top_k": 1}, "config_hash": f"h{lookback}",
        "n_windows": 4,
        "stability": {"mean_sharpe": score, "std_sharpe": 0.2, "min_sharpe": score - 0.3,
                      "pct_positive_windows": 0.75},
        "holdout": {"n_bars": 100, "sharpe": 0.5, "total_return": 0.04, "max_drawdown": -0.06},
        "score": score,
    }


def _sweep():
    return SweepResult(
        strategy="ew", data_source="SyntheticProvider", snapshot_id=None, timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        grid={"lookback": [20, 40], "top_k": [1]}, n_combos=2, rank_by="mean_sharpe",
        ranked=[_combo(20, 1.4), _combo(40, 1.1)],
        best={"params": {"lookback": 20, "top_k": 1}, "score": 1.4},
    )


def test_log_sweep_parent_and_children(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    parent_id = log_sweep(_sweep(), tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    runs = client.search_runs([exp.experiment_id])
    parents = [r for r in runs if r.data.tags.get("kind") == "sweep"]
    children = [r for r in runs if r.data.tags.get("kind") == "sweep_combo"]
    assert len(parents) == 1 and parents[0].info.run_id == parent_id
    assert parents[0].data.params["n_combos"] == "2"
    assert abs(parents[0].data.metrics["best_score"] - 1.4) < 1e-9
    assert len(children) == 2
    for child in children:
        assert child.data.tags["mlflow.parentRunId"] == parent_id
        assert "score" in child.data.metrics
        assert "param.lookback" in child.data.params


def test_sweep_child_runs_carry_shared_stamps(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_sweep(_sweep(), tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    children = [r for r in client.search_runs([exp.experiment_id])
                if r.data.tags.get("kind") == "sweep_combo"]
    assert children
    for c in children:
        assert c.data.params["timeframe"] == "1d"
        assert c.data.params["windows"] == "4"
        assert "snapshot_id" in c.data.params
