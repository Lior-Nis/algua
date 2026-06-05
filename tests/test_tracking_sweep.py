import math

from algua.backtest.sweep import SweepResult
from algua.tracking.mlflow_tracker import log_sweep


def _combo(lookback, score):
    return {
        "params": {"lookback": lookback, "top_k": 1}, "config_hash": f"h{lookback}",
        "n_windows": 4,
        "stability": {"mean_sharpe": score, "std_sharpe": 0.2, "min_sharpe": score - 0.3,
                      "pct_positive_windows": 0.75},
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


def test_sweep_n_combos_is_param_not_metric(tmp_path):
    """n_combos is a sweep config count — must be logged as a param, never as a metric."""
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_sweep(_sweep(), tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    parents = [r for r in client.search_runs([exp.experiment_id])
               if r.data.tags.get("kind") == "sweep"]
    assert len(parents) == 1
    assert parents[0].data.params["n_combos"] == "2"
    assert "n_combos" not in parents[0].data.metrics


def test_sweep_nonfinite_best_score_not_logged(tmp_path):
    """Non-finite best['score'] must not reach MLflow (log_metric would reject it)."""
    from mlflow.tracking import MlflowClient

    sweep = SweepResult(
        strategy="ew_inf_best", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        grid={"lookback": [20], "top_k": [1]}, n_combos=1, rank_by="mean_sharpe",
        ranked=[_combo(20, 1.4)],
        best={"params": {"lookback": 20, "top_k": 1}, "score": float("nan")},
    )
    uri = str(tmp_path / "mlruns")
    log_sweep(sweep, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew_inf_best")
    parents = [r for r in client.search_runs([exp.experiment_id])
               if r.data.tags.get("kind") == "sweep"]
    assert len(parents) == 1
    assert "best_score" not in parents[0].data.metrics


def test_sweep_nonfinite_best_score_finite_still_logged(tmp_path):
    """Finite best['score'] must still be logged after the guard is applied."""
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_sweep(_sweep(), tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    parents = [r for r in client.search_runs([exp.experiment_id])
               if r.data.tags.get("kind") == "sweep"]
    assert len(parents) == 1
    assert abs(parents[0].data.metrics["best_score"] - 1.4) < 1e-9


def test_sweep_nonfinite_entry_score_not_logged(tmp_path):
    """Non-finite entry['score'] in a ranked combo must not reach the child MLflow run."""
    from mlflow.tracking import MlflowClient

    nan_score_combo = {
        "params": {"lookback": 20, "top_k": 1}, "config_hash": "hnan",
        "n_windows": 4,
        "stability": {"mean_sharpe": 0.8, "std_sharpe": 0.1, "min_sharpe": 0.5,
                      "pct_positive_windows": 0.75},
        "score": float("nan"),
    }
    sweep = SweepResult(
        strategy="ew_nan_score", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        grid={"lookback": [20], "top_k": [1]}, n_combos=1, rank_by="mean_sharpe",
        ranked=[nan_score_combo],
        best={"params": {"lookback": 20, "top_k": 1}, "score": 0.8},
    )
    uri = str(tmp_path / "mlruns")
    log_sweep(sweep, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew_nan_score")
    children = [r for r in client.search_runs([exp.experiment_id])
                if r.data.tags.get("kind") == "sweep_combo"]
    assert len(children) == 1
    assert "score" not in children[0].data.metrics


def test_sweep_finite_entry_score_still_logged(tmp_path):
    """Finite entry['score'] must still reach the child MLflow run after the guard."""
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_sweep(_sweep(), tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    children = [r for r in client.search_runs([exp.experiment_id])
                if r.data.tags.get("kind") == "sweep_combo"]
    assert len(children) == 2
    for child in children:
        assert "score" in child.data.metrics
        assert math.isfinite(child.data.metrics["score"])


def test_sweep_drops_nonfinite_child_metrics(tmp_path):
    """NaN/inf in combo stability metrics must not reach MLflow child runs."""
    from mlflow.tracking import MlflowClient

    bad_combo = {
        "params": {"lookback": 20, "top_k": 1}, "config_hash": "hbad",
        "n_windows": 4,
        "stability": {
            "mean_sharpe": float("nan"), "std_sharpe": float("inf"),
            "min_sharpe": -0.3, "pct_positive_windows": 0.5,
        },
        "score": 0.5,
    }
    sweep = SweepResult(
        strategy="ew_bad", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        grid={"lookback": [20], "top_k": [1]}, n_combos=1, rank_by="mean_sharpe",
        ranked=[bad_combo], best={"params": {"lookback": 20, "top_k": 1}, "score": 0.5},
    )
    uri = str(tmp_path / "mlruns")
    log_sweep(sweep, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew_bad")
    children = [r for r in client.search_runs([exp.experiment_id])
                if r.data.tags.get("kind") == "sweep_combo"]
    assert len(children) == 1
    for v in children[0].data.metrics.values():
        assert math.isfinite(v), f"Non-finite metric found: {v}"
