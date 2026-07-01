import math

from algua.backtest.result import BacktestResult
from algua.tracking.mlflow_tracker import (
    ExperimentTracker,
    _flatten,
    _numeric_metrics,
    log_backtest,
)


def test_flatten_nested():
    assert _flatten({"a": 1, "b": {"c": 2, "d": 3}}) == {"a": 1, "b.c": 2, "b.d": 3}


# ---------------------------------------------------------------------------
# _numeric_metrics — #46
# ---------------------------------------------------------------------------

def test_numeric_metrics_drops_non_finite():
    d = {
        "sharpe": 1.5,
        "bad_nan": float("nan"),
        "bad_inf": float("inf"),
        "bad_neginf": float("-inf"),
    }
    result = _numeric_metrics(d)
    assert "sharpe" in result
    assert "bad_nan" not in result
    assert "bad_inf" not in result
    assert "bad_neginf" not in result
    assert all(math.isfinite(v) for v in result.values())


def test_numeric_metrics_drops_bool_and_str():
    d = {"ok": 2.0, "flag": True, "label": "abc", "none_val": None}
    result = _numeric_metrics(d)
    assert result == {"ok": 2.0}


def test_numeric_metrics_integer_count_survives():
    """n_rebalances (int) is a metric and must survive when finite."""
    d = {"sharpe": 1.1, "n_rebalances": 7}
    result = _numeric_metrics(d)
    assert result["n_rebalances"] == 7.0


# ---------------------------------------------------------------------------
# ExperimentTracker protocol — #45
# ---------------------------------------------------------------------------

def test_experiment_tracker_protocol_is_structural():
    """Any object with the three log_* methods satisfies ExperimentTracker structurally."""
    # Protocol is importable and has the expected attributes
    for method in ("log_backtest", "log_sweep", "log_walk_forward"):
        assert hasattr(ExperimentTracker, method)

    class Fake:
        def log_backtest(self, result, params, *, tracking_uri):
            return "run-1"

        def log_sweep(self, result, *, tracking_uri):
            return "run-2"

        def log_walk_forward(self, result, params, *, tracking_uri):
            return "run-3"

    # Structural check: all methods present
    for method in ("log_backtest", "log_sweep", "log_walk_forward"):
        assert hasattr(Fake(), method)


# ---------------------------------------------------------------------------
# log_backtest integration — existing + non-finite guard
# ---------------------------------------------------------------------------

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


def test_log_backtest_drops_nan_metrics(tmp_path):
    """Degenerate metrics (NaN/inf) must not reach MLflow."""
    from mlflow.tracking import MlflowClient

    result = BacktestResult(
        strategy="ew_nan",
        metrics={"sharpe": float("nan"), "cagr": float("inf"), "n_rebalances": 0},
        config_hash="x", data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2022-01-01", "end": "2022-12-31"}, seed=0, snapshot_id=None,
    )
    uri = str(tmp_path / "mlruns")
    log_backtest(result, {}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew_nan")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    # non-finite metrics must be absent
    assert "sharpe" not in runs[0].data.metrics
    assert "cagr" not in runs[0].data.metrics
    # finite int metric must still be present
    assert "n_rebalances" in runs[0].data.metrics


def test_log_backtest_stamps_universe_mode(tmp_path):
    # #333: a named-universe run logs universe_mode="pit" + the name; a static-universe run
    # (universe_name is None) logs universe_mode="static". The mode enum is a controlled value
    # that can never be confused with a real universe name.
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    named = BacktestResult(
        strategy="named", metrics={"sharpe": 1.0}, config_hash="c", data_source="Synthetic",
        timeframe="1d", period={"start": "2022-01-01", "end": "2022-12-31"}, seed=0,
        snapshot_id=None, universe_name="liquid10",
    )
    log_backtest(named, {}, tracking_uri=uri)
    log_backtest(_result(), {}, tracking_uri=uri)  # _result() is static (universe_name=None)

    client = MlflowClient(tracking_uri=uri)
    named_run = client.search_runs(
        [client.get_experiment_by_name("named").experiment_id])[0]
    assert named_run.data.params["universe_mode"] == "pit"
    assert named_run.data.params["universe_name"] == "liquid10"
    static_run = client.search_runs(
        [client.get_experiment_by_name("ew").experiment_id])[0]
    assert static_run.data.params["universe_mode"] == "static"


def test_log_backtest_n_rebalances_is_metric_not_param(tmp_path):
    """n_rebalances comes from metrics dict and must land in run metrics, not params."""
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_backtest(_result(), {"lookback": 60}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    run = client.search_runs([exp.experiment_id])[0]
    assert "n_rebalances" in run.data.metrics
    assert run.data.metrics["n_rebalances"] == 7.0
