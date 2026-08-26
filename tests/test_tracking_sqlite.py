"""Tests for the SQLite-backed MLflow tracker (mlflow filestore deprecation, #605)."""

from algua.backtest.result import BacktestResult
from algua.backtest.sweep import SweepResult
from algua.backtest.walkforward import WalkForwardResult
from algua.tracking.base import ExperimentTracker
from algua.tracking.sqlite_tracker import SqliteMlflowTracker, _sqlite_tracking_uri

# ---------------------------------------------------------------------------
# _sqlite_tracking_uri — the bare-path -> sqlite:/// adaptation
# ---------------------------------------------------------------------------

def test_bare_path_becomes_sqlite_uri():
    assert _sqlite_tracking_uri("mlruns") == "sqlite:///mlruns.db"


def test_bare_path_with_db_suffix_is_not_double_suffixed():
    assert _sqlite_tracking_uri("mlflow.db") == "sqlite:///mlflow.db"


def test_already_schemed_uri_is_passed_through_unchanged():
    assert _sqlite_tracking_uri("sqlite:///custom.db") == "sqlite:///custom.db"
    assert _sqlite_tracking_uri("postgresql://host/db") == "postgresql://host/db"
    assert _sqlite_tracking_uri("http://mlflow-server:5000") == "http://mlflow-server:5000"


def test_bare_nested_path_becomes_sqlite_uri(tmp_path):
    target = tmp_path / "mlruns"
    assert _sqlite_tracking_uri(str(target)) == f"sqlite:///{target}.db"


# ---------------------------------------------------------------------------
# Selectable the same way as "mlflow" / "noop"
# ---------------------------------------------------------------------------

def test_registered_under_factory_and_satisfies_protocol():
    from algua.tracking.factory import get_tracker

    tracker = get_tracker("mlflow-sqlite")
    assert isinstance(tracker, SqliteMlflowTracker)
    assert isinstance(tracker, ExperimentTracker)
    for method in ("log_backtest", "log_sweep", "log_walk_forward"):
        assert callable(getattr(tracker, method))


# ---------------------------------------------------------------------------
# Integration: logging actually lands in a sqlite-backed MLflow store, not FileStore
# ---------------------------------------------------------------------------

def _backtest_result():
    return BacktestResult(
        strategy="ew_sqlite", metrics={"sharpe": 1.25, "cagr": 0.2, "n_rebalances": 7},
        config_hash="abc123", data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2022-01-01", "end": "2023-12-31"}, seed=0, snapshot_id=None,
    )


def test_log_backtest_writes_to_sqlite_db_file(tmp_path):
    from mlflow.tracking import MlflowClient

    bare_uri = str(tmp_path / "mlruns")
    tracker = SqliteMlflowTracker()
    run_id = tracker.log_backtest(_backtest_result(), {"lookback": 60}, tracking_uri=bare_uri)

    db_path = tmp_path / "mlruns.db"
    assert db_path.exists(), "expected a sqlite db file, not a FileStore directory"
    assert not (tmp_path / "mlruns").exists(), "must not fall back to the deprecated FileStore"

    client = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    exp = client.get_experiment_by_name("ew_sqlite")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_id == run_id
    assert abs(runs[0].data.metrics["sharpe"] - 1.25) < 1e-9


def test_log_backtest_respects_explicit_sqlite_uri(tmp_path):
    """An operator-supplied sqlite:// URI is honoured verbatim, not re-adapted."""
    from mlflow.tracking import MlflowClient

    db_path = tmp_path / "explicit.db"
    explicit_uri = f"sqlite:///{db_path}"
    tracker = SqliteMlflowTracker()
    tracker.log_backtest(_backtest_result(), {}, tracking_uri=explicit_uri)

    assert db_path.exists()
    client = MlflowClient(tracking_uri=explicit_uri)
    assert client.get_experiment_by_name("ew_sqlite") is not None


def test_log_sweep_via_sqlite_tracker(tmp_path):
    from mlflow.tracking import MlflowClient

    sweep = SweepResult(
        strategy="sweep_sqlite", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2, grid={"lookback": [20, 40], "top_k": [1]}, n_combos=2,
        rank_by="mean_sharpe",
        ranked=[
            {"params": {"lookback": 20, "top_k": 1}, "config_hash": "h20", "n_windows": 4,
             "stability": {"mean_sharpe": 1.4, "std_sharpe": 0.2, "min_sharpe": 1.1,
                           "pct_positive_windows": 0.75}, "score": 1.4},
        ],
        best={"params": {"lookback": 20, "top_k": 1}, "score": 1.4},
    )
    bare_uri = str(tmp_path / "mlruns")
    tracker = SqliteMlflowTracker()
    parent_id = tracker.log_sweep(sweep, tracking_uri=bare_uri)

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}")
    exp = client.get_experiment_by_name("sweep_sqlite")
    runs = client.search_runs([exp.experiment_id])
    parents = [r for r in runs if r.data.tags.get("kind") == "sweep"]
    assert len(parents) == 1 and parents[0].info.run_id == parent_id


def test_log_walk_forward_via_sqlite_tracker(tmp_path):
    from mlflow.tracking import MlflowClient

    wf = WalkForwardResult(
        strategy="wf_sqlite", config_hash="abc", data_source="SyntheticProvider",
        snapshot_id=None, timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        window_metrics=[{"index": 0, "start": "2022-01-03", "end": "2022-06-01", "n_bars": 100,
                         "total_return": 0.1, "ann_return": 0.2, "ann_volatility": 0.15,
                         "sharpe": 1.3, "max_drawdown": -0.05}],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": 120,
                         "total_return": 0.05, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": 0.8, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.1, "std_sharpe": 0.3, "min_sharpe": 0.7,
                   "pct_positive_windows": 0.75},
    )
    bare_uri = str(tmp_path / "mlruns")
    tracker = SqliteMlflowTracker()
    tracker.log_walk_forward(wf, {"lookback": 60}, tracking_uri=bare_uri)

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlruns.db'}")
    exp = client.get_experiment_by_name("wf_sqlite")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    assert abs(runs[0].data.metrics["mean_sharpe"] - 1.1) < 1e-9
