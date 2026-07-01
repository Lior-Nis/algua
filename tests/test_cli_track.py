import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))


def _runs(tmp_path, experiment="cross_sectional_momentum"):
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    exp = client.get_experiment_by_name(experiment)
    return [] if exp is None else client.search_runs([exp.experiment_id])


def test_run_track_logs_a_run(tmp_path):
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31", "--track"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["mlflow_run_id"]
    assert len(_runs(tmp_path)) == 1


def test_run_without_track_logs_nothing(tmp_path):
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    assert "mlflow_run_id" not in json.loads(result.stdout)
    assert _runs(tmp_path) == []


def test_sweep_track_logs_parent_and_children(tmp_path):
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--track"])
    assert result.exit_code == 0, result.stdout
    runs = _runs(tmp_path)
    assert sum(1 for r in runs if r.data.tags.get("kind") == "sweep") == 1
    assert sum(1 for r in runs if r.data.tags.get("kind") == "sweep_combo") == 2


def test_track_failure_is_non_fatal(tmp_path, monkeypatch):
    """A tracker failure must NOT discard a completed backtest (#341): the command still exits 0
    with the full result, surfacing the failure as a non-fatal `mlflow_tracking_error`."""
    import algua.cli.backtest_cmd as bt

    def boom(*a, **k):
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(bt, "log_backtest", boom)
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31", "--track"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["mlflow_run_id"] is None
    assert "RuntimeError: mlflow down" in payload["mlflow_tracking_error"]
