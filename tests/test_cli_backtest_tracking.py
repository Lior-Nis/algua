"""MLflow tracking is a best-effort side effect (#341, item 1).

A tracker failure must NOT discard an already-completed backtest. These tests pin the three-state
JSON contract for `--track`: not-requested (no tracking keys), succeeded (`mlflow_run_id` set),
failed (`mlflow_run_id` null + `mlflow_tracking_error`, still exit 0). Covered for run,
walk-forward, and sweep.
"""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from algua.cli import backtest_cmd
from algua.cli.main import app

runner = CliRunner()
STRAT = "cross_sectional_momentum"
DEMO = ["--demo", "--start", "2022-01-01", "--end", "2023-12-31"]


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _payload(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _boom(*_a, **_k):
    raise RuntimeError("flaky uri")


# --- not requested: no tracking keys at all (distinct from a failed run) ---


def test_run_without_track_omits_tracking_keys():
    p = _payload(["backtest", "run", STRAT, *DEMO])
    assert "mlflow_run_id" not in p and "mlflow_tracking_error" not in p


# --- requested + succeeds: run id recorded, no error ---


def test_run_track_success_records_run_id(monkeypatch):
    monkeypatch.setattr(
        backtest_cmd, "get_tracker",
        lambda: SimpleNamespace(log_backtest=lambda *a, **k: "RUN123"))
    p = _payload(["backtest", "run", STRAT, *DEMO, "--track"])
    assert p["mlflow_run_id"] == "RUN123" and "mlflow_tracking_error" not in p


# --- requested + fails: non-fatal, result preserved, error surfaced with exception type ---
# (the run/log_backtest failure path is covered in test_cli_track.py; here we cover wf + sweep)


def test_walk_forward_track_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(
        backtest_cmd, "get_tracker", lambda: SimpleNamespace(log_walk_forward=_boom))
    p = _payload(["backtest", "walk-forward", STRAT, *DEMO, "--track"])
    assert p["ok"] is True
    assert p["mlflow_run_id"] is None
    assert "RuntimeError: flaky uri" in p["mlflow_tracking_error"]


def test_sweep_track_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(backtest_cmd, "get_tracker", lambda: SimpleNamespace(log_sweep=_boom))
    p = _payload(["backtest", "sweep", STRAT, *DEMO, "--param", "lookback=20,40", "--track"])
    assert p["ok"] is True
    assert p["mlflow_run_id"] is None
    assert "RuntimeError: flaky uri" in p["mlflow_tracking_error"]


# --- --summary mode must still surface the tracking error (it is in the keep-lists) ---


def test_walk_forward_summary_track_failure_keeps_error(monkeypatch):
    monkeypatch.setattr(
        backtest_cmd, "get_tracker", lambda: SimpleNamespace(log_walk_forward=_boom))
    p = _payload(["backtest", "walk-forward", STRAT, *DEMO, "--track", "--summary"])
    assert p["summary"] is True
    assert p["mlflow_run_id"] is None
    assert "RuntimeError: flaky uri" in p["mlflow_tracking_error"]


def test_sweep_summary_track_failure_keeps_error(monkeypatch):
    monkeypatch.setattr(backtest_cmd, "get_tracker", lambda: SimpleNamespace(log_sweep=_boom))
    p = _payload(
        ["backtest", "sweep", STRAT, *DEMO, "--param", "lookback=20,40", "--track", "--summary"])
    assert p["summary"] is True
    assert p["mlflow_run_id"] is None
    assert "RuntimeError: flaky uri" in p["mlflow_tracking_error"]


# --- no-op backend (stage 5a): honest "skipped", never a fabricated run id ---


def test_noop_backend_reports_skipped_not_success(monkeypatch):
    """A no-op backend must not fabricate a run id -- the payload says skipped, not succeeded."""
    monkeypatch.setenv("ALGUA_TRACKING_BACKEND", "noop")
    p = _payload(["backtest", "run", STRAT, *DEMO, "--track"])
    assert p["mlflow_run_id"] is None
    assert p["mlflow_tracking_skipped"] == "tracking backend logs nothing"
    assert "mlflow_tracking_error" not in p


def test_noop_backend_skipped_key_survives_summary(monkeypatch):
    """--summary must keep mlflow_tracking_skipped, or a skipped run looks like a bare null id
    (the exact ambiguity this fourth state exists to avoid)."""
    monkeypatch.setenv("ALGUA_TRACKING_BACKEND", "noop")
    p = _payload(["backtest", "walk-forward", STRAT, *DEMO, "--track", "--summary"])
    assert p["summary"] is True
    assert p["mlflow_run_id"] is None
    assert p["mlflow_tracking_skipped"] == "tracking backend logs nothing"
    assert "mlflow_tracking_error" not in p
