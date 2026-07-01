"""CLI smoke tests for `algua monitoring drift` (issue #343)."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_drift_demo_emits_verdict():
    payload = _json(runner.invoke(app, [
        "monitoring", "drift", "cross_sectional_momentum", "--demo",
        "--start", "2023-01-01", "--end", "2023-12-31",
    ]))
    assert payload["ok"] is True
    assert payload["strategy"] == "cross_sectional_momentum"
    assert payload["verdict"] in ("ok", "warn", "alarm", "insufficient_data")
    for key in ("signal_distribution_psi", "turnover_drift", "coverage_drift"):
        assert key in payload["leading"]
    assert "ic_decay" in payload["corroborating"]
    assert isinstance(payload["limitations"], list) and payload["limitations"]


def test_drift_pinned_reference_end():
    payload = _json(runner.invoke(app, [
        "monitoring", "drift", "cross_sectional_momentum", "--demo",
        "--start", "2023-01-01", "--end", "2023-06-30", "--reference-end", "2023-03-31",
    ]))
    assert payload["reference"]["end"] < payload["recent"]["start"]


def test_drift_rejects_unknown_strategy():
    result = runner.invoke(app, ["monitoring", "drift", "nope", "--demo"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_drift_rejects_non_positive_horizon():
    result = runner.invoke(app, [
        "monitoring", "drift", "cross_sectional_momentum", "--demo", "--horizon", "0",
    ])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False
