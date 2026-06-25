"""Smoke tests for `research family-audit` CLI command (#228)."""
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


def test_family_audit_empty_registry_returns_clean_json():
    """Empty registry: command exits 0, ok=True, empty clusters, telemetry keys present."""
    result = runner.invoke(app, ["research", "family-audit"])
    payload = _json(result)
    assert payload["ok"] is True
    assert payload["clusters"] == []
    assert "n_families_scanned" in payload
    assert "wall_time_seconds" in payload
