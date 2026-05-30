import json

from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def test_version_emits_json():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "algua"


def test_doctor_passes_in_clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert {c["check"] for c in payload["checks"]} >= {"python", "registry_db", "calendar"}
