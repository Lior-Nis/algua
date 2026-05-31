import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app, main

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


def test_main_usage_errors_emit_json(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["registry", "add"])

    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "Missing argument" in payload["error"]


def test_main_unknown_options_emit_json(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["version", "--wat"])

    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "No such option" in payload["error"]
