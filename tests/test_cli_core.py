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

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "Missing argument" in payload["error"]


def test_main_unknown_options_emit_json(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["version", "--wat"])

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "No such option" in payload["error"]


def test_doctor_reports_knowledge_base_check(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(app, ["doctor"])
    payload = json.loads(result.stdout)
    assert "knowledge_base" in {c["check"] for c in payload["checks"]}


def test_doctor_flags_missing_strategy_doc(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    runner.invoke(app, ["registry", "add", "alpha"])  # registry row, no doc
    result = runner.invoke(app, ["doctor"])
    payload = json.loads(result.stdout)
    kb = next(c for c in payload["checks"] if c["check"] == "knowledge_base")
    assert kb["ok"] is False
    assert "alpha" in kb["detail"]
    assert result.exit_code == 1


def test_main_decorated_errors_exit_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    with pytest.raises(SystemExit) as exc:
        main(["registry", "show", "ghost"])

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "ghost" in payload["error"]
