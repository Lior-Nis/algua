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


def test_doctor_advisory_rows_do_not_gate_exit(monkeypatch, tmp_path):
    # No paper creds and no bars snapshot in a clean env -> advisory rows fail, but doctor still
    # passes (exit 0, ok=True) because advisory rows carry required=False.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ALGUA_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_API_SECRET", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    rows = {c["check"]: c for c in payload["checks"]}
    for name in ("paper_credentials", "bars_snapshot", "generated_provenance"):
        assert rows[name]["required"] is False
    assert rows["paper_credentials"]["ok"] is False
    assert rows["bars_snapshot"]["ok"] is False


def test_doctor_clean_env_safety_rows_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    rows = {c["check"]: c for c in payload["checks"]}
    for name in ("global_halt", "kill_switches", "live_authorizations"):
        assert rows[name]["required"] is False
        assert rows[name]["ok"] is True


def test_doctor_flags_engaged_global_halt(monkeypatch, tmp_path):
    from algua.cli._common import registry_conn
    from algua.risk import global_halt

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with registry_conn() as conn:
        global_halt.engage(conn, reason="test halt", actor="human")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0  # advisory, does not gate
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    gh = {c["check"]: c for c in payload["checks"]}["global_halt"]
    assert gh["ok"] is False
    assert "ENGAGED" in gh["detail"] and "test halt" in gh["detail"]


def test_doctor_flags_tripped_kill_switch(monkeypatch, tmp_path):
    from algua.cli._common import registry_conn
    from algua.risk import kill_switch

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with registry_conn() as conn:
        kill_switch.trip(conn, "alpha", reason="test kill", actor="human")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    ks = {c["check"]: c for c in payload["checks"]}["kill_switches"]
    assert ks["ok"] is False
    assert "alpha" in ks["detail"]


def test_doctor_no_live_strategies_authorizations_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    la = {c["check"]: c for c in payload["checks"]}["live_authorizations"]
    assert la["ok"] is True
    assert la["detail"] == "no live strategies"


def test_has_generated_by_recognizes_plain_and_annotated(tmp_path):
    from algua.cli.app import _has_generated_by

    plain = tmp_path / "plain.py"
    plain.write_text('GENERATED_BY = "agent"\n')
    annotated = tmp_path / "annotated.py"
    annotated.write_text('GENERATED_BY: str = "agent"\n')
    missing = tmp_path / "missing.py"
    missing.write_text("X = 1\n")
    bare = tmp_path / "bare.py"
    bare.write_text("GENERATED_BY: str\n")  # annotation only, assigns no marker value
    broken = tmp_path / "broken.py"
    broken.write_text("def (\n")  # unparseable -> False, never raises
    assert _has_generated_by(plain) is True
    assert _has_generated_by(annotated) is True
    assert _has_generated_by(missing) is False
    assert _has_generated_by(bare) is False
    assert _has_generated_by(broken) is False


def test_main_decorated_errors_exit_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    with pytest.raises(SystemExit) as exc:
        main(["registry", "show", "ghost"])

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "ghost" in payload["error"]
