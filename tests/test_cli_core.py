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


def test_doctor_clean_env_safety_rows(monkeypatch, tmp_path):
    # global_halt is REQUIRED (gates in every mode) but passes clean; kill_switches is advisory;
    # live_authorizations is a skip row (its body — a strategy-module import — is gated behind
    # --live). Every row carries a machine-legible `skipped` boolean.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    rows = {c["check"]: c for c in payload["checks"]}
    assert rows["global_halt"]["required"] is True
    assert rows["global_halt"]["ok"] is True and rows["global_halt"]["skipped"] is False
    assert rows["kill_switches"]["required"] is False
    assert rows["kill_switches"]["ok"] is True and rows["kill_switches"]["skipped"] is False
    la = rows["live_authorizations"]
    assert la["required"] is False and la["ok"] is True and la["skipped"] is True
    assert la["detail"].startswith("skipped:")


def test_doctor_flags_engaged_global_halt(monkeypatch, tmp_path):
    # An engaged global halt is a trading-readiness failure that GATES: exit 1, ok:false.
    from algua.cli._common import registry_conn
    from algua.risk import global_halt

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with registry_conn() as conn:
        global_halt.engage(conn, reason="test halt", actor="human")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    gh = {c["check"]: c for c in payload["checks"]}["global_halt"]
    assert gh["ok"] is False and gh["required"] is True
    assert "ENGAGED" in gh["detail"] and "test halt" in gh["detail"]


def test_doctor_flags_tripped_kill_switch(monkeypatch, tmp_path):
    from algua.cli._common import registry_conn
    from algua.risk import kill_switch

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with registry_conn() as conn:
        kill_switch.trip(conn, "alpha", reason="test kill", actor="human")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0  # advisory, never gates
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    ks = {c["check"]: c for c in payload["checks"]}["kill_switches"]
    assert ks["ok"] is False
    assert "alpha" in ks["detail"]


def test_doctor_default_does_not_run_live_authorizations_body(monkeypatch, tmp_path):
    # HIGH-2 regression guard: plain `doctor` must NEVER invoke verify_live_authorization (its
    # load_strategy side effect imports strategy modules). The row is an inert skip.
    import algua.registry.live_gate as live_gate

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    def _boom(*a, **k):
        raise AssertionError("verify_live_authorization must not run without --live")

    monkeypatch.setattr(live_gate, "verify_live_authorization", _boom)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    la = {c["check"]: c for c in json.loads(result.stdout)["checks"]}["live_authorizations"]
    assert la["skipped"] is True and la["ok"] is True and la["required"] is False


def test_doctor_live_flag_no_live_strategies_ready(monkeypatch, tmp_path):
    # Under --live with an empty live book: live_authorizations runs, is required, and passes
    # vacuously — detail surfaces live_strategies=0 (ready, not a bare green). Exit 0.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["doctor", "--live"])
    assert result.exit_code == 0
    la = {c["check"]: c for c in json.loads(result.stdout)["checks"]}["live_authorizations"]
    assert la["required"] is True and la["ok"] is True and la["skipped"] is False
    assert "live_strategies=0" in la["detail"]


def test_doctor_live_flag_gates_on_unverifiable_live_strategy(monkeypatch, tmp_path):
    # A Stage.LIVE strategy with no re-verifiable authorization -> live_authorizations red; under
    # --live it is required, so doctor exits 1 (the strict all-must-verify rule).
    from algua.cli._common import registry_conn
    from algua.registry.store import SqliteStrategyRepository

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    with registry_conn() as conn:
        SqliteStrategyRepository(conn).add("ghostlive")
        conn.execute("UPDATE strategies SET stage = 'live' WHERE name = 'ghostlive'")
        conn.commit()
    result = runner.invoke(app, ["doctor", "--live"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    la = {c["check"]: c for c in payload["checks"]}["live_authorizations"]
    assert la["ok"] is False and la["required"] is True and la["skipped"] is False
    assert "ghostlive" in la["detail"] and "live_strategies=1" in la["detail"]


def test_doctor_live_flag_help_disclaims_tradability():
    # The narrowed-claim contract (HIGH-3) lives in the flag help text an operator/agent reads.
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "authorization" in out and "fleet" in out


def test_doctor_paper_flag_gates_paper_credentials(monkeypatch, tmp_path):
    # --paper promotes the (default-advisory) paper_credentials probe to required, so absent creds
    # now gate: exit 1.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.delenv("ALGUA_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_API_SECRET", raising=False)
    result = runner.invoke(app, ["doctor", "--paper"])
    assert result.exit_code == 1
    pc = {c["check"]: c for c in json.loads(result.stdout)["checks"]}["paper_credentials"]
    assert pc["required"] is True and pc["ok"] is False


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
