import json

import pytest
from typer.testing import CliRunner

from algua.audit.log import append
from algua.cli.main import app
from algua.registry.db import connect, migrate

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _seed(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", reason="2 orders", strategy="alpha")
    append(conn, actor="human", action="kill_switch_trip", reason="manual", strategy="beta")
    append(conn, actor="agent", action="flatten", strategy="alpha")
    conn.close()


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_audit_log_emits_bare_array_most_recent_first(tmp_path):
    _seed(tmp_path)
    out = _json(runner.invoke(app, ["audit", "log"]))
    # bare JSON array (like `registry list`), most-recent-first (id DESC)
    assert isinstance(out, list)
    assert [r["action"] for r in out] == ["flatten", "kill_switch_trip", "paper_run"]


def test_audit_log_strategy_filter(tmp_path):
    _seed(tmp_path)
    out = _json(runner.invoke(app, ["audit", "log", "--strategy", "alpha"]))
    assert {r["strategy"] for r in out} == {"alpha"}
    assert len(out) == 2


def test_audit_log_actor_and_action_filters(tmp_path):
    _seed(tmp_path)
    out = _json(runner.invoke(app, ["audit", "log", "--actor", "human", "--action",
                                    "kill_switch_trip"]))
    assert len(out) == 1
    assert out[0]["actor"] == "human"


def test_audit_log_limit_and_offset(tmp_path):
    _seed(tmp_path)
    page1 = _json(runner.invoke(app, ["audit", "log", "--limit", "1", "--offset", "0"]))
    page2 = _json(runner.invoke(app, ["audit", "log", "--limit", "1", "--offset", "1"]))
    assert len(page1) == 1 and len(page2) == 1
    assert page1[0]["id"] != page2[0]["id"]


def test_audit_log_all_returns_everything(tmp_path):
    _seed(tmp_path)
    out = _json(runner.invoke(app, ["audit", "log", "--all"]))
    assert len(out) == 3


def test_audit_log_all_with_explicit_limit_errors(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, ["audit", "log", "--all", "--limit", "5"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_audit_log_offset_aware_since_converts_to_utc(tmp_path):
    """An offset-aware --since is converted to the true UTC instant before filtering."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    # stored UTC rows one hour apart
    for ts in ("2026-01-01T10:00:00+00:00", "2026-01-01T12:00:00+00:00"):
        conn.execute("INSERT INTO audit_log(ts, actor, action) VALUES (?,?,?)",
                     (ts, "agent", "a"))
    conn.commit()
    conn.close()
    # 13:00+02:00 == 11:00 UTC -> excludes the 10:00 row, includes the 12:00 row
    out = _json(runner.invoke(app, ["audit", "log", "--since", "2026-01-01T13:00:00+02:00"]))
    assert [r["ts"] for r in out] == ["2026-01-01T12:00:00+00:00"]


def test_audit_log_naive_since_assumed_utc(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    for ts in ("2026-01-01T10:00:00+00:00", "2026-01-01T12:00:00+00:00"):
        conn.execute("INSERT INTO audit_log(ts, actor, action) VALUES (?,?,?)",
                     (ts, "agent", "a"))
    conn.commit()
    conn.close()
    out = _json(runner.invoke(app, ["audit", "log", "--since", "2026-01-01T11:00:00"]))
    assert [r["ts"] for r in out] == ["2026-01-01T12:00:00+00:00"]


def test_audit_log_invalid_timestamp_errors(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, ["audit", "log", "--since", "not-a-timestamp"])
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["ok"] is False


def test_audit_log_empty_trail_returns_empty_array(tmp_path):
    out = _json(runner.invoke(app, ["audit", "log"]))
    assert out == []
