"""CLI seam tests for the governance inventory (issue #393) — registry binding + gate-eval
integrity + fail-closed overdue exit code."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.registry.db import connect, migrate

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))


def _add(name: str):
    r = runner.invoke(app, ["registry", "add", name])
    assert r.exit_code == 0, r.stdout


def _scaffold(tmp_path, name: str):
    d = tmp_path / "vault" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(f"---\nname: {name}\nstage: idea\n---\n# {name}\n")


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_record_and_list(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    out = _json(runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior",
        "--assumption", "liquid", "--limitation", "regime shift",
        "--validation-summary", "wf pass", "--next-review", "2099-01-01",
        "--last-validated", "2026-01-01",
    ]))
    assert out["owner"] == "lior"
    assert out["overdue"] is False
    listed = _json(runner.invoke(app, ["governance", "list"]))
    assert isinstance(listed, list)
    assert listed[0]["name"] == "alpha"
    assert listed[0]["owner"] == "lior"


def test_record_phantom_strategy_is_not_found(tmp_path):
    # No registry row => must refuse, never fabricate governance for a phantom.
    r = runner.invoke(app, [
        "governance", "record", "ghost", "--owner", "lior", "--next-review", "2099-01-01",
    ])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["code"] == "not_found"


def test_record_registered_but_no_doc_is_file_not_found(tmp_path):
    _add("alpha")  # registered, but no kb doc scaffolded
    r = runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
    ])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["code"] == "file_not_found"


def test_gate_eval_id_must_belong_to_strategy(tmp_path):
    _add("alpha")
    _add("beta")
    _scaffold(tmp_path, "alpha")
    # Seed a gate_evaluations row owned by BETA, then try to cite it from ALPHA.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    beta_id = conn.execute("SELECT id FROM strategies WHERE name='beta'").fetchone()["id"]
    conn.execute(
        "INSERT INTO gate_evaluations (strategy_id, passed, n_funnel, own_lifetime_combos, "
        "windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok, holdout_n_bars, "
        "min_holdout_observations, code_hash, config_hash, data_source, period_start, period_end, "
        "holdout_frac, actor, decision_json, created_at) VALUES "
        "(?,1,1,1,1,90,'measured',1,63,63,'c','c','demo','s','e',0.3,'agent','{}','t')",
        (beta_id,),
    )
    conn.commit()
    gid = conn.execute("SELECT id FROM gate_evaluations").fetchone()["id"]
    conn.close()
    r = runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
        "--gate-eval-id", str(gid),
    ])
    assert r.exit_code == 1
    body = json.loads(r.stdout)
    assert body["code"] == "invalid_input"
    assert "belongs to" in body["error"]


def test_gate_eval_id_nonexistent_is_not_found(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    r = runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
        "--gate-eval-id", "9999",
    ])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["code"] == "not_found"


def test_overdue_exits_nonzero_when_any_overdue(tmp_path):
    # A registered strategy with NO governance record => overdue => nonzero exit (the monitor hook).
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    r = runner.invoke(app, ["governance", "overdue"])
    assert r.exit_code == 1
    body = json.loads(r.stdout)
    assert body["overdue_count"] == 1
    assert body["overdue"][0]["name"] == "alpha"
    assert body["overdue"][0]["present"] is False


def test_overdue_exits_zero_when_all_current(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
    ])
    r = runner.invoke(app, ["governance", "overdue"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["overdue_count"] == 0


def test_overdue_past_next_review_flags(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2000-01-01",
    ])
    r = runner.invoke(app, ["governance", "overdue"])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["overdue"][0]["name"] == "alpha"


def test_list_registry_driven_ignores_orphan_doc(tmp_path):
    # A vault doc without a registry row must NOT appear (list is registry-driven).
    _scaffold(tmp_path, "orphan")
    listed = _json(runner.invoke(app, ["governance", "list"]))
    assert listed == []


def _seed_gate(tmp_path, strategy_name: str, passed: int) -> int:
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    sid = conn.execute(
        "SELECT id FROM strategies WHERE name=?", (strategy_name,)
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO gate_evaluations (strategy_id, passed, n_funnel, own_lifetime_combos, "
        "windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok, holdout_n_bars, "
        "min_holdout_observations, code_hash, config_hash, data_source, period_start, period_end, "
        "holdout_frac, actor, decision_json, created_at) VALUES "
        "(?,?,1,1,1,90,'measured',1,63,63,'c','c','demo','s','e',0.3,'agent','{}','t')",
        (sid, passed),
    )
    conn.commit()
    gid = conn.execute("SELECT MAX(id) AS m FROM gate_evaluations").fetchone()["m"]
    conn.close()
    return gid


def test_record_rejects_failed_gate_eval(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    gid = _seed_gate(tmp_path, "alpha", passed=0)
    r = runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
        "--gate-eval-id", str(gid),
    ])
    assert r.exit_code == 1
    assert "did not pass" in json.loads(r.stdout)["error"]


def test_list_flags_valid_gate_eval(tmp_path):
    _add("alpha")
    _scaffold(tmp_path, "alpha")
    gid = _seed_gate(tmp_path, "alpha", passed=1)
    runner.invoke(app, [
        "governance", "record", "alpha", "--owner", "lior", "--next-review", "2099-01-01",
        "--gate-eval-id", str(gid),
    ])
    listed = _json(runner.invoke(app, ["governance", "list"]))
    assert listed[0]["gate_eval_valid"] is True
    assert listed[0]["overdue"] is False


def test_stale_gate_eval_id_in_doc_reads_fail_closed(tmp_path):
    # A hand-edited doc citing a phantom gate id must read as overdue (unverifiable evidence),
    # even with a future next-review date.
    _add("alpha")
    d = tmp_path / "vault" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    (d / "alpha.md").write_text(
        "---\nname: alpha\ngovernance_owner: lior\n"
        "governance_next_review: 2099-01-01\ngovernance_gate_eval_id: 9999\n---\nbody\n"
    )
    listed = _json(runner.invoke(app, ["governance", "list"]))
    assert listed[0]["gate_eval_valid"] is False
    assert listed[0]["overdue"] is True
    r = runner.invoke(app, ["governance", "overdue"])
    assert r.exit_code == 1


def test_malformed_gate_id_in_doc_reads_fail_closed(tmp_path):
    # A present-but-malformed cited gate id (not a positive int) is unverifiable evidence and must
    # read fail-closed overdue, not as 'no citation', even with a future next-review date.
    _add("alpha")
    d = tmp_path / "vault" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    (d / "alpha.md").write_text(
        "---\nname: alpha\ngovernance_owner: lior\n"
        "governance_next_review: 2099-01-01\ngovernance_gate_eval_id: abc\n---\nbody\n"
    )
    listed = _json(runner.invoke(app, ["governance", "list"]))
    assert listed[0]["gate_eval_valid"] is False
    assert listed[0]["overdue"] is True
    assert runner.invoke(app, ["governance", "overdue"]).exit_code == 1
