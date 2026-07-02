"""Tests for `research run-all` — the long-lived research batch worker (#326).

Covers: multi-action batch in ONE warm process, per-task fault isolation (#374), the holdout
single-use guard surviving process reuse, the construction._POLICIES state-leak reset, malformed
tasks-file rejection, and the meaningful exit code.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _write_tasks(tmp_path, tasks) -> str:
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(tasks))
    return str(p)


def _run_all(tmp_path, tasks):
    return runner.invoke(app, ["research", "run-all", "--tasks", _write_tasks(tmp_path, tasks)])


def _seed_bars(tmp_path):
    """Seed ~330 calendar-day bars over AAPL/MSFT/NVDA so a 0.2 holdout clears the 63-obs floor.
    Returns (bars_snapshot_id, end_iso)."""
    store = DataStore(tmp_path)
    idx = pd.date_range("2025-01-01", periods=330, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "MSFT", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])
    end_iso = idx[-1].date().isoformat()
    rec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                            end=end_iso, as_of="2026-01-01T00:00:00Z", source="t", frame=bars)
    return rec.snapshot_id, end_iso


def _holdout_count(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0]
    finally:
        conn.close()


# --- happy path: multiple actions in one warm process --------------------------------------------

def test_multi_action_batch_runs_in_one_process(tmp_path):
    r = _run_all(tmp_path, [
        {"action": "backtest", "name": "cross_sectional_momentum", "demo": True},
        {"action": "sweep", "name": "cross_sectional_momentum", "demo": True,
         "param": ["lookback=20,40"], "top": 3},
    ])
    assert r.exit_code == 0, r.stdout
    env = json.loads(r.stdout)
    assert env["ok"] is True
    assert env["tasks_total"] == 2
    assert env["ok_count"] == 2 and env["error_count"] == 0
    assert [x["action"] for x in env["results"]] == ["backtest", "sweep"]
    assert all(x["ok"] for x in env["results"])
    # each result carries its own strategy payload (backtest metrics; sweep recorded breadth)
    assert "metrics" in env["results"][0]
    assert env["results"][1]["recorded_breadth"]["n_combos"] == 2


# --- fault isolation (#374): one bad task must not abort the batch --------------------------------

def test_bad_task_is_isolated_siblings_still_run(tmp_path):
    r = _run_all(tmp_path, [
        {"action": "backtest", "name": "does_not_exist", "demo": True},
        {"action": "backtest", "name": "cross_sectional_momentum", "demo": True},
    ])
    # a task erred -> envelope ok False + non-zero exit, but the sibling STILL ran
    assert r.exit_code == 1, r.stdout
    env = json.loads(r.stdout)
    assert env["ok"] is False
    assert env["error_count"] == 1 and env["ok_count"] == 1
    bad, good = env["results"]
    assert bad["ok"] is False and bad["kind"] == "task_error" and bad["code"] == "not_found"
    assert good["ok"] is True and good["name"] == "cross_sectional_momentum"


# --- holdout single-use guard survives process reuse ---------------------------------------------

def test_holdout_single_use_preserved_across_warm_batch(tmp_path):
    bid, end_iso = _seed_bars(tmp_path)
    # register -> backtested first (batch backtest task with --register)
    assert _run_all(tmp_path, [
        {"action": "backtest", "name": "cross_sectional_momentum", "snapshot": bid,
         "start": "2025-01-01", "end": end_iso, "register": True},
    ]).exit_code == 0
    relax = {"min_holdout_sharpe": -100.0, "min_holdout_return": -100.0, "min_pct_positive": 0.0,
             "min_window_sharpe": -100.0, "windows": 2, "n_combos": 9, "allow_non_pit": True,
             "actor": "human", "new_family": "csm", "snapshot": bid,
             "start": "2025-01-01", "end": end_iso}
    # TWO promote tasks on the SAME holdout window in ONE warm process: the 2nd must fail closed
    # exactly as two separate cold processes would (the burn is a committed DB row).
    r = _run_all(tmp_path, [
        {"action": "promote", "name": "cross_sectional_momentum", **relax},
        {"action": "promote", "name": "cross_sectional_momentum", **relax},
    ])
    env = json.loads(r.stdout)
    assert env["results"][0]["ok"] is True, env["results"][0]
    second = env["results"][1]
    assert second["ok"] is False, second  # single-use guard fired inside the warm process
    assert env["error_count"] == 1
    assert r.exit_code == 1
    # exactly ONE burn recorded despite two attempts in one process
    assert _holdout_count(tmp_path) == 1


# --- construction._POLICIES state-leak reset (Codex GATE-1 concrete leak) -------------------------

def test_construction_policy_mutation_does_not_leak_between_tasks(tmp_path):
    from algua.portfolio import construction

    pristine = dict(construction._POLICIES)

    # A task body that mutates the shared construction registry, then one that reads it back.
    calls: list[bool] = []

    def _mutating_task(name, *, reload=False, **_):
        construction._POLICIES["__evil__"] = object()  # type: ignore[assignment]
        return {"mutated": True}

    def _observing_task(name, *, reload=False, **_):
        calls.append("__evil__" in construction._POLICIES)
        return {"saw_evil": calls[-1]}

    import algua.cli.research_batch_cmd as batch

    original = dict(batch._DISPATCH)
    batch._DISPATCH["backtest"] = _mutating_task
    batch._DISPATCH["sweep"] = _observing_task
    batch._ALLOWED_KEYS["sweep"] = batch._ALLOWED_KEYS["sweep"] | {"__ignore__"}
    try:
        r = _run_all(tmp_path, [
            {"action": "backtest", "name": "s1", "demo": True},
            {"action": "sweep", "name": "s2", "demo": True},
        ])
    finally:
        batch._DISPATCH.clear()
        batch._DISPATCH.update(original)
        construction._POLICIES.clear()
        construction._POLICIES.update(pristine)

    assert r.exit_code == 0, r.stdout
    # the observing task ran AFTER the mutating one but saw a PRISTINE registry (reset per task)
    assert calls == [False]


# --- malformed tasks-file rejection (fails the whole batch, cleanly) ------------------------------

def test_not_an_array_is_json_error(tmp_path):
    r = _run_all(tmp_path, {"action": "backtest", "name": "x"})
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False


def test_unknown_action_rejected(tmp_path):
    r = _run_all(tmp_path, [{"action": "frobnicate", "name": "x"}])
    assert r.exit_code == 1
    body = json.loads(r.stdout)
    assert body["ok"] is False and "unknown action" in body["error"]


def test_unsupported_key_rejected(tmp_path):
    # `track` is single-command-only and must be rejected up front (no MLflow in a warm batch)
    r = _run_all(tmp_path, [{"action": "backtest", "name": "x", "track": True}])
    assert r.exit_code == 1
    assert "track" in json.loads(r.stdout)["error"]


def test_missing_name_rejected(tmp_path):
    r = _run_all(tmp_path, [{"action": "backtest"}])
    assert r.exit_code == 1
    assert "name" in json.loads(r.stdout)["error"]


def test_missing_tasks_file_is_json_error(tmp_path):
    r = runner.invoke(app, ["research", "run-all", "--tasks", str(tmp_path / "nope.json")])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False
