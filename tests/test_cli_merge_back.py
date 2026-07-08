"""Tests for the `paper merge-back` CLI wiring (#485, Task 7).

The command is thin glue: it takes the repo-global flock, builds the concrete
`RealGitOps`/`JsonlJournal`/registry/broker seams + the CODEOWNERS text, and drives the REAL pure
`run_merge_back` state machine. These tests stub the git/gate/broker/promote/intake seams (no real
git, no real quality-gate subprocess, no real broker, no real walk-forward) but keep the REAL
registry read helpers (`passing_gate_by_token`, the target-allocation read) so the token-bound
outcome attribution (finding #5) is exercised end-to-end against a real DB the stubs mutate.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.cli.paper_cmd as paper_cmd
import algua.cli.research_cmd as research_cmd
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.alpaca_broker import AccountState
from algua.operator.diff_policy import DiffEntry
from algua.operator.gitops import RemoteMovedError
from algua.operator.mergeback import merge_back_lock
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

runner = CliRunner()
_STRAT = "cross_sectional_momentum"

_GATE_COLS = (
    "strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos, funnel_window_days,"
    " breadth_provenance, pit_ok, holdout_n_bars, min_holdout_observations, code_hash, config_hash,"
    " data_source, period_start, period_end, holdout_frac, actor, decision_json, created_at,"
    " attempt_token"
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")


class _FakeBroker:
    def account(self) -> AccountState:
        return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
                            account_id="t")


class _FakeGit:
    """A GitOps stub implementing the new protocol: clean main, an authoritative origin, a merge
    that lands and whose blobs are present on origin/main."""

    def __init__(self, entries=None):
        self.calls: list[object] = []
        self.origin_main = "BASE"
        self.entries = entries if entries is not None else [
            DiffEntry("100644", "A", None, "algua/strategies/momentum/x.py")]

    def merge_in_progress(self): return False
    def abort_merge(self): self.calls.append("abort")
    def current_branch(self): return "main"
    def working_tree_clean(self): return True
    def fetch_remote(self, ref): pass
    # Local `main` HEAD tracks origin (no drift); any other ref resolves to the branch tip. This
    # keeps the finding #1 precondition (local main == freshly-fetched origin/main) satisfied.
    def resolve(self, ref): return self.origin_main if ref == "main" else "TIP"
    def remote_tip(self, ref): return self.origin_main
    def merge_base(self, a, b): return "MB"
    def changed_entries(self, base, tip): return self.entries
    def begin_merge(self, tip): self.calls.append(("begin", tip))
    def commit_merge(self): self.calls.append("commit")
    def merge_commit_of(self, tip): return "MERGE"
    def commit_second_parent(self, sha): return "TIP"
    def is_ancestor(self, sha, ref): return True

    def push_cas(self, merge_sha, expected_base):
        self.calls.append(("push", merge_sha))
        self.origin_main = merge_sha

    def tree_blobs(self, sha, paths): return {p: f"blob:{p}" for p in paths}
    def blob_at(self, ref, path): return f"blob:{path}"
    def revert_merge(self, sha):
        self.calls.append(("revert", sha))
        return "REVERT"

    def push_revert(self, revert_sha, expected_merge_sha):
        if expected_merge_sha != self.origin_main:
            raise RemoteMovedError("origin/main moved before revert push")
        self.calls.append(("revert_push", revert_sha))
        self.origin_main = revert_sha


def _register_backtested(name: str) -> None:
    r = runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
    assert r.exit_code == 0, r.output


def _insert_passing_gate(conn, strategy_id, token):
    conn.execute(
        f"INSERT INTO gate_evaluations ({_GATE_COLS}) VALUES"
        " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (strategy_id, 1, 1, 1, 1, 365, "measured", 1, 100, 63, "c", "cfg", "Demo",
         "2022-01-01", "2022-12-31", 0.2, "agent", "{}", "2022-01-01T00:00:00Z", token))


def _wire(monkeypatch, *, gate: bool, git: _FakeGit, promote_calls: list,
          promote_commits: bool = True):
    monkeypatch.setattr(paper_cmd, "RealGitOps", lambda repo_root: git)
    monkeypatch.setattr(paper_cmd, "_run_quality_gate", lambda repo_root: gate)
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", lambda: _FakeBroker())

    def _fake_promote(**kwargs):
        promote_calls.append(kwargs)
        if promote_commits:
            # Simulate a real commit: advance the stage + mint the token-stamped passing gate row so
            # the driver's REAL passing_gate_by_token attributes the outcome to THIS attempt.
            with closing(connect(get_settings().db_path)) as conn:
                migrate(conn)
                rec = SqliteStrategyRepository(conn).get(kwargs["name"])
                conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
                _insert_passing_gate(conn, rec.id, kwargs["attempt_token"])
                conn.commit()
        return {"promoted": promote_commits}

    monkeypatch.setattr(research_cmd, "promote_task", _fake_promote)

    def _fake_intake(conn, *, equity, max_concurrent, actor):
        # Stand in for the FIFO admit: move THIS strategy to paper + seed an allocation so the
        # driver's REAL target-allocation read emits promoted_allocated.
        rec = SqliteStrategyRepository(conn).get(_STRAT)
        conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (rec.id,))
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor)"
            " VALUES (?,?,?,?)",
            (rec.id, 20_000.0, datetime.now(UTC).isoformat(), "agent"))
        conn.commit()
        return {"admitted": [{"strategy": _STRAT, "capital": 20_000.0}], "queued": []}

    monkeypatch.setattr(paper_cmd, "_run_intake", _fake_intake)


def _invoke():
    return runner.invoke(app, [
        "paper", "merge-back", "--branch", "feat/strat", "--strategy", _STRAT,
        "--universe", "sp500", "--start", "2022-01-01", "--end", "2023-12-31"])


def test_green_gate_and_promote_allocates(monkeypatch):
    _register_backtested(_STRAT)
    git = _FakeGit()
    promote_calls: list = []
    _wire(monkeypatch, gate=True, git=git, promote_calls=promote_calls)

    result = _invoke()
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["status"] == "promoted_allocated"
    assert payload["merged"] is True and payload["promoted"] is True
    assert payload["reverted"] is False
    assert payload["attempt_token"] and payload["gate_id"]
    # Merge committed + pushed (remote CAS), never reverted.
    assert ("begin", "TIP") in git.calls and "commit" in git.calls
    assert ("push", "MERGE") in git.calls
    assert not any(isinstance(c, tuple) and c[0] == "revert" for c in git.calls)
    # Promote driven with strict-agent inputs + the per-attempt token (finding #5).
    assert len(promote_calls) == 1
    assert promote_calls[0]["actor"] == "agent"
    assert promote_calls[0]["attempt_token"] == payload["attempt_token"]


def test_red_gate_fails_closed_without_promote(monkeypatch):
    _register_backtested(_STRAT)
    git = _FakeGit()
    promote_calls: list = []
    _wire(monkeypatch, gate=False, git=git, promote_calls=promote_calls)

    result = _invoke()
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "gate_failed"
    assert payload["merged"] is False and payload["promoted"] is False
    assert "abort" in git.calls and "commit" not in git.calls
    assert promote_calls == []


def test_diff_policy_rejects_protected_path(monkeypatch):
    _register_backtested(_STRAT)
    # A branch that touches CODEOWNERS-protected store.py is rejected BEFORE any merge.
    git = _FakeGit(entries=[DiffEntry("100644", "M", None, "algua/registry/store.py")])
    promote_calls: list = []
    _wire(monkeypatch, gate=True, git=git, promote_calls=promote_calls)

    result = _invoke()
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "diff_policy_rejected"
    assert not any(isinstance(c, tuple) and c[0] == "begin" for c in git.calls)
    assert promote_calls == []


def test_promote_not_committed_reverts(monkeypatch):
    _register_backtested(_STRAT)
    git = _FakeGit()
    promote_calls: list = []
    _wire(monkeypatch, gate=True, git=git, promote_calls=promote_calls, promote_commits=False)

    result = _invoke()
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "promote_failed"
    assert payload["reverted"] is True
    assert ("revert", "MERGE") in git.calls


def test_held_lock_fails_second_invocation(monkeypatch):
    _register_backtested(_STRAT)
    git = _FakeGit()
    _wire(monkeypatch, gate=True, git=git, promote_calls=[])
    lock_path = Path(get_settings().db_path).parent / "merge_back.lock"
    with merge_back_lock(lock_path):
        result = _invoke()
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "another merge-back cycle is in progress" in payload["error"]
