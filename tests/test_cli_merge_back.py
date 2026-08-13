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
_GATED_TREE = "f" * 40  # the tree sha the stubbed green gate returns (binds commit_merge)

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
    def commit_merge(self, *, expected_tree): self.calls.append(("commit", expected_tree))
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
    # The gate seam's real contract: gated tree sha on green, None on red.
    monkeypatch.setattr(paper_cmd, "_run_quality_gate",
                        lambda repo_root: _GATED_TREE if gate else None)
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
    # --demo satisfies the fail-fast transport preflight (GATE-2 #5: exactly one of
    # --demo/--snapshot); these tests stub promote/intake, so no real provider is touched.
    return runner.invoke(app, [
        "paper", "merge-back", "--branch", "feat/strat", "--strategy", _STRAT,
        "--universe", "sp500", "--start", "2022-01-01", "--end", "2023-12-31", "--demo"])


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
    # Merge committed (bound to the gate-blessed tree sha) + pushed (remote CAS), never reverted.
    assert ("begin", "TIP") in git.calls and ("commit", _GATED_TREE) in git.calls
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
    assert "abort" in git.calls
    assert not any(isinstance(c, tuple) and c[0] == "commit" for c in git.calls)
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
    # HIGH-4: the lock is anchored at the repo's own git dir (per-checkout), NOT db_path.parent, so
    # two invocations on the SAME checkout serialize regardless of ALGUA_DB_PATH. Acquire the same
    # git-dir-rooted lock the command computes to prove the second invocation fails closed.
    import subprocess as _sp
    git_dir = _sp.run(["git", "rev-parse", "--absolute-git-dir"],
                      cwd=Path(paper_cmd.__file__).resolve().parent,
                      capture_output=True, text=True, check=True).stdout.strip()
    lock_path = Path(git_dir) / "merge_back.lock"
    with merge_back_lock(lock_path):
        result = _invoke()
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "another merge-back cycle is in progress" in payload["error"]


# --- authoritative intake (mergeback-authoritative-intake design): CLI E2E ------------------------


def _invoke_with_context(*extra: str):
    return runner.invoke(app, [
        "paper", "merge-back", "--branch", "feat/strat", "--strategy", _STRAT,
        "--universe", "sp500", "--start", "2022-01-01", "--end", "2023-12-31",
        "--demo", *extra])


def test_unregistered_strategy_full_cycle_registers_produces_evidence_and_promotes(monkeypatch):
    # NO _register_backtested: the strategy has no authoritative row at all (the factory-fresh
    # state). The REAL ensure_backtested + produce_evidence run (real registry DB, real demo sweep
    # + backtest via the injected task bodies); promote/intake stay stubbed.
    git = _FakeGit()
    promote_calls: list = []
    _wire(monkeypatch, gate=True, git=git, promote_calls=promote_calls)
    # The REAL evidence sweep/backtest resolve the PIT universe — seed its membership timeline.
    r = runner.invoke(app, ["data", "ingest-universe", "sp500",
                            "--symbols", "AAPL,MSFT,NVDA,AMZN,GOOGL",
                            "--effective-date", "2021-12-31"])
    assert r.exit_code == 0, r.output

    result = _invoke_with_context("--sweep-param", "lookback=20,60")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "promoted_allocated"

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.row_factory = __import__("sqlite3").Row
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(_STRAT)  # the row EXISTS now (created by ensure_backtested)
        # Provenance-tagged registration + the idea->backtested intake transition.
        assert "mergeback:intake" in rec.tags
        transitions = repo.list_transitions(_STRAT)
        assert [(t["from_stage"], t["to_stage"]) for t in transitions[:2]] == [
            (None, "idea"), ("idea", "backtested")]
        # Authoritative evidence landed: measured breadth (one trial, the exact grid) + returns.
        trials = conn.execute(
            "SELECT n_combos, grid_json FROM search_trials WHERE strategy_name=?",
            (_STRAT,)).fetchall()
        assert len(trials) == 1
        assert trials[0]["n_combos"] == 2
        assert json.loads(trials[0]["grid_json"]) == {"lookback": [20, 60]}
        assert repo.load_backtest_returns(_STRAT) is not None
        # The evidence marker completed for this (strategy, branch_tip).
        marker = conn.execute(
            "SELECT status FROM mergeback_evidence me JOIN strategies s ON s.id=me.strategy_id"
            " WHERE s.name=? AND me.branch_tip=?", (_STRAT, "TIP")).fetchone()
        assert marker is not None and marker["status"] == "completed"

    # The promote seam received the transported eval context (strict-agent inputs otherwise).
    assert len(promote_calls) == 1
    assert promote_calls[0]["demo"] is True
    assert promote_calls[0]["snapshot"] is None
    assert promote_calls[0]["actor"] == "agent"


def test_preregistered_strategy_with_breadth_skips_evidence_production(monkeypatch):
    # The direct-authoritative-funnel no-op: a pre-existing backtested strategy that already
    # carries authoritative measured breadth is NOT re-swept (no new trial, no marker).
    _register_backtested(_STRAT)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        SqliteStrategyRepository(conn).record_search_trial(
            _STRAT, 5, json.dumps({"lookback": [10, 20, 30, 40, 60]}))
    git = _FakeGit()
    promote_calls: list = []
    _wire(monkeypatch, gate=True, git=git, promote_calls=promote_calls)

    result = _invoke_with_context("--sweep-param", "lookback=20,60")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "promoted_allocated"

    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n_trials = conn.execute(
            "SELECT COUNT(*) FROM search_trials WHERE strategy_name=?", (_STRAT,)).fetchone()[0]
        assert n_trials == 1  # only the seeded row — the transported grid was NOT re-swept
        n_markers = conn.execute("SELECT COUNT(*) FROM mergeback_evidence").fetchone()[0]
        assert n_markers == 0
    assert len(promote_calls) == 1


# --- (GATE-2 #5) fail-fast transport preflight: a typo'd invocation dies BEFORE any git/journal --


@pytest.mark.parametrize("extra,msg", [
    ([], "exactly one of --demo or --snapshot"),                       # neither
    (["--demo", "--snapshot", "bars-1"], "exactly one of --demo or --snapshot"),  # both
    (["--demo", "--rank-by", "sharpe"], "--rank-by must be one of"),
    (["--demo", "--sweep-param", "malformed"], "malformed --param"),
])
def test_bad_transport_combos_fail_before_any_git_or_journal_mutation(monkeypatch, tmp_path,
                                                                      extra, msg):
    _register_backtested(_STRAT)
    git_constructed: list = []
    monkeypatch.setattr(paper_cmd, "RealGitOps",
                        lambda repo_root: git_constructed.append(repo_root) or _FakeGit())

    result = runner.invoke(app, [
        "paper", "merge-back", "--branch", "feat/strat", "--strategy", _STRAT,
        "--universe", "sp500", "--start", "2022-01-01", "--end", "2023-12-31", *extra])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert msg in payload["error"]
    # The saga never began: no git seam constructed, no journal file written.
    assert git_constructed == []
    assert list(get_settings().db_path.parent.glob("merge_back.*.journal")) == []
