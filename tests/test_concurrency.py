"""Real-subprocess concurrency tests for the shared registry DB (#164).

Each test seeds a DB on the main thread, then spawns genuine OS-process workers
(tests/_concurrency_worker.py) against the same file. A lock-holder worker forces
deterministic contention so serialization is observable, not timing-dependent.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from algua.registry.db import connect, migrate

REPO_ROOT = Path(__file__).resolve().parent.parent
JOIN_TIMEOUT = 30.0
POLL = 0.005


def test_connect_sets_busy_timeout(tmp_path):
    conn = connect(tmp_path / "b.db")
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def _spawn(op, db_path, barrier, wid, *args):
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "tests._concurrency_worker",
         op, str(db_path), str(barrier), str(wid), *map(str, args)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _wait_file(path: Path, timeout: float = JOIN_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"sentinel {path.name} not created within {timeout}s")
        time.sleep(POLL)


def _collect(proc, timeout: float = JOIN_TIMEOUT) -> dict:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        raise AssertionError(  # noqa: B904
            f"worker timed out after {timeout}s; stderr:\n{err}"
        )
    assert proc.returncode == 0, f"worker exited {proc.returncode}; stderr:\n{err}"
    lines = [ln for ln in (out or "").splitlines() if ln.strip()]
    assert lines, f"worker produced no JSON; stderr:\n{err}"
    return json.loads(lines[-1])


def _seed_strategy(db_path: Path, name: str = "s") -> None:
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(db_path)
    migrate(conn)
    SqliteStrategyRepository(conn).add(name)
    conn.close()


def test_concurrent_writer_and_reader_no_lock_errors(tmp_path):
    db = tmp_path / "r.db"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _seed_strategy(db, "s")

    holder = _spawn("write-hold", db, barrier, "h", "s", "backtested")
    reader = _spawn("read-poll", db, barrier, "r", "s", "200")

    # reader runs all its reads while the holder's write is uncommitted
    result = _collect(reader)
    (barrier / "release").write_text("1")  # let the holder commit + exit
    holder_result = _collect(holder)

    assert holder_result["ok"] is True
    assert result["ok"] is True, result
    # WAL: reader never saw the dirty uncommitted write, only the old committed snapshot,
    # and every multi-statement read was cross-table consistent (asserted inside the worker).
    assert result["seen"] == ["idea"], result


def _release_after_contention(barrier: Path, wids: list[str]):
    """Wait until every contender is queued on the lock, then free the holder."""
    _wait_file(barrier / "lock-held")
    for wid in wids:
        _wait_file(barrier / f"ready-{wid}")
    (barrier / "go").write_text("1")
    for wid in wids:
        _wait_file(barrier / f"attempting-{wid}")
    time.sleep(0.1)  # grace: let both contenders actually block on the writer lock
    (barrier / "release").write_text("1")


def test_concurrent_writers_serialize(tmp_path):
    db = tmp_path / "r.db"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _seed_strategy(db, "a")
    # add a second strategy on the same DB
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(db)
    SqliteStrategyRepository(conn).add("b")
    conn.close()

    holder = _spawn("lock-hold", db, barrier, "h")
    w1 = _spawn("transition", db, barrier, "w1", "a")
    w2 = _spawn("transition", db, barrier, "w2", "b")

    _release_after_contention(barrier, ["w1", "w2"])

    r1, r2, rh = _collect(w1), _collect(w2), _collect(holder)
    assert rh["ok"] is True
    assert r1["ok"] is True, r1
    assert r2["ok"] is True, r2

    # final state: both advanced, each with exactly one idea->backtested transition row
    check = connect(db)
    for name in ("a", "b"):
        row = check.execute(
            "SELECT id, stage FROM strategies WHERE name=?", (name,)
        ).fetchone()
        assert row["stage"] == "backtested"
        n = check.execute(
            "SELECT COUNT(*) c FROM stage_transitions"
            " WHERE strategy_id=? AND to_stage='backtested'",
            (row["id"],)).fetchone()["c"]
        assert n == 1, f"{name} has {n} backtested transitions"


def test_concurrent_allocate_respects_cap(tmp_path):
    db = tmp_path / "r.db"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _seed_strategy(db, "a")
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(db)
    SqliteStrategyRepository(conn).add("b")
    conn.close()

    holder = _spawn("lock-hold", db, barrier, "h")
    w1 = _spawn("allocate", db, barrier, "w1", "a", "30000", "50000")
    w2 = _spawn("allocate", db, barrier, "w2", "b", "30000", "50000")

    _release_after_contention(barrier, ["w1", "w2"])

    r1, r2, rh = _collect(w1), _collect(w2), _collect(holder)
    assert rh["ok"] is True

    # invariant first: the cap is never breached, regardless of interleaving
    from algua.registry import allocations
    check = connect(db)
    assert allocations.total_allocated(check) <= 50_000
    for name in ("a", "b"):
        sid = check.execute(
            "SELECT id FROM strategies WHERE name=?", (name,)
        ).fetchone()["id"]
        n_active = check.execute(
            "SELECT COUNT(*) c FROM strategy_allocations"
            " WHERE strategy_id=? AND revoked_ts IS NULL",
            (sid,)).fetchone()["c"]
        assert n_active <= 1, f"{name} has {n_active} active allocations"

    # outcome: exactly one succeeded; the loser got a domain rejection, not a lock error
    outcomes = [r1, r2]
    assert sum(1 for r in outcomes if r["ok"]) == 1, outcomes
    loser = next(r for r in outcomes if not r["ok"])
    assert loser["error"] == "AllocationError", loser


def _seed_backtested_with_gate_token(db_path: Path, name: str = "s") -> tuple[int, int]:
    """Create a strategy at `backtested` and mint one passing AGENT gate token. Returns
    (strategy_id, gate_id)."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.transitions import transition_strategy

    conn = connect(db_path)
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(name)
    transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT, reason="setup")
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=30, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=100, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="test", snapshot_id=None, period_start="2026-01-01",
        period_end="2026-02-01", holdout_frac=0.3, actor="agent", decision_json="{}")
    conn.close()
    return rec.id, gate_id


def test_concurrent_candidate_gate_single_use(tmp_path):
    db = tmp_path / "r.db"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    sid, gate_id = _seed_backtested_with_gate_token(db, "s")

    holder = _spawn("lock-hold", db, barrier, "h")
    w1 = _spawn("gate-consume", db, barrier, "w1", "s", str(gate_id))
    w2 = _spawn("gate-consume", db, barrier, "w2", "s", str(gate_id))

    _release_after_contention(barrier, ["w1", "w2"])

    r1, r2, rh = _collect(w1), _collect(w2), _collect(holder)
    assert rh["ok"] is True

    outcomes = [r1, r2]
    assert sum(1 for r in outcomes if r["ok"]) == 1, outcomes
    loser = next(r for r in outcomes if not r["ok"])
    assert loser["error"] == "TransitionError", loser

    # final: token consumed exactly once; strategy at candidate; exactly one new transition row
    check = connect(db)
    consumed = check.execute(
        "SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,)
    ).fetchone()["consumed"]
    assert consumed == 1
    stage = check.execute(
        "SELECT stage FROM strategies WHERE id=?", (sid,)
    ).fetchone()["stage"]
    assert stage == "candidate"
    n = check.execute(
        "SELECT COUNT(*) c FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'",
        (sid,)).fetchone()["c"]
    assert n == 1, f"{n} candidate transition rows"


class _FaultyConn:
    """Delegates to a real sqlite3 connection but raises on a chosen statement, so the
    surrounding `with self._conn:` block rolls back exactly as a real mid-tx failure would."""

    def __init__(self, real: sqlite3.Connection, fail_on: str):
        self._real = real
        self._fail_on = fail_on

    def execute(self, sql, *args):
        if self._fail_on in sql:
            raise sqlite3.OperationalError("injected mid-transaction failure")
        return self._real.execute(sql, *args)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_apply_transition_rolls_back_on_failure(tmp_path):
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.transitions import transition_strategy

    db = tmp_path / "r.db"
    real = connect(db)
    migrate(real)
    repo = SqliteStrategyRepository(real)
    rec = repo.add("s")
    transition_strategy(repo, "s", Stage.BACKTESTED, Actor.AGENT, reason="setup")
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=30, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=100, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="test", snapshot_id=None, period_start="2026-01-01",
        period_end="2026-02-01", holdout_frac=0.3, actor="agent", decision_json="{}")

    # a repo whose connection raises on the transition INSERT —
    # fires AFTER token-consume + stage UPDATE, so all three must roll back together.
    faulty = SqliteStrategyRepository(_FaultyConn(real, "INSERT INTO stage_transitions"))
    rec_bt = repo.get("s")
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        faulty.apply_transition(rec_bt, to=Stage.CANDIDATE, actor=Actor.AGENT,
                                consume_gate_id=gate_id)

    # assert full rollback on a FRESH connection (nothing partially committed)
    fresh = connect(db)
    assert fresh.execute(
        "SELECT stage FROM strategies WHERE id=?", (rec.id,)
    ).fetchone()["stage"] == "backtested"
    assert fresh.execute(
        "SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,)
    ).fetchone()["consumed"] == 0
    n = fresh.execute(
        "SELECT COUNT(*) c FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'",
        (rec.id,)).fetchone()["c"]
    assert n == 0
