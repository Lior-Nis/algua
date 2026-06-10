"""Real-subprocess concurrency tests for the shared registry DB (#164).

Each test seeds a DB on the main thread, then spawns genuine OS-process workers
(tests/_concurrency_worker.py) against the same file. A lock-holder worker forces
deterministic contention so serialization is observable, not timing-dependent.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

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
        raise AssertionError(f"worker timed out after {timeout}s; stderr:\n{err}")
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
        row = check.execute("SELECT id, stage FROM strategies WHERE name=?", (name,)).fetchone()
        assert row["stage"] == "backtested"
        n = check.execute(
            "SELECT COUNT(*) c FROM stage_transitions WHERE strategy_id=? AND to_stage='backtested'",
            (row["id"],)).fetchone()["c"]
        assert n == 1, f"{name} has {n} backtested transitions"
