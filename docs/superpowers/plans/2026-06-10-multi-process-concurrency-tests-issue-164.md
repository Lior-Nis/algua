# Multi-Process Concurrency Tests (Issue 164) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the shared-sqlite-registry concurrency primitives (WAL + 5s `busy_timeout` + `BEGIN IMMEDIATE` + the `with self._conn:` atomic transition) hold under *real* separate OS processes, so the next race trips CI instead of review.

**Architecture:** A reusable subprocess worker module (`tests/_concurrency_worker.py`) whose ops each open their own `connect()` and coordinate through sentinel files; an orchestrator (`run_workers`-style helpers in `tests/test_concurrency.py`) that seeds the DB, spawns workers, forces deterministic contention via a lock-holder worker, and asserts on each worker's structured-JSON outcome. One production line makes `busy_timeout` explicit.

**Tech Stack:** Python `subprocess`/`sqlite3`, pytest, `uv run`, the existing `algua.registry` store/allocations/transitions API.

**Spec:** `docs/superpowers/specs/2026-06-10-multi-process-concurrency-tests-issue-164-design.md`

---

## File structure

- **Modify** `algua/registry/db.py` — add one line to `connect()`: explicit `PRAGMA busy_timeout=5000;`.
- **Create** `tests/_concurrency_worker.py` — runnable subprocess worker (leading underscore → not collected by pytest). Dispatches on an `op` arg to one small function each: `read-poll`, `write-hold`, `lock-hold`, `transition`, `allocate`, `gate-consume`. Also exports the sentinel helpers (`touch`, `wait_for`) reused by the orchestrator.
- **Create** `tests/test_concurrency.py` — orchestrator helpers (`_spawn`, `_wait_file`, `_collect`) + the 6 tests.

**Key cross-task contracts (use these exact names/signatures everywhere):**
- Worker CLI: `python -u -m tests._concurrency_worker <op> <db_path> <barrier_dir> <wid> [op-args...]`
- Sentinel files (in `<barrier_dir>`): `ready-<wid>`, `attempting-<wid>`, `lock-held`, `go`, `release`.
- Every worker prints exactly one final JSON line: `{"ok": true, ...}` or `{"ok": false, "error": "<ExcName>", "msg": "..."}`.
- Orchestrator `_collect(proc)` returns the parsed final JSON dict; asserts return code 0 and surfaces stderr on failure/timeout.

---

## Task 1: Make `busy_timeout` explicit + pin it

**Files:**
- Modify: `algua/registry/db.py` (the `connect()` function, around line 345)
- Test: `tests/test_concurrency.py` (new file — first test)

The cross-process tests depend on a nonzero `busy_timeout`. It is currently 5000ms only via Python's implicit `sqlite3.connect(timeout=5.0)` default. Make it explicit (defensive: survives a future `timeout=0`) and pin it. The value is already 5000 implicitly, so this test passes before and after — it is a **contract pin**, not a red→green.

- [ ] **Step 1: Write the pin test**

Create `tests/test_concurrency.py` with:

```python
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
```

- [ ] **Step 2: Run it to see current behavior**

Run: `uv run pytest tests/test_concurrency.py::test_connect_sets_busy_timeout -v`
Expected: PASS (busy_timeout is already 5000 via the driver default). This confirms the pin matches reality.

- [ ] **Step 3: Make the production value explicit**

In `algua/registry/db.py`, in `connect()`, add the `busy_timeout` pragma next to the existing WAL pragma:

```python
def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")  # WAL + busy_timeout = deliberate concurrency posture
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn
```

- [ ] **Step 4: Run the test again**

Run: `uv run pytest tests/test_concurrency.py::test_connect_sets_busy_timeout -v`
Expected: PASS. The value is now set explicitly in-code, no longer reliant on the driver default.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_concurrency.py
git commit -m "test(164): pin explicit busy_timeout=5000 in connect()"
```

---

## Task 2: Worker harness + Test 1 (WAL reader vs uncommitted writer)

**Files:**
- Create: `tests/_concurrency_worker.py`
- Modify: `tests/test_concurrency.py` (add orchestrator helpers + the reader/writer test)

This task stands up the whole subprocess harness and exercises it with the first real test: a holder worker holds an **uncommitted multi-row** write (stage UPDATE + matching transition INSERT) while a reader worker polls multi-statement snapshot reads. The reader must never block ("database is locked") and never see a torn cross-table state or a dirty (uncommitted) write.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_concurrency.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_concurrency.py::test_concurrent_writer_and_reader_no_lock_errors -v`
Expected: FAIL — `_collect` raises an AssertionError whose surfaced worker stderr contains `No module named 'tests._concurrency_worker'` (the worker module does not exist yet).

- [ ] **Step 3: Create the worker module**

Create `tests/_concurrency_worker.py`:

```python
"""Subprocess worker for the multi-process concurrency tests (#164).

Run as: python -u -m tests._concurrency_worker <op> <db_path> <barrier_dir> <wid> [args...]

Each op opens its OWN sqlite connection (genuine cross-process), coordinates via
sentinel files under <barrier_dir>, runs a critical section, and prints exactly one
JSON line of outcome to stdout. Leading underscore keeps pytest from collecting it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from algua.contracts.lifecycle import Actor, Stage
from algua.registry import allocations
from algua.registry.db import connect
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

POLL = 0.005
SENTINEL_TIMEOUT = 30.0


def touch(barrier: Path, name: str) -> None:
    (barrier / name).write_text("1")


def wait_for(barrier: Path, name: str, timeout: float = SENTINEL_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    target = barrier / name
    while not target.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"sentinel {name} not seen within {timeout}s")
        time.sleep(POLL)


def _emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def _strategy_id(conn, name: str) -> int:
    return int(conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()["id"])


def op_lock_hold(db_path, barrier, wid):
    """Hold the single writer lock open until released — forces contenders to queue."""
    conn = connect(Path(db_path))
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    touch(barrier, "lock-held")
    wait_for(barrier, "release")
    conn.execute("COMMIT")
    _emit({"ok": True, "wid": wid})


def op_write_hold(db_path, barrier, wid, name, to_stage):
    """Hold an UNCOMMITTED, internally-consistent multi-row write (stage + transition)."""
    conn = connect(Path(db_path))
    conn.isolation_level = None
    sid = _strategy_id(conn, name)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE strategies SET stage=? WHERE id=?", (to_stage, sid))
    conn.execute(
        "INSERT INTO stage_transitions"
        "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (sid, "idea", to_stage, "agent", "concurrency-test", "2026-01-01T00:00:00+00:00"),
    )
    touch(barrier, "lock-held")
    wait_for(barrier, "release")
    conn.execute("COMMIT")
    _emit({"ok": True, "wid": wid})


def op_read_poll(db_path, barrier, wid, name, n_reads):
    """Poll multi-statement snapshot reads while a writer holds an uncommitted write."""
    conn = connect(Path(db_path))
    conn.isolation_level = None
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "lock-held")
    seen: set[str] = set()
    try:
        for _ in range(int(n_reads)):
            conn.execute("BEGIN")  # one snapshot across both reads
            row = conn.execute("SELECT id, stage FROM strategies WHERE name=?", (name,)).fetchone()
            last = conn.execute(
                "SELECT to_stage FROM stage_transitions WHERE strategy_id=?"
                " ORDER BY id DESC LIMIT 1", (row["id"],)).fetchone()
            conn.execute("COMMIT")
            # cross-table consistency: the row's stage matches its latest transition
            assert row["stage"] == last["to_stage"], (
                f"torn read: stage={row['stage']} latest_to={last['to_stage']}")
            seen.add(row["stage"])
        _emit({"ok": True, "seen": sorted(seen)})
    except Exception as exc:  # a lock error or torn read fails the test, with detail
        _emit({"ok": False, "error": type(exc).__name__, "msg": str(exc)})


def op_transition(db_path, barrier, wid, name):
    """Advance a strategy idea->backtested via the validating seam (no gate, no identity)."""
    conn = connect(Path(db_path))
    repo = SqliteStrategyRepository(conn)
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "go")
    touch(barrier, f"attempting-{wid}")
    try:
        transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT, reason="concurrency-test")
        _emit({"ok": True, "wid": wid})
    except Exception as exc:
        _emit({"ok": False, "wid": wid, "error": type(exc).__name__, "msg": str(exc)})


def op_allocate(db_path, barrier, wid, name, capital, equity):
    """Race the allocation cap (BEGIN IMMEDIATE read-check-write)."""
    conn = connect(Path(db_path))
    sid = _strategy_id(conn, name)
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "go")
    touch(barrier, f"attempting-{wid}")
    try:
        allocations.allocate(conn, sid, capital=float(capital), actor="human",
                             account_equity=float(equity))
        _emit({"ok": True, "wid": wid})
    except allocations.AllocationError as exc:
        _emit({"ok": False, "wid": wid, "error": "AllocationError", "msg": str(exc)})
    except Exception as exc:
        _emit({"ok": False, "wid": wid, "error": type(exc).__name__, "msg": str(exc)})


def op_gate_consume(db_path, barrier, wid, name, gate_id):
    """Race the single-use candidate gate token via the atomic consume in apply_transition."""
    conn = connect(Path(db_path))
    repo = SqliteStrategyRepository(conn)
    rec = repo.get(name)
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "go")
    touch(barrier, f"attempting-{wid}")
    try:
        repo.apply_transition(rec, to=Stage.CANDIDATE, actor=Actor.AGENT,
                              consume_gate_id=int(gate_id))
        _emit({"ok": True, "wid": wid})
    except Exception as exc:
        _emit({"ok": False, "wid": wid, "error": type(exc).__name__, "msg": str(exc)})


_OPS = {
    "lock-hold": op_lock_hold,
    "write-hold": op_write_hold,
    "read-poll": op_read_poll,
    "transition": op_transition,
    "allocate": op_allocate,
    "gate-consume": op_gate_consume,
}


def main(argv: list[str]) -> None:
    op, db_path, barrier_dir, wid, *rest = argv
    _OPS[op](db_path, Path(barrier_dir), wid, *rest)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_concurrency.py::test_concurrent_writer_and_reader_no_lock_errors -v`
Expected: PASS. (If it errors with "database is locked", the WAL/busy_timeout posture regressed — investigate before continuing.)

- [ ] **Step 5: Commit**

```bash
git add tests/_concurrency_worker.py tests/test_concurrency.py
git commit -m "test(164): subprocess harness + WAL reader-vs-uncommitted-writer test"
```

---

## Task 3: Test 2 — two writers serialize behind the held lock

**Files:**
- Modify: `tests/test_concurrency.py`

Two writer workers each transition a **distinct** strategy idea->backtested while a holder pins the writer lock; on release they serialize via `busy_timeout`. Both must succeed with no lock error, and each strategy lands at `backtested` with exactly one new transition row.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_concurrency.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails (then passes)**

Run: `uv run pytest tests/test_concurrency.py::test_concurrent_writers_serialize -v`
Expected: PASS (the worker ops it uses — `lock-hold`, `transition` — already exist from Task 2). If it does not pass, inspect the surfaced worker stderr.

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test(164): two-writer cross-process serialization test"
```

---

## Task 4: Test 3 — allocation cap holds under a real race

**Files:**
- Modify: `tests/test_concurrency.py`

`account_equity=50_000`; two workers each `allocate(30_000)` to different strategies, forced to contend. Assert the **invariant first** (final total <= equity; no double active allocation; no partial rows), then the **outcome** (exactly one succeeds, the other raises `AllocationError` — a domain rejection, not a lock error).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_concurrency.py`:

```python
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
        sid = check.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()["id"]
        n_active = check.execute(
            "SELECT COUNT(*) c FROM strategy_allocations WHERE strategy_id=? AND revoked_ts IS NULL",
            (sid,)).fetchone()["c"]
        assert n_active <= 1, f"{name} has {n_active} active allocations"

    # outcome: exactly one succeeded; the loser got a domain rejection, not a lock error
    outcomes = [r1, r2]
    assert sum(1 for r in outcomes if r["ok"]) == 1, outcomes
    loser = next(r for r in outcomes if not r["ok"])
    assert loser["error"] == "AllocationError", loser
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_concurrency.py::test_concurrent_allocate_respects_cap -v`
Expected: PASS (uses the existing `lock-hold` + `allocate` ops).

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test(164): allocation-cap holds under a real cross-process race"
```

---

## Task 5: Test 4 — single-use candidate gate token across two processes

**Files:**
- Modify: `tests/test_concurrency.py`

The load-bearing token invariant. Seed a strategy at `backtested` with **one** passing AGENT gate token, then race two workers that both call `apply_transition(.., CANDIDATE, AGENT, consume_gate_id=tok)` against that token. The atomic `UPDATE gate_evaluations SET consumed=1 WHERE id=? AND consumed=0` (rowcount==1 guard) must let exactly one advance.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_concurrency.py` (helper to mint a token + the test):

```python
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
    assert check.execute("SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()["consumed"] == 1
    assert check.execute("SELECT stage FROM strategies WHERE id=?", (sid,)).fetchone()["stage"] == "candidate"
    n = check.execute(
        "SELECT COUNT(*) c FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'",
        (sid,)).fetchone()["c"]
    assert n == 1, f"{n} candidate transition rows"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_concurrency.py::test_concurrent_candidate_gate_single_use -v`
Expected: PASS (uses the existing `lock-hold` + `gate-consume` ops).

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test(164): single-use candidate gate token across two processes"
```

---

## Task 6: Test 5 — apply_transition rolls back on a mid-transaction failure

**Files:**
- Modify: `tests/test_concurrency.py`

Single-process transaction-atomicity test (no subprocess). Inject a fault that raises only on the `INSERT INTO stage_transitions` of the tested call, using a thin connection **proxy** (monkeypatching the C-extension `execute` is unreliable; a proxy that delegates `__enter__`/`__exit__` to the real connection gives a faithful rollback). The token-consume + stage UPDATE that ran before the fault must both revert.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_concurrency.py`:

```python
import sqlite3

import pytest


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

    # a repo whose connection raises on the transition INSERT — fires AFTER token-consume + stage UPDATE
    faulty = SqliteStrategyRepository(_FaultyConn(real, "INSERT INTO stage_transitions"))
    rec_bt = repo.get("s")
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        faulty.apply_transition(rec_bt, to=Stage.CANDIDATE, actor=Actor.AGENT,
                                consume_gate_id=gate_id)

    # assert full rollback on a FRESH connection (nothing partially committed)
    fresh = connect(db)
    assert fresh.execute("SELECT stage FROM strategies WHERE id=?", (rec.id,)).fetchone()["stage"] == "backtested"
    assert fresh.execute("SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()["consumed"] == 0
    n = fresh.execute(
        "SELECT COUNT(*) c FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'",
        (rec.id,)).fetchone()["c"]
    assert n == 0
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/test_concurrency.py::test_apply_transition_rolls_back_on_failure -v`
Expected: PASS. If `consumed` reads `1` or stage reads `candidate`, the transaction is **not** atomic — that is a real production bug to report, not a test fix.

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test(164): apply_transition rolls back token-consume + stage on failure"
```

---

## Task 7: Full gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole concurrency file**

Run: `uv run pytest tests/test_concurrency.py -v`
Expected: 6 passed.

- [ ] **Step 2: Run it a few times to shake out flakiness**

Run: `uv run pytest tests/test_concurrency.py -p no:randomly -q --count=5` (if `pytest-repeat` is available) or simply rerun 3-5×.
Expected: green every run. Any "database is locked" or timeout is a real defect to investigate (holder release timing, busy_timeout), not to paper over.

- [ ] **Step 3: Full project gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. Note: `_concurrency_worker.py` lives under `tests/` so import-linter (which governs `algua/`) is unaffected; the only `algua/` change is the one `connect()` line.

- [ ] **Step 4: Final commit (if anything was adjusted)**

```bash
git add -A && git commit -m "test(164): multi-process concurrency suite green end-to-end"
```

---

## Self-review notes (author)

- **Spec coverage:** Test 1 ↔ spec test 1 (WAL reader/uncommitted writer + multi-statement snapshot consistency); Test 2 ↔ spec test 2 (two-writer serialize); Test 3 ↔ spec test 3 (allocate cap, invariant-first); Test 4 ↔ spec test 4 (single-use gate token cross-process); Test 5 ↔ spec test 5 (rollback fault injection); Task 1 ↔ spec test 6 + the one production line. Lock-holder contention, `-u`/flush, stderr capture, ~30s join, tmp_path barriers all present.
- **Deliberate design choices encoded:** Test 4 drives `apply_transition` directly (the atomic single-use consume is the concurrency primitive; identity recompute in `_validate_shortlist_gate` is orthogonal and unit-tested elsewhere). Test 2 drives `transition_strategy` (the validating seam) since it is a normal serialized transition. Holder uses bare `BEGIN IMMEDIATE` for tests 2/3/4 (just needs the lock) and a multi-row uncommitted write for test 1 (reader needs something to not-see).
- **Out of scope (per spec):** data-manifest/parquet concurrency (#158), deadlock tests (SQLite single-writer precludes A↔B), multi-connection-per-worker, WAL checkpoint/side-file lifecycle.
