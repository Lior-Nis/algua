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
        allocations.allocate_in_lane(
            conn, sid, capital=float(capital), actor="human", account_equity=float(equity),
            allowed_stages=frozenset({"paper", "forward_tested"}))
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


def op_reserve_holdout(db_path, barrier, wid, name, period_start, period_end, holdout_frac,
                       holdout_start, holdout_end):
    """Race the atomic holdout reservation (BEGIN IMMEDIATE re-check + insert)."""
    conn = connect(Path(db_path))
    repo = SqliteStrategyRepository(conn)
    sid = _strategy_id(conn, name)
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "go")
    touch(barrier, f"attempting-{wid}")
    try:
        rid, reused = repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start=period_start, period_end=period_end,
            holdout_frac=float(holdout_frac),
            holdout_start=holdout_start, holdout_end=holdout_end, allow_reuse=False)
        _emit({"ok": True, "wid": wid, "rid": rid, "reused": reused})
    except ValueError as exc:
        _emit({"ok": False, "wid": wid, "error": "ValueError", "msg": str(exc)})
    except Exception as exc:
        _emit({"ok": False, "wid": wid, "error": type(exc).__name__, "msg": str(exc)})


_OPS = {
    "lock-hold": op_lock_hold,
    "write-hold": op_write_hold,
    "read-poll": op_read_poll,
    "transition": op_transition,
    "allocate": op_allocate,
    "gate-consume": op_gate_consume,
    "reserve-holdout": op_reserve_holdout,
}


def main(argv: list[str]) -> None:
    op, db_path, barrier_dir, wid, *rest = argv
    _OPS[op](db_path, Path(barrier_dir), wid, *rest)


if __name__ == "__main__":
    main(sys.argv[1:])
