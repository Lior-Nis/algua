"""Real-subprocess concurrency tests for the shared registry DB (#164).

Each test seeds a DB on the main thread, then spawns genuine OS-process workers
(tests/_concurrency_worker.py) against the same file. A lock-holder worker forces
deterministic contention so serialization is observable, not timing-dependent.
"""
from __future__ import annotations

import contextlib
import json
import os
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

# The console script the `[project.scripts]` wiring installs sits next to the venv's python.
ALGUA_BIN = Path(sys.executable).parent / "algua"
STRATEGY_E2E = "cross_sectional_momentum"


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


@contextlib.contextmanager
def _worker_group(barrier: Path):
    """Spawn-and-clean-up scope: any worker still alive when the block exits (including on a
    failed assertion) is released and terminated, so a failing test never leaks subprocesses."""
    procs: list[subprocess.Popen] = []

    def spawn(op, db_path, wid, *args):
        p = _spawn(op, db_path, barrier, wid, *args)
        procs.append(p)
        return p

    try:
        yield spawn
    finally:
        with contextlib.suppress(OSError):
            (barrier / "release").write_text("1")  # unblock any holder still waiting
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.communicate()


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

    with _worker_group(barrier) as spawn:
        holder = spawn("write-hold", db, "h", "s", "backtested")
        reader = spawn("read-poll", db, "r", "s", "200")

        # reader runs all its reads while the holder's write is uncommitted
        result = _collect(reader)
        (barrier / "release").write_text("1")  # let the holder commit + exit
        holder_result = _collect(holder)

        assert holder_result["ok"] is True
        assert result["ok"] is True, result
        # WAL: reader never saw the dirty uncommitted write, only the old committed snapshot,
        # and every multi-statement read was cross-table consistent (asserted inside the worker).
        assert result["seen"] == ["idea"], result


def _release_after_contention(barrier, contenders, wids):
    """Wait until every contender is queued on the held lock, then free the holder.
    Asserting each contender is still running just before release proves it genuinely
    blocked on the lock — a contender that sailed through without contending would have
    already exited, which is caught here rather than passing vacuously. A worker that dies
    before writing its sentinel makes `_wait_file` time out, failing the test (no hang)."""
    _wait_file(barrier / "lock-held")
    for wid in wids:
        _wait_file(barrier / f"ready-{wid}")
    (barrier / "go").write_text("1")
    for wid in wids:
        _wait_file(barrier / f"attempting-{wid}")
    time.sleep(0.1)  # grace: let each contender reach its lock-acquiring statement
    for proc, wid in zip(contenders, wids, strict=True):
        assert proc.poll() is None, f"contender {wid} exited before release — it did not contend"
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

    with _worker_group(barrier) as spawn:
        holder = spawn("lock-hold", db, "h")
        w1 = spawn("transition", db, "w1", "a")
        w2 = spawn("transition", db, "w2", "b")

        _release_after_contention(barrier, [w1, w2], ["w1", "w2"])

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

    with _worker_group(barrier) as spawn:
        holder = spawn("lock-hold", db, "h")
        w1 = spawn("allocate", db, "w1", "a", "30000", "50000")
        w2 = spawn("allocate", db, "w2", "b", "30000", "50000")

        _release_after_contention(barrier, [w1, w2], ["w1", "w2"])

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

    with _worker_group(barrier) as spawn:
        holder = spawn("lock-hold", db, "h")
        w1 = spawn("gate-consume", db, "w1", "s", str(gate_id))
        w2 = spawn("gate-consume", db, "w2", "s", str(gate_id))

        _release_after_contention(barrier, [w1, w2], ["w1", "w2"])

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
    surrounding `with self._conn:` block rolls back exactly as a real mid-tx failure would.
    Relies on the underlying sqlite3.Connection's context-manager behavior for transaction
    handling (commit on success, rollback on exception)."""

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


def test_concurrent_reserve_holdout_single_burn(tmp_path):
    """Two processes reserve the SAME holdout window under forced contention: exactly one wins,
    the other fails closed with the consumed ValueError (not a lock error), and the DB holds
    exactly ONE reservation row for the window. Proves the BEGIN IMMEDIATE re-check+insert is
    atomic — the regression guard for the #161 TOCTOU."""
    db = tmp_path / "r.db"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _seed_strategy(db, "s")

    with _worker_group(barrier) as spawn:
        holder = spawn("lock-hold", db, "h")
        w1 = spawn("reserve-holdout", db, "w1", "s", "2022-01-01", "2022-12-31", "0.2")
        w2 = spawn("reserve-holdout", db, "w2", "s", "2022-01-01", "2022-12-31", "0.2")

        _release_after_contention(barrier, [w1, w2], ["w1", "w2"])

        r1, r2, rh = _collect(w1), _collect(w2), _collect(holder)
        assert rh["ok"] is True

        outcomes = [r1, r2]
        # exactly one wins; the loser failed closed with the consumed error, NOT a lock error
        assert sum(1 for r in outcomes if r["ok"]) == 1, outcomes
        loser = next(r for r in outcomes if not r["ok"])
        assert loser["error"] == "ValueError", loser
        assert "holdout already consumed" in loser["msg"], loser

    # invariant: exactly one row for the window (the winner's reservation)
    check = connect(db)
    n = check.execute(
        "SELECT COUNT(*) c FROM holdout_evaluations WHERE strategy_id="
        "(SELECT id FROM strategies WHERE name='s')").fetchone()["c"]
    assert n == 1, f"expected exactly one reservation row, found {n}"


def test_sequential_reserve_blocks_second_claim(tmp_path):
    """A committed reservation row blocks a second overlapping reserve from a DISTINCT connection
    — fast deterministic guard independent of subprocess timing."""
    from algua.registry.store import SqliteStrategyRepository
    db = tmp_path / "r.db"
    _seed_strategy(db, "s")
    c1 = connect(db)
    sid = c1.execute("SELECT id FROM strategies WHERE name='s'").fetchone()["id"]
    SqliteStrategyRepository(c1).reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    c2 = connect(db)
    with pytest.raises(ValueError, match="holdout already consumed"):
        SqliteStrategyRepository(c2).reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2022-06-01", period_end="2023-06-01", holdout_frac=0.2, allow_reuse=False)


@pytest.mark.skipif(not ALGUA_BIN.exists(), reason="algua console script not installed")
def test_concurrent_research_promote_single_burn_e2e(tmp_path):
    """The issue's literal ask: two real `algua research promote` processes race the SAME
    strategy+window. Exactly one exits 0 (PASS, transitions to candidate), the other fails closed;
    the DB holds exactly ONE committed holdout burn for the window. Robust to interleaving — the
    loser may fail at reserve (holdout already consumed) or at the stage/preflight check if the
    winner fully transitioned first; both are valid fail-closed outcomes."""
    db = tmp_path / "e2e.db"
    env = {**os.environ, "ALGUA_DB_PATH": str(db), "ALGUA_DATA_DIR": str(tmp_path)}

    # idea -> backtested: a single backtest run that also registers the strategy.
    seed = subprocess.run(
        [str(ALGUA_BIN), "backtest", "run", STRATEGY_E2E, "--demo",
         "--start", "2022-01-01", "--end", "2023-12-31", "--register"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    assert seed.returncode == 0, f"backtest seed failed:\n{seed.stderr}\n{seed.stdout}"

    promote_args = [str(ALGUA_BIN), "research", "promote", STRATEGY_E2E, "--demo",
                    "--start", "2022-01-01", "--end", "2023-12-31",
                    "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                    "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                    "--n-combos", "9", "--allow-non-pit", "--actor", "human"]

    def _launch():
        return subprocess.Popen(promote_args, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env)

    # launch BOTH back-to-back against the same DB so they genuinely race the reservation
    p1 = _launch()
    p2 = _launch()
    procs_raw = [p1, p2]
    try:
        out1, err1 = p1.communicate(timeout=180)
        out2, err2 = p2.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        for p in procs_raw:
            if p.poll() is None:
                p.kill()
                p.communicate()
        pytest.fail("e2e promote processes timed out after 180s")

    procs = [(p1, out1, err1), (p2, out2, err2)]
    winners = [(p, out, err) for p, out, err in procs if p.returncode == 0]
    losers = [(p, out, err) for p, out, err in procs if p.returncode != 0]
    assert len(winners) == 1, (
        f"expected exactly one winner, got {len(winners)}; "
        f"rc={[p.returncode for p, _, _ in procs]}\n"
        f"out1={out1}\nerr1={err1}\nout2={out2}\nerr2={err2}")

    # the winner promoted; final stage is candidate. Each command emits one (pretty-printed)
    # JSON document on stdout, so parse the whole stream, not a single line.
    _, win_out, _ = winners[0]
    win_payload = json.loads(win_out)
    assert win_payload["promoted"] is True, win_payload

    check = connect(db)
    n_burn = check.execute(
        "SELECT COUNT(*) c FROM holdout_evaluations WHERE committed_at IS NOT NULL"
    ).fetchone()["c"]
    assert n_burn == 1, f"expected exactly one committed holdout burn, found {n_burn}"
    n_total = check.execute(
        "SELECT COUNT(*) c FROM holdout_evaluations WHERE strategy_id="
        "(SELECT id FROM strategies WHERE name=?)", (STRATEGY_E2E,)).fetchone()["c"]
    assert n_total == 1, f"expected exactly one holdout row total, found {n_total}"
    stage = check.execute(
        "SELECT stage FROM strategies WHERE name=?", (STRATEGY_E2E,)
    ).fetchone()["stage"]
    assert stage == "candidate", stage

    # best-effort: loser emitted parseable fail-closed JSON (don't gate flakiness on the message)
    _, lose_out, _ = losers[0]
    if lose_out.strip():
        with contextlib.suppress(json.JSONDecodeError):
            lose_payload = json.loads(lose_out)
            assert lose_payload.get("ok") is False, lose_payload
