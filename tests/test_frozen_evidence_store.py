"""Story 1.3d §3: the frozen invocation recorder and the tick writer's invocation link.

``record_frozen_invocation`` persists one :class:`FrozenAttempt` as one append-only row, committed
in its own transaction; it refuses (explicitly, not by ``assert``) a connection that already has a
transaction open, and SQLite errors propagate. ``record_tick_snapshot`` stores an optional
``frozen_invocation_id``; ticks without one behave exactly as before.
"""
from __future__ import annotations

import sqlite3
from dataclasses import fields
from pathlib import Path

import pytest

from algua.execution.tick_snapshots import record_tick_snapshot
from algua.registry.db import connect, migrate
from algua.registry.deployment import DeploymentError
from algua.registry.store.frozen_evidence import frozen_invocation, record_frozen_invocation
from tests._frozen_evidence_helpers import (
    IDENTITY,
    attempt,
    failed_attempt,
    record_final_invocation,
    seed_deployment,
)


@pytest.fixture
def conn(tmp_path: Path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    yield c
    c.close()


def _deployment(conn: sqlite3.Connection, name: str = "f", *,
                source_kind: str = "frozen") -> tuple[int, int]:
    return seed_deployment(conn, name, source_kind=source_kind)


def test_records_one_row_and_commits_it(conn, tmp_path):
    _sid, deployment = _deployment(conn)
    recorded = attempt(deployment_id=deployment)

    invocation_id = record_frozen_invocation(conn, recorded)

    assert not conn.in_transaction
    other = connect(tmp_path / "r.db")
    try:
        row = other.execute("SELECT * FROM frozen_invocations").fetchall()
        assert len(row) == 1 and row[0]["id"] == invocation_id
    finally:
        other.close()
    assert frozen_invocation(conn, invocation_id) == recorded


def test_every_attempt_column_is_persisted(conn):
    _sid, deployment = _deployment(conn)
    recorded = failed_attempt(
        deployment_id=deployment, stdout_exceeded=True, stderr_truncated=True, returncode=-9)

    invocation_id = record_frozen_invocation(conn, recorded)

    row = conn.execute("SELECT * FROM frozen_invocations WHERE id=?", (invocation_id,)).fetchone()
    assert set(row.keys()) == {"id"} | {f.name for f in fields(recorded)}
    for f in fields(recorded):
        value = getattr(recorded, f.name)
        assert row[f.name] == (int(value) if isinstance(value, bool) else value), f.name
    assert (row["timed_out"], row["stdout_exceeded"], row["stderr_truncated"]) == (1, 1, 1)
    assert frozen_invocation(conn, invocation_id) == recorded


def test_a_phase_b_attempt_and_a_pre_launch_refusal_are_recorded(conn):
    _sid, deployment = _deployment(conn)
    a = record_frozen_invocation(conn, attempt(deployment_id=deployment))
    b = record_frozen_invocation(conn, attempt(
        deployment_id=deployment, phase="b", phase_a_invocation_id=a, result_kind="decision"))
    refused = failed_attempt(
        deployment_id=deployment, request_id="b" * 32, failure_code="frozen_request_too_large",
        request_json=None, request_sha256=None, bars_sha256=None, bars_start=None,
        bars_end=None, signal=None, timed_out=False, diagnostic=None)
    r = record_frozen_invocation(conn, refused)

    assert a < b < r
    assert frozen_invocation(conn, r) == refused
    assert frozen_invocation(conn, b).phase_a_invocation_id == a  # type: ignore[union-attr]


def test_refuses_a_connection_with_an_open_transaction(conn):
    _sid, deployment = _deployment(conn)
    conn.execute("UPDATE strategies SET updated_at='u'")
    assert conn.in_transaction

    with pytest.raises(RuntimeError, match="open transaction"):
        record_frozen_invocation(conn, attempt(deployment_id=deployment))

    # Nothing was written and the caller's transaction was left for the caller to settle.
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM frozen_invocations").fetchone()[0] == 0


def test_the_transaction_check_survives_optimised_python(conn):
    """The check is an explicit raise, so ``python -O`` (which strips asserts) keeps it."""
    import ast
    import inspect

    from algua.registry.store import frozen_evidence

    tree = ast.parse(inspect.getsource(frozen_evidence))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_sqlite_errors_propagate_and_leave_no_open_transaction(conn):
    _sid, deployment = _deployment(conn)
    record_frozen_invocation(conn, attempt(deployment_id=deployment))

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        record_frozen_invocation(conn, failed_attempt(deployment_id=deployment))
    assert not conn.in_transaction
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        record_frozen_invocation(conn, attempt(deployment_id=deployment + 99,
                                               request_id="c" * 32))
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM frozen_invocations").fetchone()[0] == 1


def test_an_unknown_invocation_reads_as_none(conn):
    assert frozen_invocation(conn, 1) is None


# --- record_tick_snapshot's invocation link --------------------------------------------------


def _stamp(conn: sqlite3.Connection, name: str, strategy_id: int, deployment_id: int | None,
           **kw) -> None:
    record_tick_snapshot(
        conn, name, tick_ts="2026-09-30T20:00:00+00:00",
        decision_ts="2026-09-29T20:00:00+00:00", equity=1000.0, peak_equity=1000.0,
        positions={"AAPL": 1.0}, n_submitted=1, reconcile_ok=True, lane="paper",
        strategy_id=strategy_id, code_hash=IDENTITY[0], config_hash=IDENTITY[1],
        dependency_hash=IDENTITY[2], account_id="paper-account", cash=10.0,
        clock_source="broker", snapshot_id="snap-1", deployment_id=deployment_id, **kw)


def test_a_frozen_tick_stores_its_final_invocation_link(conn):
    sid, deployment = _deployment(conn)
    link = record_final_invocation(conn, deployment_id=deployment, snapshot_id="snap-1")

    _stamp(conn, "f", sid, deployment, frozen_invocation_id=link)

    row = conn.execute("SELECT deployment_id, snapshot_id, frozen_invocation_id"
                       " FROM tick_snapshots").fetchone()
    assert tuple(row) == (deployment, "snap-1", link)
    assert not conn.in_transaction


def test_a_frozen_tick_without_its_link_is_refused_by_the_schema(conn):
    sid, deployment = _deployment(conn)
    with pytest.raises(sqlite3.IntegrityError, match="must link its successful final"):
        _stamp(conn, "f", sid, deployment)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM tick_snapshots").fetchone()[0] == 0


def test_the_deployment_guard_still_runs_before_the_link(conn):
    sid, deployment = _deployment(conn)
    link = record_final_invocation(conn, deployment_id=deployment, snapshot_id="snap-1")
    with pytest.raises(DeploymentError, match="artifact identity"):
        record_tick_snapshot(
            conn, "f", tick_ts="t", decision_ts=None, equity=1.0, peak_equity=None,
            positions={}, n_submitted=0, reconcile_ok=True, lane="paper", strategy_id=sid,
            code_hash="e" * 32, config_hash=IDENTITY[1], dependency_hash=IDENTITY[2],
            account_id="a", cash=1.0, clock_source="broker", snapshot_id="snap-1",
            deployment_id=deployment, frozen_invocation_id=link)


def test_working_tree_and_legacy_ticks_record_unlinked_as_before(conn):
    from tests._deployment_helpers import force_legacy_strategy

    wt_sid, working_tree = _deployment(conn, "w", source_kind="working_tree")
    legacy_sid = conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES ('l','paper','t','t')"
    ).lastrowid
    assert legacy_sid is not None
    force_legacy_strategy(conn, legacy_sid)

    _stamp(conn, "w", wt_sid, working_tree)
    _stamp(conn, "l", legacy_sid, None)

    rows = conn.execute("SELECT strategy, deployment_id, frozen_invocation_id, positions,"
                        " n_submitted, cash FROM tick_snapshots ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("w", working_tree, None, '{"AAPL": 1.0}', 1, 10.0),
        ("l", None, None, '{"AAPL": 1.0}', 1, 10.0),
    ]


def test_a_working_tree_tick_cannot_carry_a_link(conn):
    wt_sid, working_tree = _deployment(conn, "w", source_kind="working_tree")
    _fsid, frozen = _deployment(conn)
    link = record_final_invocation(conn, deployment_id=frozen, snapshot_id="snap-1")
    with pytest.raises(sqlite3.IntegrityError, match="only a frozen tick"):
        _stamp(conn, "w", wt_sid, working_tree, frozen_invocation_id=link)
