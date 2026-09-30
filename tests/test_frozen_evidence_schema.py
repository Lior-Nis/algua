"""Story 1.3d §2: the v48 frozen invocation evidence schema and the tick-to-invocation link.

Every constraint is exercised against a database built by the repository's real ``migrate()``:
the ``frozen_invocations`` CHECKs, its append-only triggers (including both ``INSERT OR REPLACE``
forms on a raw connection with ``recursive_triggers`` OFF), the Phase B trigger, the tick link
triggers and index, link immutability, and the forward-only migration from a genuine v47 database.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from algua.registry.db import SCHEMA_VERSION, connect, migrate
from tests._frozen_evidence_helpers import seed_deployment

FAILURE_CODES = (
    "frozen_content_unavailable", "frozen_content_unsupported", "frozen_request_too_large",
    "frozen_launch_failed", "frozen_timeout", "frozen_exit_abnormal", "frozen_output_exceeded",
    "frozen_result_invalid", "frozen_planner_rejected", "frozen_live_unsupported",
)
RESULT_KINDS = (
    "early_no_decision", "snapshot_required", "risk_failure", "late_no_decision", "decision",
)
APPEND_ONLY = "frozen invocation evidence is append-only"
PHASE_B = "phase b must follow a successful phase a of the same tick"
MUST_LINK = "a frozen tick must link its successful final invocation"
ONLY_FROZEN = "only a frozen tick may link a frozen invocation"
LINK_IMMUTABLE = "a tick's deployment and invocation link cannot change"

_SUCCESS: dict[str, Any] = {
    "deployment_id": 1, "request_id": "a" * 32, "phase": "a", "phase_a_invocation_id": None,
    "snapshot_id": "snap-1", "bars_start": "2026-06-01T00:00:00+00:00",
    "bars_end": "2026-09-29T00:00:00+00:00", "request_json": '{"request":"a"}',
    "request_sha256": "1" * 64, "bars_sha256": "2" * 64, "phase_a_binding": "binding-1",
    "result_kind": "snapshot_required", "result_sha256": "3" * 64, "failure_code": None,
    "returncode": 0, "signal": None, "timed_out": 0, "stdout_exceeded": 0,
    "stderr_truncated": 0, "diagnostic": None,
    "started_at": "2026-09-30T20:00:00+00:00", "ended_at": "2026-09-30T20:00:01+00:00",
}
_FAILURE: dict[str, Any] = {
    "result_kind": None, "result_sha256": None, "failure_code": "frozen_timeout",
    "returncode": None, "signal": 9, "timed_out": 1, "diagnostic": "timed out",
}


class World:
    """A migrated registry: frozen deployments 1 and 3, working-tree deployment 2, legacy 'l'."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = connect(path)
        migrate(self.conn)
        self.frozen_sid, self.frozen = seed_deployment(self.conn, "f", source_kind="frozen")
        self.wt_sid, self.working_tree = seed_deployment(
            self.conn, "w", source_kind="working_tree")
        self.other_sid, self.other_frozen = seed_deployment(self.conn, "g", source_kind="frozen")
        self.legacy_sid = self.conn.execute(
            "INSERT INTO strategies(name, stage, created_at, updated_at)"
            " VALUES ('l','paper','t','t')").lastrowid
        self.conn.commit()
        self._requests = 0

    def insert(self, conn: sqlite3.Connection | None = None, *, verb: str = "INSERT",
               **overrides: Any) -> int:
        row = {**_SUCCESS, "deployment_id": self.frozen, **overrides}
        columns = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        target = conn or self.conn
        cur = target.execute(
            f"{verb} INTO frozen_invocations({columns}) VALUES ({marks})", tuple(row.values()))
        target.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def request_id(self) -> str:
        self._requests += 1
        return f"{self._requests:032x}"

    def phase_a(self, **overrides: Any) -> int:
        return self.insert(**{"request_id": overrides.pop("request_id", None)
                              or self.request_id(), **overrides})

    def final(self, *, deployment_id: int | None = None, snapshot_id: str = "snap-1",
              result_kind: str = "decision") -> int:
        """A successful phase a + phase b pair; returns the phase b id."""
        deployment = self.frozen if deployment_id is None else deployment_id
        rid = self.request_id()
        a = self.insert(deployment_id=deployment, request_id=rid, snapshot_id=snapshot_id)
        return self.insert(deployment_id=deployment, request_id=rid, phase="b",
                           phase_a_invocation_id=a, snapshot_id=snapshot_id,
                           result_kind=result_kind)

    def tick(self, *, deployment_id: int | None, strategy_id: int | None = None,
             snapshot_id: str | None = "snap-1", link: int | None = None) -> int:
        sid = strategy_id if strategy_id is not None else {
            self.frozen: self.frozen_sid, self.working_tree: self.wt_sid,
            self.other_frozen: self.other_sid, None: self.legacy_sid,
        }[deployment_id]
        name = self.conn.execute("SELECT name FROM strategies WHERE id=?", (sid,)).fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
            " n_submitted, reconcile_ok, lane, strategy_id, snapshot_id, deployment_id,"
            " frozen_invocation_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, "2026-09-30T20:00:00+00:00", "2026-09-29T20:00:00+00:00", 1000.0, "{}", 0,
             1, "paper", sid, snapshot_id, deployment_id, link),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)


@pytest.fixture
def world(tmp_path: Path):
    w = World(tmp_path / "r.db")
    yield w
    w.conn.close()


def _refused(world: World, match: str, **overrides: Any) -> None:
    with pytest.raises(sqlite3.IntegrityError, match=match):
        world.insert(**overrides)
    world.conn.rollback()


# --- frozen_invocations: shape and CHECKs ---------------------------------------------------


def test_a_valid_success_and_a_valid_failure_are_accepted(world):
    success = world.phase_a()
    failure = world.phase_a(**_FAILURE)
    rows = world.conn.execute(
        "SELECT id, result_kind, failure_code FROM frozen_invocations ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        (success, "snapshot_required", None), (failure, None, "frozen_timeout")]


@pytest.mark.parametrize("column", [
    "deployment_id", "request_id", "phase", "snapshot_id", "started_at", "ended_at",
])
def test_required_columns_are_not_null(world, column):
    _refused(world, "NOT NULL constraint failed", **{column: None})


@pytest.mark.parametrize("request_id", ["a" * 31, "a" * 33, ""])
def test_request_id_is_exactly_32_characters(world, request_id):
    _refused(world, "CHECK constraint failed", request_id=request_id)


def test_phase_vocabulary_is_a_or_b(world):
    _refused(world, "CHECK constraint failed", phase="c")


@pytest.mark.parametrize("column", ["request_sha256", "bars_sha256", "result_sha256"])
@pytest.mark.parametrize("length", [63, 65])
def test_digests_are_exactly_64_characters(world, column, length):
    _refused(world, "CHECK constraint failed", **{column: "f" * length})


def test_request_json_is_bounded_in_bytes_not_characters(world):
    world.phase_a(request_json="x" * 262144)
    # 131073 two-byte characters: under the limit in characters, over it in UTF-8 bytes.
    _refused(world, "CHECK constraint failed", request_id=world.request_id(),
             request_json="é" * 131073)
    _refused(world, "CHECK constraint failed", request_id=world.request_id(),
             request_json="x" * 262145)


@pytest.mark.parametrize("kind", RESULT_KINDS)
def test_every_result_kind_in_the_vocabulary_is_accepted(world, kind):
    world.phase_a(result_kind=kind)


def test_an_unknown_result_kind_is_refused(world):
    _refused(world, "CHECK constraint failed", result_kind="refused")


@pytest.mark.parametrize("code", FAILURE_CODES)
def test_every_failure_code_in_the_vocabulary_is_accepted(world, code):
    world.phase_a(**{**_FAILURE, "failure_code": code})


def test_an_unknown_failure_code_is_refused(world):
    _refused(world, "CHECK constraint failed", **{**_FAILURE, "failure_code": "frozen_crashed"})


@pytest.mark.parametrize("column", ["timed_out", "stdout_exceeded", "stderr_truncated"])
def test_flags_are_zero_or_one(world, column):
    _refused(world, "CHECK constraint failed", **{**_FAILURE, column: 2})


def test_flags_default_to_zero(world):
    row = {k: v for k, v in _SUCCESS.items()
           if k not in {"timed_out", "stdout_exceeded", "stderr_truncated"}}
    row["deployment_id"] = world.frozen
    world.conn.execute(
        f"INSERT INTO frozen_invocations({', '.join(row)}) VALUES"
        f" ({', '.join('?' for _ in row)})", tuple(row.values()))
    assert tuple(world.conn.execute(
        "SELECT timed_out, stdout_exceeded, stderr_truncated FROM frozen_invocations"
    ).fetchone()) == (0, 0, 0)


def test_diagnostic_is_bounded_to_8_kib(world):
    world.phase_a(**{**_FAILURE, "diagnostic": "d" * 8192})
    _refused(world, "CHECK constraint failed", request_id=world.request_id(),
             **{**_FAILURE, "diagnostic": "d" * 8193})


@pytest.mark.parametrize("overrides", [
    pytest.param({"failure_code": "frozen_timeout"}, id="success-and-failure"),
    pytest.param({**_FAILURE, "failure_code": None, "diagnostic": None}, id="neither"),
])
def test_exactly_one_of_result_and_failure(world, overrides):
    _refused(world, "CHECK constraint failed", **overrides)


@pytest.mark.parametrize("overrides", [
    pytest.param({"result_kind": None}, id="result-without-kind"),
    pytest.param({**_FAILURE, "result_kind": "decision"}, id="kind-without-result"),
])
def test_a_result_kind_accompanies_exactly_a_result(world, overrides):
    _refused(world, "CHECK constraint failed", **overrides)


@pytest.mark.parametrize("overrides", [
    pytest.param({"request_json": None}, id="digest-without-bytes"),
    pytest.param({"request_sha256": None}, id="bytes-without-digest"),
])
def test_request_bytes_and_digest_are_recorded_together(world, overrides):
    _refused(world, "CHECK constraint failed", **overrides)


def test_a_pre_launch_refusal_records_no_request_bytes(world):
    world.phase_a(**{**_FAILURE, "failure_code": "frozen_request_too_large",
                     "request_json": None, "request_sha256": None, "bars_sha256": None,
                     "signal": None, "timed_out": 0, "bars_start": None, "bars_end": None})


def test_only_a_phase_b_row_names_a_phase_a_row(world):
    a = world.phase_a()
    _refused(world, "CHECK constraint failed", request_id=world.request_id(),
             phase_a_invocation_id=a)
    # A phase b row without its phase a id: the BEFORE INSERT trigger refuses it before the CHECK.
    _refused(world, PHASE_B, request_id=world.request_id(), phase="b")


def test_only_a_failure_carries_a_diagnostic(world):
    _refused(world, "CHECK constraint failed", diagnostic="x")


def test_snapshot_required_needs_a_binding(world):
    _refused(world, "CHECK constraint failed", phase_a_binding=None)
    world.phase_a(result_kind="early_no_decision", phase_a_binding=None)


def test_snapshot_id_is_required_even_on_a_failure(world):
    _refused(world, "NOT NULL constraint failed", snapshot_id=None, **_FAILURE)


# --- frozen_invocations: append-only ---------------------------------------------------------


@pytest.mark.parametrize("assignment", [
    "result_kind='decision'", "failure_code=NULL", "diagnostic='x'", "request_json='{}'",
    "snapshot_id='other'", "id=id",
])
def test_rows_cannot_be_updated(world, assignment):
    world.phase_a()
    with pytest.raises(sqlite3.IntegrityError, match=APPEND_ONLY):
        world.conn.execute(f"UPDATE frozen_invocations SET {assignment}")


def test_rows_cannot_be_deleted(world):
    world.phase_a()
    with pytest.raises(sqlite3.IntegrityError, match=APPEND_ONLY):
        world.conn.execute("DELETE FROM frozen_invocations")


def test_a_duplicate_request_and_phase_is_refused(world):
    rid = world.request_id()
    world.phase_a(request_id=rid)
    _refused(world, APPEND_ONLY, request_id=rid, **_FAILURE)


@pytest.mark.parametrize("verb", ["INSERT OR REPLACE", "REPLACE"])
@pytest.mark.parametrize("collision", ["id", "request-and-phase"])
def test_replace_cannot_rewrite_a_row_on_a_raw_connection(world, verb, collision):
    """``INSERT OR REPLACE`` deletes-then-inserts without firing the delete trigger when
    ``recursive_triggers`` is OFF (a raw connection's default); the insert trigger refuses it."""
    rid = world.request_id()
    original = world.phase_a(request_id=rid)
    raw = sqlite3.connect(world.path)
    try:
        assert raw.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        clash = ({"id": original, "request_id": world.request_id()} if collision == "id"
                 else {"request_id": rid})
        with pytest.raises(sqlite3.IntegrityError, match=APPEND_ONLY):
            world.insert(raw, verb=verb, **{**_FAILURE, **clash})
        raw.rollback()
    finally:
        raw.close()
    rows = world.conn.execute("SELECT id, request_id, result_kind FROM frozen_invocations")
    assert [tuple(r) for r in rows] == [(original, rid, "snapshot_required")]


# --- frozen_invocations: phase b follows a successful phase a --------------------------------


def _phase_b(world: World, a: int, rid: str, **overrides: Any) -> int:
    return world.insert(**{"deployment_id": world.frozen, "request_id": rid, "phase": "b",
                           "phase_a_invocation_id": a, "result_kind": "decision",
                           **overrides})


def test_a_phase_b_row_following_its_successful_phase_a_is_accepted(world):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    _phase_b(world, a, rid)
    rid2 = world.request_id()
    a2 = world.phase_a(request_id=rid2)
    _phase_b(world, a2, rid2, **_FAILURE)  # a failed phase b is still recorded


def test_phase_b_without_its_phase_a_is_refused(world):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    with pytest.raises(sqlite3.IntegrityError, match=PHASE_B):
        _phase_b(world, a + 100, rid)


@pytest.mark.parametrize("phase_a", [
    pytest.param(_FAILURE, id="failed"),
    pytest.param({"result_kind": "early_no_decision"}, id="settled-early"),
    pytest.param({"result_kind": "risk_failure"}, id="risk-failure"),
])
def test_phase_b_after_a_phase_a_that_did_not_require_a_snapshot_is_refused(world, phase_a):
    rid = world.request_id()
    a = world.phase_a(request_id=rid, **phase_a)
    with pytest.raises(sqlite3.IntegrityError, match=PHASE_B):
        _phase_b(world, a, rid)


def test_phase_b_naming_a_phase_b_row_is_refused(world):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    b = _phase_b(world, a, rid)
    with pytest.raises(sqlite3.IntegrityError, match=PHASE_B):
        _phase_b(world, b, world.request_id())


@pytest.mark.parametrize("mismatch", ["deployment", "request", "binding", "no-binding",
                                      "snapshot"])
def test_phase_b_must_match_its_phase_a(world, mismatch):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    overrides: dict[str, Any] = {
        "deployment": {"deployment_id": world.other_frozen},
        "request": {"request_id": world.request_id()},
        "binding": {"phase_a_binding": "binding-2"},
        "no-binding": {"phase_a_binding": None},
        "snapshot": {"snapshot_id": "snap-2"},
    }[mismatch]
    with pytest.raises(sqlite3.IntegrityError, match=PHASE_B):
        _phase_b(world, a, rid, **overrides)


# --- tick_snapshots: the link to the final invocation ----------------------------------------


@pytest.mark.parametrize("kind", ["decision", "late_no_decision"])
def test_a_frozen_tick_linking_its_successful_final_invocation_is_accepted(world, kind):
    link = world.final(result_kind=kind)
    tick = world.tick(deployment_id=world.frozen, link=link)
    assert world.conn.execute(
        "SELECT frozen_invocation_id FROM tick_snapshots WHERE id=?", (tick,)
    ).fetchone()[0] == link


def test_a_frozen_tick_without_a_link_is_refused(world):
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen)


def test_a_frozen_tick_linking_a_failed_final_invocation_is_refused(world):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    failed = _phase_b(world, a, rid, **_FAILURE)
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen, link=failed)


@pytest.mark.parametrize("kind", ["early_no_decision", "risk_failure", "snapshot_required"])
def test_a_frozen_tick_linking_a_non_decision_final_invocation_is_refused(world, kind):
    rid = world.request_id()
    a = world.phase_a(request_id=rid)
    b = _phase_b(world, a, rid, result_kind=kind)
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen, link=b)


def test_a_frozen_tick_linking_a_phase_a_invocation_is_refused(world):
    a = world.phase_a()
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen, link=a)


def test_a_frozen_tick_linking_another_deployments_invocation_is_refused(world):
    foreign = world.final(deployment_id=world.other_frozen)
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen, link=foreign)


@pytest.mark.parametrize("tick_snapshot", ["snap-2", None])
def test_a_frozen_tick_on_another_snapshot_is_refused(world, tick_snapshot):
    link = world.final(snapshot_id="snap-1")
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        world.tick(deployment_id=world.frozen, snapshot_id=tick_snapshot, link=link)


def test_working_tree_and_legacy_ticks_without_a_link_are_accepted(world):
    world.tick(deployment_id=world.working_tree)
    world.tick(deployment_id=None)
    world.tick(deployment_id=world.working_tree, snapshot_id=None)
    rows = world.conn.execute(
        "SELECT deployment_id, frozen_invocation_id FROM tick_snapshots ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        (world.working_tree, None), (None, None), (world.working_tree, None)]


@pytest.mark.parametrize("tenant", ["working_tree", "legacy"])
def test_a_non_frozen_tick_with_a_link_is_refused(world, tenant):
    link = world.final()
    deployment = world.working_tree if tenant == "working_tree" else None
    with pytest.raises(sqlite3.IntegrityError, match=ONLY_FROZEN):
        world.tick(deployment_id=deployment, link=link)


def test_one_tick_per_final_invocation(world):
    link = world.final()
    world.tick(deployment_id=world.frozen, link=link)
    with pytest.raises(sqlite3.IntegrityError,
                       match="UNIQUE constraint failed: tick_snapshots.frozen_invocation_id"):
        world.tick(deployment_id=world.frozen, link=link)


@pytest.mark.parametrize("assignment", [
    "frozen_invocation_id=NULL", "frozen_invocation_id=frozen_invocation_id",
    "deployment_id=NULL", "deployment_id=deployment_id", "snapshot_id='snap-2'",
    "snapshot_id=snapshot_id", "strategy_id=strategy_id + 1",
])
def test_a_ticks_link_columns_cannot_change(world, assignment):
    world.tick(deployment_id=world.frozen, link=world.final())
    world.tick(deployment_id=world.working_tree)
    with pytest.raises(sqlite3.IntegrityError, match=LINK_IMMUTABLE):
        world.conn.execute(f"UPDATE tick_snapshots SET {assignment}")


def test_a_ticks_other_columns_stay_updatable(world):
    frozen = world.tick(deployment_id=world.frozen, link=world.final())
    working_tree = world.tick(deployment_id=world.working_tree)
    world.conn.execute(
        "UPDATE tick_snapshots SET positions='{\"AAPL\": 1.0}', recorded_at='r', equity=1.0")
    world.conn.commit()
    rows = world.conn.execute(
        "SELECT id, positions, recorded_at, equity FROM tick_snapshots ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [
        (frozen, '{"AAPL": 1.0}', "r", 1.0), (working_tree, '{"AAPL": 1.0}', "r", 1.0)]


# --- migration: v47 -> v48 -----------------------------------------------------------------

# The v47 schema fingerprint pinned by tests/test_registry_db.py before this story. The v47
# database below is rebuilt from today's schema minus exactly the v48 objects and is required to
# reproduce it, so the migration is tested from a genuine v47 shape.
_V47_OBJECT_COUNT = 128
_V47_DIGEST = "6f52c8265450481b00af17838947879e169c50712b5169d94d77dca8eec51c36"
_V48_OBJECTS = frozenset({
    "frozen_invocations", "sqlite_autoindex_frozen_invocations_1",
    "frozen_invocations_no_update", "frozen_invocations_no_delete",
    "frozen_invocations_no_replace", "frozen_invocations_phase_b_follows_a",
    "tick_snapshots_one_tick_per_invocation", "tick_snapshots_frozen_link",
    "tick_snapshots_link_immutable",
})
_LINK_COLUMN = ", frozen_invocation_id INTEGER REFERENCES frozen_invocations(id)"


def _fingerprint(conn: sqlite3.Connection) -> tuple[int, str]:
    rows = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    dump = "\n".join(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3] or ''}" for r in rows)
    return len(rows), hashlib.sha256(dump.encode()).hexdigest()


def _v47_database(tmp_path: Path) -> sqlite3.Connection:
    fresh = connect(tmp_path / "fresh.db")
    migrate(fresh)
    objects = fresh.execute(
        "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"
        " ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, rowid").fetchall()
    fresh.close()
    conn = connect(tmp_path / "v47.db")
    for name, sql in objects:
        if name in _V48_OBJECTS or name == "sqlite_sequence":
            continue
        if name == "tick_snapshots":
            assert sql.count(_LINK_COLUMN) == 1
            sql = sql.replace(_LINK_COLUMN, "")
        conn.execute(sql)
    conn.execute("INSERT INTO deployment_migrations(id, legacy_cohort_captured_at) VALUES (1,'t')")
    conn.execute("PRAGMA user_version=47")
    conn.commit()
    assert _fingerprint(conn) == (_V47_OBJECT_COUNT, _V47_DIGEST)
    return conn


def _v47_tick(conn: sqlite3.Connection, strategy_id: int, deployment_id: int | None) -> None:
    conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
        " n_submitted, reconcile_ok, lane, strategy_id, snapshot_id, deployment_id)"
        " SELECT name, '2026-09-29T20:00:00+00:00', NULL, 1000.0, '{}', 0, 1, 'paper', id,"
        " 'snap-0', ? FROM strategies WHERE id=?",
        (deployment_id, strategy_id),
    )
    conn.commit()


def test_migration_from_v47_keeps_existing_ticks_unlinked_and_is_idempotent(tmp_path):
    conn = _v47_database(tmp_path)
    frozen_sid, frozen = seed_deployment(conn, "f", source_kind="frozen")
    wt_sid, working_tree = seed_deployment(conn, "w", source_kind="working_tree")
    legacy_sid = conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES ('l','paper','t','t')"
    ).lastrowid
    assert legacy_sid is not None
    _v47_tick(conn, frozen_sid, frozen)  # a 1.3c-era frozen tick: no link, never backfilled
    _v47_tick(conn, wt_sid, working_tree)
    _v47_tick(conn, legacy_sid, None)
    before = [tuple(r) for r in conn.execute("SELECT * FROM tick_snapshots ORDER BY id")]

    fresh = connect(tmp_path / "fresh-v48.db")
    migrate(fresh)
    for _ in range(2):
        migrate(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 48
        assert _fingerprint(conn) == _fingerprint(fresh)
        rows = conn.execute("SELECT * FROM tick_snapshots ORDER BY id").fetchall()
        assert [tuple(r)[:-1] for r in rows] == before
        assert [r["frozen_invocation_id"] for r in rows] == [None, None, None]
    fresh.close()

    # The migrated database enforces the link: a new unlinked frozen tick is refused ...
    with pytest.raises(sqlite3.IntegrityError, match=MUST_LINK):
        _v47_tick(conn, frozen_sid, frozen)
    conn.rollback()
    # ... while working-tree and legacy ticks record exactly as before.
    _v47_tick(conn, wt_sid, working_tree)
    _v47_tick(conn, legacy_sid, None)
    # The pre-existing unlinked frozen tick stays, and its link columns are now immutable.
    with pytest.raises(sqlite3.IntegrityError, match=LINK_IMMUTABLE):
        conn.execute("UPDATE tick_snapshots SET frozen_invocation_id=NULL WHERE id=1")
    conn.close()
