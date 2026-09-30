"""Frozen planner invocation evidence (Story 1.3d, v48): ``frozen_invocations`` and the tick link.

The DDL below is the protected schema review of the Story 1.3d contract
(``docs/development/specs/spec-story-1-3d-frozen-evidence-and-qualification/
frozen-evidence-contract.md`` §2) and is reproduced from it verbatim. Change it only through that
contract.

``SCHEMA`` joins the ``executescript`` bootstrap in ``schema.py``. The tick link is different:
``tick_snapshots`` already exists on every populated database, so ``migrate()`` adds
``TICK_LINK_COLUMN`` with a guarded ALTER after the v47 block and only then executes
``TICK_LINK_STATEMENTS`` (the index and triggers reference that column), following the
family_members ALTER-then-trigger precedent. Each statement is idempotent (``IF NOT EXISTS``).

Recording evidence with a raw ``sqlite3.connect()`` is covered too: ``INSERT OR REPLACE`` would
resolve a conflict by deleting the old row without firing the delete trigger when
``recursive_triggers`` is OFF, so ``frozen_invocations_no_replace`` refuses any colliding insert
before conflict resolution runs.
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS frozen_invocations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id         INTEGER NOT NULL REFERENCES strategy_deployments(id),
    request_id            TEXT    NOT NULL CHECK (length(request_id) = 32),
    phase                 TEXT    NOT NULL CHECK (phase IN ('a', 'b')),
    phase_a_invocation_id INTEGER REFERENCES frozen_invocations(id),
    snapshot_id           TEXT    NOT NULL,
    bars_start            TEXT,
    bars_end              TEXT,
    request_json          TEXT CHECK (request_json IS NULL OR length(CAST(request_json AS BLOB)) <= 262144),
    request_sha256        TEXT CHECK (request_sha256 IS NULL OR length(request_sha256) = 64),
    bars_sha256           TEXT CHECK (bars_sha256 IS NULL OR length(bars_sha256) = 64),
    phase_a_binding       TEXT,
    result_kind           TEXT CHECK (result_kind IS NULL OR result_kind IN (
                              'early_no_decision', 'snapshot_required', 'risk_failure',
                              'late_no_decision', 'decision')),
    result_sha256         TEXT CHECK (result_sha256 IS NULL OR length(result_sha256) = 64),
    failure_code          TEXT CHECK (failure_code IS NULL OR failure_code IN (
                              'frozen_content_unavailable', 'frozen_content_unsupported',
                              'frozen_request_too_large', 'frozen_launch_failed', 'frozen_timeout',
                              'frozen_exit_abnormal', 'frozen_output_exceeded',
                              'frozen_result_invalid', 'frozen_planner_rejected',
                              'frozen_live_unsupported')),
    returncode            INTEGER,
    signal                INTEGER,
    timed_out             INTEGER NOT NULL DEFAULT 0 CHECK (timed_out IN (0, 1)),
    stdout_exceeded       INTEGER NOT NULL DEFAULT 0 CHECK (stdout_exceeded IN (0, 1)),
    stderr_truncated      INTEGER NOT NULL DEFAULT 0 CHECK (stderr_truncated IN (0, 1)),
    diagnostic            TEXT CHECK (diagnostic IS NULL OR length(diagnostic) <= 8192),
    started_at            TEXT NOT NULL,
    ended_at              TEXT NOT NULL,
    CHECK ((result_sha256 IS NULL) <> (failure_code IS NULL)),
    CHECK ((result_sha256 IS NULL) = (result_kind IS NULL)),
    CHECK ((request_json IS NULL) = (request_sha256 IS NULL)),
    CHECK ((phase = 'b') = (phase_a_invocation_id IS NOT NULL)),
    CHECK (diagnostic IS NULL OR failure_code IS NOT NULL),
    CHECK (result_kind IS NOT 'snapshot_required' OR phase_a_binding IS NOT NULL),
    UNIQUE (request_id, phase)
);

CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_update
BEFORE UPDATE ON frozen_invocations
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_delete
BEFORE DELETE ON frozen_invocations
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

-- INSERT OR REPLACE would delete-and-reinsert without firing the delete trigger when recursive
-- triggers are off (a raw connection); refuse any insert that collides with an existing row.
CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_replace
BEFORE INSERT ON frozen_invocations
WHEN EXISTS (SELECT 1 FROM frozen_invocations
             WHERE id = NEW.id OR (request_id = NEW.request_id AND phase = NEW.phase))
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

-- A Phase B attempt must follow a successful Phase A attempt of the same tick and deployment,
-- and carry the binding that Phase A produced.
CREATE TRIGGER IF NOT EXISTS frozen_invocations_phase_b_follows_a
BEFORE INSERT ON frozen_invocations WHEN NEW.phase = 'b'
BEGIN
    SELECT RAISE(ABORT, 'phase b must follow a successful phase a of the same tick')
    WHERE NOT EXISTS (
        SELECT 1 FROM frozen_invocations a
        WHERE a.id = NEW.phase_a_invocation_id AND a.phase = 'a'
          AND a.deployment_id = NEW.deployment_id AND a.request_id = NEW.request_id
          AND a.result_kind = 'snapshot_required'
          AND a.phase_a_binding IS NEW.phase_a_binding
          AND a.snapshot_id IS NEW.snapshot_id);
END;
"""  # noqa: E501 -- the contract's DDL verbatim (one CHECK line exceeds 100 columns)

#: The one column v48 adds to ``tick_snapshots`` (guarded ALTER in ``migrate()``).
TICK_LINK_COLUMN = {"frozen_invocation_id": "INTEGER REFERENCES frozen_invocations(id)"}

TICK_LINK_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS tick_snapshots_one_tick_per_invocation
    ON tick_snapshots(frozen_invocation_id) WHERE frozen_invocation_id IS NOT NULL
"""

TICK_FROZEN_LINK_TRIGGER = """
-- A tick of a frozen deployment must link a successful final (phase b) invocation of the same
-- deployment and snapshot; any other tick must not link one.
CREATE TRIGGER IF NOT EXISTS tick_snapshots_frozen_link
BEFORE INSERT ON tick_snapshots
BEGIN
    SELECT RAISE(ABORT, 'a frozen tick must link its successful final invocation')
    WHERE EXISTS (
        SELECT 1 FROM strategy_deployments d JOIN deployment_artifacts x ON x.id = d.artifact_id
        WHERE d.id = NEW.deployment_id AND x.source_kind = 'frozen')
      AND NOT EXISTS (
        SELECT 1 FROM frozen_invocations i
        WHERE i.id = NEW.frozen_invocation_id AND i.phase = 'b'
          AND i.deployment_id = NEW.deployment_id
          AND i.result_kind IN ('decision', 'late_no_decision')
          AND i.snapshot_id IS NEW.snapshot_id);
    SELECT RAISE(ABORT, 'only a frozen tick may link a frozen invocation')
    WHERE NEW.frozen_invocation_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM strategy_deployments d JOIN deployment_artifacts x ON x.id = d.artifact_id
        WHERE d.id = NEW.deployment_id AND x.source_kind = 'frozen');
END
"""

TICK_LINK_IMMUTABLE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS tick_snapshots_link_immutable
BEFORE UPDATE OF frozen_invocation_id, deployment_id, snapshot_id, strategy_id ON tick_snapshots
BEGIN SELECT RAISE(ABORT, 'a tick''s deployment and invocation link cannot change'); END
"""

#: Executed in order by ``migrate()`` after ``TICK_LINK_COLUMN`` exists.
TICK_LINK_STATEMENTS = (TICK_LINK_INDEX, TICK_FROZEN_LINK_TRIGGER, TICK_LINK_IMMUTABLE_TRIGGER)
