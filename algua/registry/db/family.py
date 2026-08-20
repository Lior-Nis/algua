"""Family-governance context: ``families``, ``family_members``, ``family_parents``,
``family_events``.

The #222/#524 family DAG and its append-only triggers (operated by
``algua/registry/store/family.py``). Cross-module coupling: ``families.founder_gate_id`` REFERENCES
``gate_evaluations(id)``, which lives in ``gate.py`` -- a further split must not separate the two
carelessly. The #524 append-only rationale in the trigger comment below also covers
``backtest_returns``' two triggers, which live in ``backtest_returns.py`` with their table.
"""
from __future__ import annotations

SCHEMA = """
-- v26 (#222): family registry tables.
-- families: canonical family registry
CREATE TABLE IF NOT EXISTS families (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    created_at       TEXT NOT NULL,
    created_by_actor TEXT NOT NULL,
    created_by_strategy TEXT,
    -- v37 (#524): durable LIFETIME breadth prior for a family. Seeded (>0) only when an agent
    -- founds a NOVEL family at the pass moment (§5.1), so it starts as if it had already
    -- accumulated the funnel-wide LIFETIME test effort (removes the reset-gaming incentive). It
    -- is a lifetime-only prior (NOT in windowed sums). Human NOVEL/PARENTAGE creates keep 0.
    seeded_prior_combos INTEGER NOT NULL DEFAULT 0 CHECK (seeded_prior_combos >= 0),
    -- v37 (#524, R9-M2): FK to the founding gate_evaluations row (the gate that founded an
    -- agent NOVEL family). NULL for legacy + human-created families (not agent-gate-founded).
    founder_gate_id  INTEGER REFERENCES gate_evaluations(id)
);
-- family_members: APPEND-ONLY (removed_at SET never DELETE; breadth never decreases).
-- v37 (#524, R9-H3): member_code_hash/member_factors_json persist the (code_hash, sorted
-- factors) the member was classified under, MATERIALISED AT ASSIGNMENT. IMMUTABLE once
-- non-NULL (append-only trigger below): the classifier's member-profile input is DB state,
-- transactionally covered by family_graph_fingerprint. NULL only on un-materialised legacy rows.
CREATE TABLE IF NOT EXISTS family_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id       INTEGER NOT NULL REFERENCES families(id),
    strategy_name   TEXT NOT NULL,
    joined_at       TEXT NOT NULL,
    joined_by_actor TEXT NOT NULL,
    removed_at      TEXT,
    member_code_hash    TEXT,
    member_factors_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_family_members_strategy_family
    ON family_members(strategy_name, family_id) WHERE removed_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_family_members_active
    ON family_members(strategy_name) WHERE removed_at IS NULL;
-- family_parents: parentage DAG (multi-parent; cycle-guarded at write time)
CREATE TABLE IF NOT EXISTS family_parents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    child_family_id  INTEGER NOT NULL REFERENCES families(id),
    parent_family_id INTEGER NOT NULL REFERENCES families(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_family_parents
    ON family_parents(child_family_id, parent_family_id);
-- family_events: governance audit log
CREATE TABLE IF NOT EXISTS family_events (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type              TEXT NOT NULL,
    family_id               INTEGER REFERENCES families(id),
    strategy_name           TEXT,
    actor                   TEXT NOT NULL,
    clustering_verdict      TEXT,
    similarity_score        REAL,
    clustering_version      TEXT,
    clustering_config_json  TEXT,
    axis_json               TEXT,
    matched_family_id       INTEGER REFERENCES families(id),
    created_at              TEXT NOT NULL
);
-- v37 (#524, R9-H1): make the classifier read-set append-only IN THE ENGINE, not just by
-- discipline. families / family_parents / family_events / backtest_returns are pure INSERT-append
-- → forbid UPDATE and DELETE on all four. family_members permits exactly two one-way UPDATEs: the
-- removed_at tombstone flip (NULL→ts) and the one-time legacy profile materialisation
-- (member_code_hash/member_factors_json NULL→value); everything else ABORTs. An EXPLICIT
-- DELETE or UPDATE is aborted through ANY connection (store API, repair tool, raw shell) —
-- those fire the BEFORE triggers unconditionally. The IMPLICIT delete SQLite performs to resolve
-- an `INSERT/REPLACE ... OR REPLACE` conflict, however, only fires the BEFORE DELETE trigger when
-- `PRAGMA recursive_triggers=ON` — a PER-CONNECTION setting that db.connect() turns on but a raw
-- sqlite3.connect() bypassing that helper does NOT get. So the REPLACE-append-only guarantee holds
-- only for connections opened via db.connect() (every production path); a repair tool that opens a
-- raw handle without recursive_triggers could still REPLACE a row in place. Within that scope,
-- family_graph_fingerprint's monotone (COUNT, MAX(id)) digest is exact.
CREATE TRIGGER IF NOT EXISTS trg_families_append_only_upd BEFORE UPDATE ON families
  BEGIN SELECT RAISE(ABORT, 'families is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_families_append_only_del BEFORE DELETE ON families
  BEGIN SELECT RAISE(ABORT, 'families is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_parents_append_only_upd BEFORE UPDATE ON family_parents
  BEGIN SELECT RAISE(ABORT, 'family_parents is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_parents_append_only_del BEFORE DELETE ON family_parents
  BEGIN SELECT RAISE(ABORT, 'family_parents is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_events_append_only_upd BEFORE UPDATE ON family_events
  BEGIN SELECT RAISE(ABORT, 'family_events is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_events_append_only_del BEFORE DELETE ON family_events
  BEGIN SELECT RAISE(ABORT, 'family_events is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_members_no_delete BEFORE DELETE ON family_members
  BEGIN SELECT RAISE(ABORT, 'family_members is append-only (#524)'); END;
-- NB: trg_family_members_append_only_upd references the v37 member_code_hash/member_factors_json
-- columns, which do NOT exist on a legacy family_members table at executescript(SCHEMA) time
-- (they are added by ALTER in migrate() afterwards). That trigger is therefore created in
-- migrate() AFTER the ALTER, not here -- defensively: SQLite actually resolves a trigger's column
-- references when the trigger FIRES, not at CREATE time (verified on 3.45.1), so creating it early
-- would not raise here, but it would leave a window in which the trigger is unfireable. Keeping
-- creation after the ALTER keeps the guarantee unconditional.
"""
