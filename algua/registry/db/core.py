"""Core registry context: ``strategies``, ``stage_transitions``, ``approvals``.

The normalized relational spine the lifecycle runs on (operated by ``algua/registry/store/crud.py``
and ``store/approvals.py``), plus the one-time ``shortlisted`` -> ``candidate`` stage rename (#120).
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    from_stage TEXT,
    to_stage TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    code_hash TEXT,
    config_hash TEXT,
    -- dependency_hash mirrors code_hash/config_hash: it is the locked-dependency identity pinned
    -- by the live gate, recorded here so the "what was promoted to live" audit trail carries the
    -- full (code, config, dependency) identity. NULL for non-live transitions (no hashes), exactly
    -- as code_hash/config_hash are.
    dependency_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    -- dependency_hash is nullable on purpose: rows written before this column existed carry
    -- NULL and MUST never satisfy the live gate (fail-closed), and `has_valid_approval` refuses
    -- a NULL probe outright. New approvals always write a concrete hash.
    dependency_hash TEXT,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
"""


def _migrate_shortlisted_to_candidate(conn: sqlite3.Connection) -> None:
    """Rewrite the renamed lifecycle stage value `shortlisted` -> `candidate` (#120) in the typed
    stage columns. Runs BEFORE the `CREATE TABLE IF NOT EXISTS` bootstrap, so each table is guarded
    independently — a fresh DB has neither table yet. Idempotent: the `WHERE` matches nothing on a
    second run, and it does NOT gate on `user_version`, so a DB already stamped at the new version
    but still holding `shortlisted` rows is still corrected.

    Only the typed `stage` / `from_stage` / `to_stage` columns are rewritten — the free-text audit
    trail (`audit_log`, `stage_transitions.reason`) and `gate_evaluations.decision_json` are
    immutable history and intentionally left as written."""
    def _has(table: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is not None
        )

    def _has_col(table: str, col: str) -> bool:
        return any(
            row[1] == col
            for row in conn.execute(f"PRAGMA table_info({table})")
        )

    if _has("strategies") and _has_col("strategies", "stage"):
        conn.execute("UPDATE strategies SET stage='candidate' WHERE stage='shortlisted'")
    if _has("stage_transitions"):
        if _has_col("stage_transitions", "from_stage"):
            conn.execute(
                "UPDATE stage_transitions SET from_stage='candidate'"
                " WHERE from_stage='shortlisted'"
            )
        if _has_col("stage_transitions", "to_stage"):
            conn.execute(
                "UPDATE stage_transitions SET to_stage='candidate' WHERE to_stage='shortlisted'"
            )
