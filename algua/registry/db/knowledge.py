"""Knowledge/audit context: ``audit_log`` and ``negative_results``.

The immutable actor trail and the advisory failed-hypothesis experience log (#332, operated by
``algua/registry/negative_results.py``). Both are deliberately keyed by strategy NAME rather than a
``strategies(id)`` FK so they survive their strategy -- see the denormalization rationale above
``paper_orders`` in ``execution.py``.
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    strategy TEXT
);
-- v32 (#332): negative_results is an ADVISORY experience log capturing failed/rejected hypotheses
-- (gate FAILs, discards, research dead-ends) so knowledge is not lost with the branch. It NEVER
-- gates promotion and NEVER touches the live/paper path; it is written best-effort as a side effect
-- of the reject path and via a manual CLI. `gate_evaluation_id` is a NULLABLE advisory back-link to
-- the authoritative gate_evaluations row (not a hard FK — the log survives even if the reference is
-- unknown). CHECK constraints keep `kind`/`source` to their known vocabularies.
CREATE TABLE IF NOT EXISTS negative_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    strategy_name       TEXT,
    gate_evaluation_id  INTEGER,
    kind                TEXT NOT NULL CHECK (kind IN ('gate_fail', 'discard', 'dead_end')),
    verdict             TEXT NOT NULL,
    actor               TEXT NOT NULL,
    reason              TEXT NOT NULL,
    hypothesis          TEXT,
    params_json         TEXT,
    tags                TEXT,
    source              TEXT NOT NULL
        CHECK (source IN ('auto:research_promote', 'manual'))
);
CREATE INDEX IF NOT EXISTS ix_negative_results_strategy ON negative_results(strategy_name);
CREATE INDEX IF NOT EXISTS ix_negative_results_created ON negative_results(created_at);
CREATE INDEX IF NOT EXISTS ix_negative_results_kind ON negative_results(kind);
"""
