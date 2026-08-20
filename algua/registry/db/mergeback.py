"""Merge-back context: ``mergeback_evidence``.

The per-(strategy, branch_tip) evidence-production idempotency marker the autonomous merge-back
drainer writes, operated by ``algua/registry/mergeback_intake.py``.
"""
from __future__ import annotations

SCHEMA = """
-- v38 (merge-back authoritative intake): the per-(strategy, branch_tip) evidence marker the
-- merge-back drainer's produce_evidence uses for attempt-idempotency. search_trials/backtest_
-- returns writers are AUTOCOMMIT single-row inserts inside the reused sweep/backtest task
-- functions, so trial + returns + marker cannot land in one transaction; instead the marker is
-- written 'started' (with a search_trials MAX(id) watermark — read+inserted under ONE BEGIN
-- IMMEDIATE — and recipe_hash: a canonical hash over the FULL evidence recipe, grid AND data
-- context, so a resume with a drifted context fails closed instead of silently reusing the
-- marker) BEFORE the compute and flipped 'completed' AFTER both rows landed. A crash mid-compute
-- leaves 'started': the re-run dedups the trial layer on (strategy_name, grid_json,
-- id > watermark) — RESUME ONLY; a freshly-created marker always sweeps — duplicate trials are
-- NOT harmless (they permanently inflate funnel/window breadth and the agent-NOVEL lifetime
-- seed) — and a 'completed' marker blocks any re-record. UNIQUE(strategy_id, branch_tip):
-- one evidence production per merge-back attempt identity.
CREATE TABLE IF NOT EXISTS mergeback_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    branch_tip TEXT NOT NULL,
    recipe_hash TEXT NOT NULL,
    search_trials_watermark INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('started','completed')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(strategy_id, branch_tip)
);
"""
