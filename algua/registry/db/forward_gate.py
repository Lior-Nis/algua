"""Forward-gate context: ``forward_gate_evaluations``.

The single-use agent-only PAPER->FORWARD_TESTED token ledger (#124), operated by
``algua/registry/store/forward_gate.py``. Kept separate from ``gate.py`` for a 1:1 correspondence
with ``store/gate.py`` / ``store/forward_gate.py``, so "the DDL for what ``store/forward_gate.py``
operates on" is findable without knowing the two were merged.
"""
from __future__ import annotations

SCHEMA = """
-- forward_gate_evaluations is the single-use AGENT-ONLY token ledger for the forward-test gate
-- (#124): each row records one gate evaluation (pass AND fail) and — for passing agent rows — is
-- the consumable token the PAPER->FORWARD_TESTED transition requires (mirroring gate_evaluations
-- for the BACKTESTED->CANDIDATE edge). A passing row is minted by the forward-gate run once the
-- strategy has accumulated enough forward-test observations; the transition consumes THAT row's id
-- in the same transaction as the stage change. Legacy NULL tick_snapshot rows (pre-v21) are
-- DELIBERATELY inadmissible as gate evidence — fail-closed, no backfill. FK into strategies(id).
-- NOTE: SQLite ALTER TABLE cannot add CHECK constraints, so lane/clock_source value discipline is
-- enforced by the writers (order_state.py), not the schema.
CREATE TABLE IF NOT EXISTS forward_gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    passed INTEGER NOT NULL,
    n_forward_observations INTEGER NOT NULL,
    min_forward_observations INTEGER NOT NULL,
    session_coverage REAL,
    realized_sharpe REAL,
    holdout_sharpe REAL,
    degradation_factor REAL NOT NULL,
    sharpe_floor REAL NOT NULL,
    realized_vol REAL,
    min_forward_vol REAL NOT NULL,
    realized_max_drawdown REAL,
    max_forward_drawdown REAL NOT NULL,
    first_tick_id INTEGER,
    last_tick_id INTEGER,
    first_tick_ts TEXT,
    last_tick_ts TEXT,
    max_staleness_sessions INTEGER NOT NULL,
    n_reconcile_failures INTEGER NOT NULL,
    n_concurrent_forward INTEGER NOT NULL,
    account_id TEXT,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dependency_hash TEXT,
    actor TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_forward_gate_strategy ON forward_gate_evaluations(strategy_id);

"""
