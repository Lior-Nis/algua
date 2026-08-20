"""Search-breadth context: ``search_trials``.

The MEASURED per-sweep combination counts the promotion gate's multiple-testing defense scales on
(operated by ``algua/registry/store/search_breadth.py``). NB: the ``n_combos`` CHECK below
hard-codes ``1000000000``; that literal is ``algua.registry.db.constants.MAX_N_COMBOS``, duplicated
because DDL cannot interpolate a Python constant.
"""
from __future__ import annotations

SCHEMA = """
-- search_trials records the MEASURED search breadth of each parameter sweep so the promotion
-- gate's multiple-testing defense can scale on the real count of combinations tried, not a
-- self-reported flag. One row per `backtest sweep`: n_combos is the actual size of that sweep's
-- grid; grid_json is the JSON grid for the audit trail. The promotion gate sums n_combos across
-- all rows for a strategy (cumulative trials searched in the family — the conservative, honest
-- count).
-- KEYED BY strategy NAME (free text), NOT a strategies(id) FK, ON PURPOSE: a sweep can run
-- BEFORE a strategy is registered (exploration precedes registration). Keying by id would force
-- pre-registration sweeps to record nothing, letting an agent search broadly first and then
-- promote a freshly-registered strategy under a smaller DECLARED breadth — defeating the gate.
-- Keying by name lets those measured trials persist and be summed at promotion. (Same
-- denormalized-by-name rationale as paper_orders/audit_log above.)
-- INTENTIONAL: there is no grid deduplication. Re-running an identical sweep inserts another row
-- and permanently raises the cumulative count — and therefore the promotion bar. This is the
-- conservative choice: exploratory re-runs are real search effort and should count; silently
-- deduplicating them would quietly weaken the multiple-testing defense.
CREATE TABLE IF NOT EXISTS search_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    -- v37 (#524, R9-M3): type-safe + bounded so a corrupt/overlarge row can never overflow the
    -- funnel-lifetime seed SUM. Fresh DBs enforce it here; migrated DBs rely on the writer
    -- validation in record_search_trial + the WHERE-filtered mint SUM (ALTER cannot add a CHECK).
    n_combos INTEGER NOT NULL CHECK (typeof(n_combos)='integer' AND n_combos >= 1
                                     AND n_combos <= 1000000000),
    grid_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    trial_sharpe_count INTEGER,
    trial_sharpe_mean REAL,
    trial_sharpe_var_ann REAL
);
CREATE INDEX IF NOT EXISTS ix_search_trials_strategy ON search_trials(strategy_name);
CREATE INDEX IF NOT EXISTS ix_search_trials_created_at ON search_trials(created_at);
"""
