"""Run-ledger context: ``runs`` + ``run_metrics``.

Every evaluation of a strategy — backtest, walk-forward, sweep, each sweep trial, and each gate
decision — is one ``runs`` row with a FIXED metric vocabulary in real columns (operated by
``algua/registry/store/runs.py``). This is the economic layer: governance-bearing and
audit-bearing, as distinct from the component (model-training) layer, which lives in MLflow and is
best-effort. See the 2026-08-23 spec §2.

NB: the ``metric_schema_version`` column pins which vocabulary generation a row was written under,
so the vocabulary can evolve without silently changing what an existing chart means. It is
``algua.registry.db.constants.METRIC_SCHEMA_VERSION``, duplicated into every row rather than
assumed globally.
"""
from __future__ import annotations

SCHEMA = """
-- runs is the economic-layer evaluation ledger. One row per evaluation.
-- KEYED BY strategy NAME (free text) with a NULLABLE strategy_id, NOT a strategies(id) FK, ON
-- PURPOSE — the same rationale search_trials documents: exploration precedes registration, and a
-- backtest or sweep of a not-yet-registered strategy must still be recorded. Keying by id would
-- silently drop pre-registration evidence, which is exactly the evidence the breadth tax counts.
-- METRIC NAMING IS A CORRECTNESS RULE, NOT A STYLE: every metric column names its sample class
-- (_is / _oos / _realized). There are four different Sharpes in this system and they are not
-- comparable; a bare `sharpe` column would let a UI sort the most overfit number to the top.
-- A metric that is UNDEFINED for a run is NULL, never 0.0 — metrics_from_returns returns 0.0
-- sentinels for degenerate series, and recording those as zero would rank them above a genuine
-- negative. The writer maps sentinels to NULL; see algua/registry/runs.py.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN
        ('backtest', 'walk_forward', 'sweep', 'sweep_trial', 'gate')),
    strategy_name TEXT NOT NULL,
    strategy_id INTEGER,
    created_at TEXT NOT NULL,
    metric_schema_version INTEGER NOT NULL,

    -- Lineage. BOTH ARE JSON LISTS, ALWAYS — '[]' when empty, never NULL, never a scalar.
    -- derived_from: parent run ids in the economic layer (a gate run points at the walk-forward
    --   run it evaluated; a sweep_trial points at its parent sweep).
    -- components: component refs in the MODEL layer (name/version/digest/training_as_of). A list
    --   from day one: a strategy composed of several models must be expressible without a
    --   migration, even though strategies/base.py currently permits only one (spec §2.1).
    derived_from TEXT NOT NULL DEFAULT '[]',
    components TEXT NOT NULL DEFAULT '[]',

    -- Provenance, mirroring BacktestResult's fields one-for-one.
    code_hash TEXT,
    config_hash TEXT,
    dependency_hash TEXT,
    data_source TEXT,
    snapshot_id TEXT,
    universe_name TEXT,
    fundamentals_snapshot TEXT,
    news_snapshot TEXT,
    delisting_snapshot TEXT,
    seed INTEGER,
    timeframe TEXT,
    period_start TEXT,
    period_end TEXT,

    -- Free-form: grid points are heterogeneous by nature and freezing them is pointless.
    config_json TEXT NOT NULL DEFAULT '{}',

    -- Fixed metric vocabulary v1.
    sharpe_is REAL,
    sharpe_oos REAL,
    sharpe_realized REAL,
    sortino_is REAL,
    sortino_oos REAL,
    total_return_is REAL,
    total_return_oos REAL,
    max_drawdown_is REAL,
    max_drawdown_oos REAL,
    ann_vol_is REAL,
    ann_vol_oos REAL,
    cagr_is REAL,
    calmar_is REAL,
    n_obs_is INTEGER,
    n_obs_oos INTEGER,
    mean_window_sharpe REAL,
    std_window_sharpe REAL,
    min_window_sharpe REAL,
    pct_positive_windows REAL,

    -- Gate outcome (NULL for non-gate kinds).
    passed INTEGER,
    -- The gate_evaluations row THIS run derives from. NULL for every non-`gate` kind. The join to
    -- gate_evaluations.decision_json — the per-check table (the 11 gate checks, binding vs
    -- advisory) and the per-regime Sharpes that this row's own fixed scalar columns deliberately
    -- do not carry. A `gate` run's gate_id is set at the SAME BEGIN IMMEDIATE that writes the
    -- gate_evaluations row it names, so the two ids are never mismatched (v43).
    gate_id INTEGER,
    -- Series pointers (slice 2, D1). The run row points AT its series rather than the series
    -- tables carrying a run_id: `record_holdout_returns` is idempotent on exact content and sits
    -- on the promote path, so widening its identity key is delicate, and this keeps the migration
    -- on the one table this feature owns. NULL is honest and common: an unregistered strategy's
    -- backtest records a run but no `backtest_returns` row, and rows written before v44 have no
    -- pointer at all (spec Q8 — no backfill; `runs series` reports "no series" rather than
    -- guessing by provenance, which is non-unique across re-runs).
    series_backtest_id INTEGER,
    series_holdout_id INTEGER,
    -- Set on a `sweep` parent when its trial rows were capped at MAX_PERSISTED_TRIALS. NULL means
    -- the trial set is COMPLETE. A silently truncated set would make the funnel-wide distribution
    -- lie about the breadth it depicts, so a reader must be able to tell.
    trials_truncated_at INTEGER
);
CREATE INDEX IF NOT EXISTS ix_runs_strategy ON runs(strategy_name);
CREATE INDEX IF NOT EXISTS ix_runs_kind_created ON runs(kind, created_at);
CREATE INDEX IF NOT EXISTS ix_runs_sharpe_oos ON runs(sharpe_oos);

-- run_metrics is the OVERFLOW key-value tail: the ~40 DSR/IR/regime diagnostics and whatever a
-- future model-bearing run wants. Queryable, but deliberately NOT part of the fixed vocabulary
-- and NOT offered as a default chart axis — a metric that matters enough to sort by should earn
-- a real column and a metric_schema_version bump.
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_run_metrics_key ON run_metrics(key);
"""
