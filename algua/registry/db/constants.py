"""Schema-wide constants -- the two values that are not scoped to a single bounded context.

They live here rather than in the package facade so ``migrate.py`` and the context modules can
import them by full submodule path without importing ``algua.registry.db`` itself, which would be
circular at import time.
"""
from __future__ import annotations

# Identifies the current schema generation. This is a marker stamped into the
# DB's user_version, NOT a migration cursor: there is no per-version migration
# logic. `migrate()` is an idempotent bootstrap (CREATE TABLE/INDEX IF NOT EXISTS)
# that ALSO performs guarded in-place column additions via `_add_missing_columns`
# (PRAGMA table_info introspection + ALTER TABLE), so it can both add new
# tables/indexes AND add columns to an already-populated table. Adding a column
# is therefore the established pattern — but a SCHEMA_VERSION bump MUST be
# accompanied by the corresponding migration step (a new table/index in the relevant context
# module's `SCHEMA` fragment, assembled into schema.py's `SCHEMA`, and/or a new entry in the
# `_add_missing_columns` calls in `migrate()`); never bump this number without the migration
# that earns it.
# v40 (simplification stage 1): the advisory shadow lane is deleted; migrate() drops its table.
# v41 (simplification stage 1): standalone factor-eval layer deleted; migrate() drops its table.
# v42 (strategy run tracking): runs + run_metrics — the economic-layer evaluation ledger.
# v43 (strategy run tracking, fix wave): runs.gate_id — the join a `gate` run needs to name its
# own gate_evaluations row (decision_json, the per-check table, per-regime Sharpes).
SCHEMA_VERSION = 43

# v37 (#524, R9-M3): the per-search_trials-row upper bound on n_combos. A per-sweep combo count
# above any legitimate grid; bounds each summand of the funnel-lifetime seed SUM so it is
# overflow-safe (2^63/1e9 ≈ 9.2e9 well-typed rows would be needed to overflow). Enforced by the
# fresh-DB search_trials CHECK, record_search_trial's writer validation, and the WHERE-filtered
# mint seed SUM (which all use this exact bound).
# NB: this bound is ALSO hard-coded as the literal 1000000000 inside the search_trials CHECK
# constraint in ``breadth.py`` -- the DDL cannot interpolate a Python constant. Since the split the
# two live in different files, which makes the duplication easy to miss: changing this value means
# changing that literal too.
MAX_N_COMBOS = 1_000_000_000

# The generation of the FIXED metric vocabulary in `runs`. Stamped into every row so the
# vocabulary can evolve without silently changing what an existing chart means: a chart that
# needs v1 semantics filters on it rather than assuming. Bumping this means adding or
# re-defining a metric COLUMN, and therefore a SCHEMA_VERSION bump too.
METRIC_SCHEMA_VERSION = 1

# Per-sweep upper bound on PERSISTED sweep_trial rows. MAX_N_COMBOS above is a 1e9 overflow guard,
# not a realistic grid size; harmless while a trial was a scalar, but once each trial is a row it
# becomes a row-count bomb. Beyond this cap the writer keeps the search_trials aggregate (which
# still governs breadth) and stamps `trials_truncated_at` on the parent sweep run, so a reader can
# never mistake a truncated trial set for a complete one.
MAX_PERSISTED_TRIALS = 10_000
