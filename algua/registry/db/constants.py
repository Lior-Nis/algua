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
# accompanied by the corresponding migration step (a new table/index in _SCHEMA
# and/or a new entry in the `_add_missing_columns` calls in `migrate()`); never
# bump this number without the migration that earns it.
# v40 (simplification stage 1): the advisory shadow lane is deleted; migrate() drops its table.
# v41 (simplification stage 1): standalone factor-eval layer deleted; migrate() drops its table.
SCHEMA_VERSION = 41

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
