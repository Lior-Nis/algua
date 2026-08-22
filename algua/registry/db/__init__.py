"""sqlite registry schema + bootstrap, split by bounded context (spec §8).

Each context owns its DDL fragment and its own migration helpers (core, breadth, holdout, gate,
forward_gate, family, backtest_returns, authz, ideas, mergeback, knowledge, execution);
``schema.py`` concatenates the fragments; ``migrate.py`` holds the single ordered bootstrap
sequence; ``connection.py`` owns ``connect()`` and its pragmas; ``constants.py`` holds the
schema-wide constants; ``_util.py`` holds the one context-agnostic ALTER helper.

Submodules import each other by full path and never import this facade -- the dependency graph is
one-way (leaves -> schema -> migrate -> here), which is what keeps the package importable.
"""
from __future__ import annotations

# Compatibility re-export, NOT public API (hence its absence from __all__):
# tests/test_db_migrations.py imports _add_missing_columns from this path directly.
from algua.registry.db._util import _add_missing_columns as _add_missing_columns
from algua.registry.db.connection import connect as connect
from algua.registry.db.constants import (
    MAX_N_COMBOS as MAX_N_COMBOS,
)
from algua.registry.db.constants import (
    MAX_PERSISTED_TRIALS as MAX_PERSISTED_TRIALS,
)
from algua.registry.db.constants import (
    METRIC_SCHEMA_VERSION as METRIC_SCHEMA_VERSION,
)
from algua.registry.db.constants import (
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from algua.registry.db.gate import (
    FDR_COHORT_SIZE as FDR_COHORT_SIZE,
)
from algua.registry.db.gate import (
    fdr_cohort_position as fdr_cohort_position,
)
from algua.registry.db.migrate import migrate as migrate

__all__ = [
    "FDR_COHORT_SIZE",
    "MAX_N_COMBOS",
    "MAX_PERSISTED_TRIALS",
    "METRIC_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "connect",
    "fdr_cohort_position",
    "migrate",
]
