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

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from algua.config.settings import get_settings

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


@contextmanager
def registry_conn() -> Iterator[sqlite3.Connection]:
    """Yield a migrated registry connection, closed on exit.

    The single idiom for opening the registry DB: connect + migrate + auto-close. Replaces the
    two competing forms (a bare ``_conn()`` and inline ``closing(connect(...))`` + ``migrate``).

    Lives on the facade rather than in ``connection.py`` because it pairs ``connect`` with
    ``migrate``, and this package's dependency graph is deliberately one-way
    (leaves -> schema -> migrate -> here); a leaf reaching up to ``migrate`` would invert it.
    It lives here rather than in ``cli/_common`` so non-CLI lanes (``algua.evaluation``) can open
    the registry without importing ``algua.cli`` -- hand-duplicating the four-line idiom to dodge
    that edge is exactly what this placement prevents.
    """
    conn = connect(get_settings().db_path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()
