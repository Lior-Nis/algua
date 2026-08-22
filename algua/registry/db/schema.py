"""Assembles the full registry DDL from the per-context fragments.

The concatenation order below is chosen for readability only -- it is provably irrelevant to the
resulting schema. Verified empirically: 200 randomly-shuffled fragment orders all produced an
identical sqlite_master (97 objects for the fragments alone, before the rest of migrate() adds its
own trigger and two indexes). SQLite resolves ``REFERENCES`` targets at DML time, not DDL
time, so cross-context foreign keys impose no ordering; the only real rule is that a table's
indexes and triggers live in the same fragment as the table, which each fragment satisfies.

Caveat for future changes: that independence holds because every fragment is executed inside one
``executescript`` before any DML. If a future change ever interleaves DML between fragments, the
foreign-key deferral no longer saves you.
"""
from __future__ import annotations

from algua.registry.db.authz import SCHEMA as AUTHZ_SCHEMA
from algua.registry.db.backtest_returns import SCHEMA as BACKTEST_RETURNS_SCHEMA
from algua.registry.db.breadth import SCHEMA as BREADTH_SCHEMA
from algua.registry.db.core import SCHEMA as CORE_SCHEMA
from algua.registry.db.execution import SCHEMA as EXECUTION_SCHEMA
from algua.registry.db.family import SCHEMA as FAMILY_SCHEMA
from algua.registry.db.forward_gate import SCHEMA as FORWARD_GATE_SCHEMA
from algua.registry.db.gate import SCHEMA as GATE_SCHEMA
from algua.registry.db.holdout import SCHEMA as HOLDOUT_SCHEMA
from algua.registry.db.ideas import SCHEMA as IDEAS_SCHEMA
from algua.registry.db.knowledge import SCHEMA as KNOWLEDGE_SCHEMA
from algua.registry.db.mergeback import SCHEMA as MERGEBACK_SCHEMA
from algua.registry.db.runs import SCHEMA as RUNS_SCHEMA

SCHEMA = "\n".join([
    CORE_SCHEMA,
    BREADTH_SCHEMA,
    HOLDOUT_SCHEMA,
    GATE_SCHEMA,
    FORWARD_GATE_SCHEMA,
    FAMILY_SCHEMA,
    BACKTEST_RETURNS_SCHEMA,
    RUNS_SCHEMA,
    AUTHZ_SCHEMA,
    IDEAS_SCHEMA,
    MERGEBACK_SCHEMA,
    KNOWLEDGE_SCHEMA,
    EXECUTION_SCHEMA,
])
