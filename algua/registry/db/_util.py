"""The one genuinely context-agnostic migration helper.

``_add_missing_columns`` is parameterised by table name and is called 15 times from ``migrate()``
across six different bounded contexts (core, execution, holdout, breadth, gate, family), which is
exactly why it has no context of its own.
"""
from __future__ import annotations

import sqlite3


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add any of ``columns`` (name -> column type) missing from ``table`` via ``ALTER TABLE``.

    Idempotent and cross-process safe: existing columns are skipped via introspection, and a
    concurrent process that adds the same column between our introspection and our ALTER (the
    lost-ALTER race) makes our ALTER raise ``duplicate column name`` — which we swallow, since the
    column now exists either way. New columns are added without a default, so on a populated table
    the existing rows get NULL — which the live/forward gates treat as fail-closed (a NULL
    ``dependency_hash`` can never match a recomputed concrete hash)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, col_type in columns.items():
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            except sqlite3.OperationalError as exc:
                # Lost the concurrent-ALTER race: another process added it first. Idempotent.
                if "duplicate column name" not in str(exc):
                    raise
