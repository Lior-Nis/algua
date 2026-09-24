"""Privileged fixture helpers for the immutable v47 migration cohort.

Production has no post-migration enrollment path. Tests that exercise compatibility behavior must
therefore make the privilege escalation explicit by temporarily dropping and restoring the schema
trigger, rather than teaching runtime code a back door.
"""
from __future__ import annotations

import sqlite3

_LATE_INSERT_TRIGGER = """
CREATE TRIGGER trg_legacy_deployment_strategies_no_late_insert
BEFORE INSERT ON legacy_deployment_strategies
WHEN EXISTS (SELECT 1 FROM deployment_migrations WHERE id=1)
BEGIN
    SELECT RAISE(ABORT, 'legacy deployment cohort was fixed at migration');
END
"""


def force_legacy_strategy(
    conn: sqlite3.Connection, strategy_id: int, *, stage: str = "paper",
) -> None:
    """Fabricate a migration-time tenant for compatibility-only tests."""
    conn.execute("DROP TRIGGER IF EXISTS trg_legacy_deployment_strategies_no_late_insert")
    try:
        conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, strategy_id))
        conn.execute(
            "INSERT OR IGNORE INTO legacy_deployment_strategies"
            "(strategy_id, original_stage, marked_at) VALUES (?,?,?)",
            (strategy_id, stage, "test-fixture"),
        )
    finally:
        conn.execute(_LATE_INSERT_TRIGGER)
    conn.commit()
