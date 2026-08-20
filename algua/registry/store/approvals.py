"""``ApprovalLedger`` — human-approval records that pin an artifact identity for the live gate."""
from __future__ import annotations

import sqlite3

from algua.registry.store._util import _now


class ApprovalLedgerMixin:
    _conn: sqlite3.Connection

    def record_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        approved_by: str,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO approvals"
                "(strategy_id, code_hash, config_hash, dependency_hash, approved_by, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (strategy_id, code_hash, config_hash, dependency_hash, approved_by, _now()),
            )
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid

    def has_valid_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> bool:
        # A NULL dependency_hash (no lockfile) can never pin a real artifact, so refuse it
        # outright rather than letting `= NULL` quietly never-match through the SQL.
        if dependency_hash is None:
            return False
        # `dependency_hash=?` against a pre-existing NULL column value yields no match, so legacy
        # approval rows written before this column existed fail closed — no `OR ... IS NULL`.
        row = self._conn.execute(
            "SELECT 1 FROM approvals WHERE strategy_id=? AND code_hash=? AND config_hash=?"
            " AND dependency_hash=? AND revoked_at IS NULL LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        return row is not None
