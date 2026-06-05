from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from algua.contracts.lifecycle import Actor, Stage
from algua.registry.repository import StrategyExists, StrategyNotFound, StrategyRecord

__all__ = [
    "SqliteStrategyRepository",
    "StrategyExists",
    "StrategyNotFound",
    "StrategyRecord",
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"], name=row["name"], stage=Stage(row["stage"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


class SqliteStrategyRepository:
    """sqlite-backed ``StrategyRepository``: the only module that embeds registry SQL."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, name: str) -> StrategyRecord:
        now = _now()
        with self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
                    (name, Stage.IDEA.value, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise StrategyExists(name) from exc
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
            )
        return self.get(name)

    def get(self, name: str) -> StrategyRecord:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise StrategyNotFound(name)
        return _row_to_record(row)

    def list_strategies(self, stage: Stage | None = None) -> list[StrategyRecord]:
        if stage is None:
            rows = self._conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM strategies WHERE stage = ? ORDER BY id", (stage.value,)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_transitions(self, name: str) -> list[dict]:
        rec = self.get(name)
        rows = self._conn.execute(
            "SELECT * FROM stage_transitions WHERE strategy_id = ? ORDER BY id", (rec.id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
    ) -> StrategyRecord:
        from_stage = rec.stage
        now = _now()
        with self._conn:  # UPDATE + INSERT commit together or not at all
            self._conn.execute(
                "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ?",
                (to.value, now, rec.id),
            )
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, code_hash, config_hash,"
                " dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, from_stage.value, to.value, actor.value, reason,
                 code_hash, config_hash, dependency_hash, now),
            )
        return self.get(rec.name)

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
