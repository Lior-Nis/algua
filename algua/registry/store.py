from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from algua.contracts.lifecycle import Actor, Stage
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.metadata import canonicalize_tags, dump_tags, load_tags
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
        family=row["family"],
        tags=load_tags(row["tags"]),
        author=Author(row["author"]) if row["author"] else Author.AGENT,
        hypothesis_status=(
            HypothesisStatus(row["hypothesis_status"])
            if row["hypothesis_status"] else HypothesisStatus.UNTESTED
        ),
        derived_from=row["derived_from"],
        description=row["description"],
    )


class SqliteStrategyRepository:
    """sqlite-backed ``StrategyRepository``: the only module that embeds registry SQL."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author = Author.AGENT,
        hypothesis_status: HypothesisStatus = HypothesisStatus.UNTESTED,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)  # raises StrategyNotFound if the parent is unknown
        now = _now()
        with self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO strategies"
                    "(name, stage, created_at, updated_at, family, tags, author,"
                    " hypothesis_status, derived_from, description)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name, Stage.IDEA.value, now, now, family, dump_tags(tags or []),
                     author.value, hypothesis_status.value, derived_from, description),
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

    def update_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
        derived_from: str | None = None,
        description: str | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> StrategyRecord:
        rec = self.get(name)
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)
        sets: list[str] = []
        params: list[object] = []
        if family is not None:
            sets.append("family = ?")
            params.append(family)
        if author is not None:
            sets.append("author = ?")
            params.append(author.value)
        if hypothesis_status is not None:
            sets.append("hypothesis_status = ?")
            params.append(hypothesis_status.value)
        if derived_from is not None:
            sets.append("derived_from = ?")
            params.append(derived_from)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if add_tags or remove_tags:
            tags = set(rec.tags)
            tags |= set(canonicalize_tags(add_tags or []))
            tags -= set(canonicalize_tags(remove_tags or []))
            sets.append("tags = ?")
            params.append(dump_tags(tags))
        if sets:
            sets.append("updated_at = ?")
            params.append(_now())
            params.append(rec.id)
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {', '.join(sets)} WHERE id = ?", params
                )
        return self.get(name)

    def list_strategies(
        self,
        stage: Stage | None = None,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
    ) -> list[StrategyRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage.value)
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        if author is not None:
            # COALESCE so legacy NULL rows (pre-metadata schema) match the default 'agent'.
            clauses.append("COALESCE(author, ?) = ?")
            params.extend((Author.AGENT.value, author.value))
        if hypothesis_status is not None:
            # Same NULL-legacy treatment; hypothesis_status defaults to 'untested'.
            clauses.append("COALESCE(hypothesis_status, ?) = ?")
            params.extend((HypothesisStatus.UNTESTED.value, hypothesis_status.value))
        for tag in canonicalize_tags(tags or []):
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(COALESCE(tags, '[]')) WHERE value = ?)"
            )
            params.append(tag)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM strategies{where} ORDER BY id", params
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def backfill_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: str | None = None,
        hypothesis_status: str | None = None,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        """Fill only currently-NULL metadata columns (one-shot recovery). Uses COALESCE so any
        column already holding a value is left untouched. Idempotent: re-running is a no-op."""
        cols: dict[str, object] = {
            "family": family,
            "tags": dump_tags(tags) if tags is not None else None,
            "author": author,
            "hypothesis_status": hypothesis_status,
            "derived_from": derived_from,
            "description": description,
        }
        # Filter to columns where the caller provided a non-None value.
        to_fill = {c: v for c, v in cols.items() if v is not None}
        if to_fill:
            rec = self.get(name)
            # COALESCE keeps any existing non-NULL value; only NULLs are filled.
            assignments = ", ".join(f"{c} = COALESCE({c}, ?)" for c in to_fill)
            params: list[object] = [*to_fill.values(), rec.id]
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {assignments} WHERE id = ?", params
                )
        return self.get(name)

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

    def record_search_trial(self, strategy_name: str, n_combos: int, grid_json: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
                " VALUES (?,?,?,?)",
                (strategy_name, n_combos, grid_json, _now()),
            )
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid

    def total_search_combos(self, strategy_name: str) -> int:
        # COALESCE so an empty result (no trials) reads as 0 rather than NULL.
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials WHERE strategy_name=?",
            (strategy_name,),
        ).fetchone()
        return int(row["total"])

    def record_holdout_evaluation(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        config_hash: str,
        reused: bool,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO holdout_evaluations"
                "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
                " config_hash, reused, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,
                 config_hash, int(reused), _now()),
            )
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid

    def overlapping_holdout_evaluations(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
    ) -> bool:
        # Data identity: when BOTH the probe and a stored row carry a snapshot_id, identity is the
        # snapshot_id; otherwise fall back to data_source equality. Period overlap is the standard
        # interval test (start1 <= end2 AND start2 <= end1). Match is on the window, never config.
        if snapshot_id is not None:
            data_match = "snapshot_id = ?"
            data_param: str = snapshot_id
        else:
            # Probe has no snapshot -> identity is the data_source (compare only rows that also
            # lack a snapshot, so a snapshot-backed row is a distinct identity, not a match).
            data_match = "snapshot_id IS NULL AND data_source = ?"
            data_param = data_source
        row = self._conn.execute(
            f"SELECT 1 FROM holdout_evaluations WHERE strategy_id = ? AND holdout_frac = ?"
            f" AND {data_match}"
            f" AND period_start <= ? AND ? <= period_end LIMIT 1",
            (strategy_id, holdout_frac, data_param, period_end, period_start),
        ).fetchone()
        return row is not None
