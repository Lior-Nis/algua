from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from algua.contracts.lifecycle import Actor, Stage, TransitionError
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
        consume_gate_id: int | None = None,
    ) -> StrategyRecord:
        from_stage = rec.stage
        now = _now()
        with self._conn:  # consume + UPDATE + INSERT commit together or not at all
            if consume_gate_id is not None:
                # Single-use, atomic with the stage change: flipping the token, the stage UPDATE,
                # and the transition INSERT all live in this one transaction. If the token row was
                # already consumed or is missing, raise so the whole transition rolls back — the
                # stage can never advance on a vanished token, nor a token be spent without the
                # stage advancing.
                # NOTE: this UPDATE does NOT re-check artifact identity (code/config/dependency) —
                # identity binding lives in `find_consumable_gate_evaluation`, so callers MUST
                # always pair find->consume and never pass a hand-held id.
                cur = self._conn.execute(
                    "UPDATE gate_evaluations SET consumed=1"
                    " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0",
                    (consume_gate_id, rec.id))
                if cur.rowcount != 1:
                    raise TransitionError(
                        f"gate evaluation {consume_gate_id} is not a consumable agent token for "
                        f"this strategy (already consumed, missing, or mismatched)")
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

    def windowed_search_combos(self, window_days: int) -> int:
        """Sum of ``n_combos`` across ALL strategies' search_trials recorded within the trailing
        ``window_days`` (funnel-wide search effort for Wall A). ISO-8601 UTC timestamps compare
        lexicographically in chronological order, so a string `>=` on created_at is correct."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
        return int(row["total"])

    def record_gate_evaluation(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_funnel: int,
        own_lifetime_combos: int,
        windowed_total_combos: int,
        funnel_window_days: int,
        breadth_provenance: str,
        pit_ok: bool,
        pit_override: bool,
        holdout_n_bars: int,
        min_holdout_observations: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        actor: str,
        decision_json: str,
    ) -> int:
        """Persist one gate evaluation (pass or fail) and return its row id. A passing AGENT row is
        the single-use token the shortlist transition consumes."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " consumed, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (strategy_id, int(passed), n_funnel, own_lifetime_combos, windowed_total_combos,
                 funnel_window_days, breadth_provenance, int(pit_ok), int(pit_override),
                 holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,
                 data_source, snapshot_id, period_start, period_end, holdout_frac, actor,
                 decision_json, _now()),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def find_consumable_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed gate row whose identity matches
        the recomputed current (code, config, dependency), or None. The ``actor='agent'`` filter
        means a human/override promote's audit row is never an agent-consumable token. A NULL
        ``dependency_hash`` matches nothing — fail-closed, mirroring has_valid_approval."""
        if dependency_hash is None:
            return None
        row = self._conn.execute(
            "SELECT id FROM gate_evaluations WHERE strategy_id=? AND passed=1 AND consumed=0"
            " AND actor='agent' AND code_hash=? AND config_hash=? AND dependency_hash=?"
            " ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        return int(row["id"]) if row is not None else None
