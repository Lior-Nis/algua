from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from algua.contracts.lifecycle import Actor, Stage, TransitionError
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

    @property
    def connection(self) -> sqlite3.Connection:
        """Read-only handle to the underlying sqlite connection, for protected verifiers (the
        live wall's forward-certificate check) that read operational tables alongside the
        repository. Deliberately NOT part of the ``StrategyRepository`` Protocol — the seam
        stays I/O-agnostic; non-sqlite repos must inject their own verifier."""
        return self._conn

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
                "EXISTS (SELECT 1 FROM json_each("
                "CASE WHEN json_valid(tags) THEN tags ELSE '[]' END"
                ") WHERE value = ?)"
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

    def default_fill_metadata_nulls(self) -> None:
        """Fill every strategy row's author/hypothesis_status/tags column from its default when
        still NULL. Used as the terminal step of the backfill-from-kb command. Idempotent.

        Runs in a single transaction so a partial run is not committed.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE strategies SET author = COALESCE(author, ?)",
                (Author.AGENT.value,),
            )
            self._conn.execute(
                "UPDATE strategies SET hypothesis_status = COALESCE(hypothesis_status, ?)",
                (HypothesisStatus.UNTESTED.value,),
            )
            self._conn.execute("UPDATE strategies SET tags = COALESCE(tags, '[]')")

    def delete(self, name: str) -> None:
        """Remove a strategy row and its transition rows. ONLY for rolling back a failed
        ``strategy new`` that just created it — there is no general deletion workflow."""
        rec = self.get(name)
        with self._conn:
            self._conn.execute(
                "DELETE FROM stage_transitions WHERE strategy_id = ?", (rec.id,)
            )
            self._conn.execute("DELETE FROM strategies WHERE id = ?", (rec.id,))

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
        consume_forward_gate_id: int | None = None,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
        if consume_gate_id is not None and consume_forward_gate_id is not None:
            raise ValueError(
                "at most one of consume_gate_id/consume_forward_gate_id may be set — a single"
                " transition spends a single token")
        with self._conn:  # consume + UPDATE + INSERT commit together or not at all
            return self._apply_transition_locked(
                rec, to, actor, reason, code_hash, config_hash, dependency_hash,
                consume_gate_id, consume_forward_gate_id, _now(),
                revoke_allocation=revoke_allocation)

    def _apply_transition_locked(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None,
        code_hash: str | None,
        config_hash: str | None,
        dependency_hash: str | None,
        consume_gate_id: int | None,
        consume_forward_gate_id: int | None,
        now: str,
        *,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
        """``apply_transition``'s body, WITHOUT opening a transaction: the caller owns the
        ``with self._conn:`` scope, so a composite write (e.g.
        ``record_forward_pass_and_promote``) can put extra statements in the SAME transaction
        as the token consume + stage CAS + transition INSERT."""
        from_stage = rec.stage
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
        if consume_forward_gate_id is not None:
            # Single-use, atomic with the stage change — same shape as the shortlist consume
            # above, EXCEPT it deliberately does NOT copy that block's lookup-trust: the WHERE
            # re-checks the FULL predicate set (identity, actor, passed, unconsumed, TTL) at
            # consume time, closing the validate-then-consume gap. The caller passes the
            # RECOMPUTED identity through code_hash/config_hash/dependency_hash; a NULL
            # dependency_hash never matches (fail-closed, mirroring has_valid_approval).
            from algua.research.forward_gates import FORWARD_TOKEN_TTL_DAYS
            cutoff = (datetime.now(UTC) - timedelta(days=FORWARD_TOKEN_TTL_DAYS)).isoformat()
            cur = self._conn.execute(
                "UPDATE forward_gate_evaluations SET consumed=1"
                " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0"
                " AND code_hash=? AND config_hash=? AND dependency_hash=? AND created_at>=?",
                (consume_forward_gate_id, rec.id, code_hash, config_hash,
                 dependency_hash, cutoff))
            if cur.rowcount != 1:
                raise TransitionError(
                    f"forward gate evaluation {consume_forward_gate_id} is not a consumable"
                    " agent token for this strategy+identity (already consumed, missing,"
                    " identity-drifted, or expired)")
        if revoke_allocation:
            # Bench wind-down (#125): revoke the live capital reservation in the SAME transaction
            # as the stage CAS below, so a raced/failed transition leaves the allocation intact.
            from algua.registry import allocations
            allocations.revoke_active_locked(self._conn, rec.id)
        # Compare-and-swap on the stage the caller read: two sessions sharing this DB must not
        # silently overwrite each other's transitions. Inside the txn, so a raced transition
        # rolls back the token consume above too.
        cur = self._conn.execute(
            "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ? AND stage = ?",
            (to.value, now, rec.id, from_stage.value),
        )
        if cur.rowcount != 1:
            raise TransitionError(
                f"concurrent transition detected for {rec.name!r}: stage is no longer"
                f" {from_stage.value!r} (another session moved it); re-read and retry")
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

    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        holdout_start: str,
        holdout_end: str,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        # TOP-LEVEL ONLY. A manual BEGIN IMMEDIATE inside an already-open transaction raises
        # "cannot start a transaction within a transaction", and a blanket rollback below could
        # roll back a caller's surrounding tx. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "reserve_holdout must be called at top level, not inside an open transaction")
        # Data identity: snapshot_id when the probe has one (a snapshot-backed row is a DISTINCT
        # identity from a non-snapshot probe), else data_source among rows lacking a snapshot.
        if snapshot_id is not None:
            data_match = "snapshot_id = ?"
            data_param: str = snapshot_id
        else:
            data_match = "snapshot_id IS NULL AND data_source = ?"
            data_param = data_source
        # Match identity is the OOS INTERVAL [holdout_start, holdout_end] — the exact bars
        # walk_forward burns (#192), NOT (full-period overlap, holdout_frac): a different
        # --holdout-frac that lands on overlapping OOS bars must NOT escape the guard. The standard
        # interval-overlap test (a.start <= b.end AND b.start <= a.end). A row with a NULL interval
        # (legacy/old-code reservation, before this column existed) matches UNCONDITIONALLY — fail
        # closed. period_*/holdout_frac are persisted as EVIDENCE only, never matched on. Matches
        # ALL rows (pending reservation OR committed burn) — no committed_at filter — so a pending
        # row blocks too.
        # BEGIN IMMEDIATE takes the write lock up front so the overlap SELECT + INSERT are one
        # atomic critical section: two concurrent reserves can't both see "no overlap" and both
        # insert. BaseException (not Exception) so a KeyboardInterrupt/SystemExit still releases the
        # lock via rollback.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                f"SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                f" AND {data_match}"
                f" AND (holdout_start IS NULL OR holdout_end IS NULL"
                f"      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, data_param, holdout_end, holdout_start),
            ).fetchone()
            overlap = row is not None
            if overlap and not allow_reuse:
                raise ValueError(
                    "holdout already consumed: an overlapping out-of-sample window was already "
                    "evaluated. Use fresh out-of-sample data, or --allow-holdout-reuse "
                    "(--actor human) to override and accept the statistical cost.")
            reused = bool(overlap)  # only reachable here with allow_reuse when overlap is True
            cur = self._conn.execute(
                "INSERT INTO holdout_evaluations"
                "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
                " config_hash, reused, created_at, committed_at, holdout_start, holdout_end)"
                " VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?)",
                (strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,
                 "", int(reused), _now(), holdout_start, holdout_end),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid, reused

    def finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str) -> None:
        with self._conn:  # UPDATE + guard commit together or roll back
            cur = self._conn.execute(
                "UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?"
                " WHERE id = ? AND committed_at IS NULL",
                (_now(), config_hash, reservation_id),
            )
            if cur.rowcount != 1:
                # Raise INSIDE the with so the mismatch rolls back (mirrors apply_transition's
                # gate-consume guard). Guards a double-finalize or a vanished/released row. Raise,
                # not assert — asserts strip under python -O.
                raise ValueError(
                    f"holdout reservation {reservation_id} is missing or already committed")

    def release_holdout_reservation(self, reservation_id: int) -> None:
        with self._conn:
            # No guard: a release after a finalize/crash is a harmless no-op (rowcount 0). Never
            # touches a committed burn (committed_at IS NULL filter).
            self._conn.execute(
                "DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL",
                (reservation_id,),
            )

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
        fundamentals_snapshot: str | None = None,
        news_snapshot: str | None = None,
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
                " fundamentals_snapshot, news_snapshot, consumed, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (strategy_id, int(passed), n_funnel, own_lifetime_combos, windowed_total_combos,
                 funnel_window_days, breadth_provenance, int(pit_ok), int(pit_override),
                 holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,
                 data_source, snapshot_id, period_start, period_end, holdout_frac, actor,
                 decision_json, fundamentals_snapshot, news_snapshot, _now()),
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

    def record_forward_gate_evaluation(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_forward_observations: int,
        min_forward_observations: int,
        session_coverage: float | None,
        realized_sharpe: float | None,
        holdout_sharpe: float | None,
        degradation_factor: float,
        sharpe_floor: float,
        realized_vol: float | None,
        min_forward_vol: float,
        realized_max_drawdown: float | None,
        max_forward_drawdown: float,
        first_tick_id: int | None,
        last_tick_id: int | None,
        first_tick_ts: str | None,
        last_tick_ts: str | None,
        max_staleness_sessions: int,
        n_reconcile_failures: int,
        n_concurrent_forward: int,
        account_id: str | None,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        actor: str,
        decision_json: str,
        consumable: bool,
    ) -> int:
        """Persist one forward-test gate evaluation (pass or fail) and return its row id. A
        passing AGENT row written ``consumable=True`` is the single-use token the paper ->
        forward_tested transition consumes; ``consumable=False`` writes the row already consumed
        — a CERTIFICATE for the live wall, never a re-entry token (#124 GATE-2)."""
        with self._conn:
            return self._insert_forward_gate_row_locked(
                strategy_id, passed=passed,
                n_forward_observations=n_forward_observations,
                min_forward_observations=min_forward_observations,
                session_coverage=session_coverage, realized_sharpe=realized_sharpe,
                holdout_sharpe=holdout_sharpe, degradation_factor=degradation_factor,
                sharpe_floor=sharpe_floor, realized_vol=realized_vol,
                min_forward_vol=min_forward_vol, realized_max_drawdown=realized_max_drawdown,
                max_forward_drawdown=max_forward_drawdown,
                first_tick_id=first_tick_id, last_tick_id=last_tick_id,
                first_tick_ts=first_tick_ts, last_tick_ts=last_tick_ts,
                max_staleness_sessions=max_staleness_sessions,
                n_reconcile_failures=n_reconcile_failures,
                n_concurrent_forward=n_concurrent_forward, account_id=account_id,
                code_hash=code_hash, config_hash=config_hash, dependency_hash=dependency_hash,
                actor=actor, decision_json=decision_json, consumed=0 if consumable else 1)

    def _insert_forward_gate_row_locked(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_forward_observations: int,
        min_forward_observations: int,
        session_coverage: float | None,
        realized_sharpe: float | None,
        holdout_sharpe: float | None,
        degradation_factor: float,
        sharpe_floor: float,
        realized_vol: float | None,
        min_forward_vol: float,
        realized_max_drawdown: float | None,
        max_forward_drawdown: float,
        first_tick_id: int | None,
        last_tick_id: int | None,
        first_tick_ts: str | None,
        last_tick_ts: str | None,
        max_staleness_sessions: int,
        n_reconcile_failures: int,
        n_concurrent_forward: int,
        account_id: str | None,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        actor: str,
        decision_json: str,
        consumed: int,
    ) -> int:
        """INSERT one forward-gate row inside the caller's already-open transaction (the caller
        owns the ``with self._conn:`` scope) and return its id."""
        cur = self._conn.execute(
            "INSERT INTO forward_gate_evaluations"
            "(strategy_id, passed, n_forward_observations, min_forward_observations,"
            " session_coverage, realized_sharpe, holdout_sharpe, degradation_factor,"
            " sharpe_floor, realized_vol, min_forward_vol, realized_max_drawdown,"
            " max_forward_drawdown, first_tick_id, last_tick_id, first_tick_ts, last_tick_ts,"
            " max_staleness_sessions, n_reconcile_failures, n_concurrent_forward, account_id,"
            " code_hash, config_hash, dependency_hash, actor, decision_json,"
            " consumed, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (strategy_id, int(passed), n_forward_observations, min_forward_observations,
             session_coverage, realized_sharpe, holdout_sharpe, degradation_factor,
             sharpe_floor, realized_vol, min_forward_vol, realized_max_drawdown,
             max_forward_drawdown, first_tick_id, last_tick_id, first_tick_ts, last_tick_ts,
             max_staleness_sessions, n_reconcile_failures, n_concurrent_forward, account_id,
             code_hash, config_hash, dependency_hash, actor, decision_json,
             consumed, _now()),
        )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def record_forward_pass_and_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        actor: Actor,
        reason: str | None = None,
    ) -> tuple[int, StrategyRecord]:
        """Record a PASSING forward-gate evaluation AND advance ``rec`` paper -> forward_tested
        in ONE sqlite transaction (#124 GATE-2). ``gate_row`` carries
        ``record_forward_gate_evaluation``'s row kwargs minus ``actor``/``consumable``; the row's
        actor column and the transition actor both come from ``actor``, so they can never drift.

        The row is born consumed (``consumed=1`` at INSERT — born-and-spent), REGARDLESS of
        actor: ``find_consumable_forward_gate_evaluation`` can never return it, while
        ``latest_forward_gate_row`` (the live wall's certificate selection) still sees it. For a
        human the observable effect is identical anyway (a human row was never consumable — the
        ``actor='agent'`` token filter); one uniform semantics, no per-actor branch. No consume
        UPDATE is needed or issued — the insert+CAS atomicity is what kills the old
        record-then-transition banking window, where a raced/failed transition left a committed
        ``consumed=0`` pass an agent could spend after a later demotion without a fresh gate run.

        The stage write is the same compare-and-swap as ``apply_transition``: if another session
        moved the stage since ``rec`` was read, ``TransitionError`` and the WHOLE transaction
        rolls back — including the row INSERT, so the loser leaves NO row at all. Its decision is
        lost on purpose: the winner's row is newer, and the loser's run can simply be re-executed
        against the new stage."""
        if not gate_row.get("passed"):
            raise ValueError(
                "record_forward_pass_and_promote is the PASS path only; record failing rows via"
                " record_forward_gate_evaluation")
        with self._conn:  # row INSERT + stage CAS + transition INSERT: one txn or nothing
            gate_id = self._insert_forward_gate_row_locked(
                rec.id, actor=actor.value, consumed=1, **gate_row)
            new_rec = self._apply_transition_locked(
                rec, Stage.FORWARD_TESTED, actor, reason,
                code_hash=gate_row["code_hash"], config_hash=gate_row["config_hash"],
                dependency_hash=gate_row["dependency_hash"],
                consume_gate_id=None, consume_forward_gate_id=None, now=_now())
        return gate_id, new_rec

    def find_consumable_forward_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        *,
        now: str,
        ttl_days: int,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed forward-gate row whose
        identity matches the recomputed current (code, config, dependency) AND whose created_at
        is within ``ttl_days`` of ``now`` — a stale token can never be banked. The
        ``actor='agent'`` filter means a human/override row is never an agent-consumable token.
        A NULL ``dependency_hash`` matches nothing — fail-closed, mirroring has_valid_approval.
        ISO-8601 UTC timestamps compare lexicographically in chronological order, so a string
        `>=` on created_at is correct."""
        if dependency_hash is None:
            return None
        cutoff = (datetime.fromisoformat(now) - timedelta(days=ttl_days)).isoformat()
        row = self._conn.execute(
            "SELECT id FROM forward_gate_evaluations WHERE strategy_id=? AND passed=1"
            " AND consumed=0 AND actor='agent' AND code_hash=? AND config_hash=?"
            " AND dependency_hash=? AND created_at>=? ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash, cutoff),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def latest_forward_gate_row(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> dict | None:
        """Return the newest forward-gate row (ALL columns, as a dict) for this strategy+identity
        regardless of passed/consumed, or None. This is the live wall's certificate selection —
        pass-or-fail ON PURPOSE: a newer FAILED re-evaluation must invalidate an older pass
        (#124), so the wall judges the latest verdict, never cherry-picks a stale success. A NULL
        ``dependency_hash`` matches nothing — fail-closed."""
        if dependency_hash is None:
            return None
        row = self._conn.execute(
            "SELECT * FROM forward_gate_evaluations WHERE strategy_id=? AND code_hash=?"
            " AND config_hash=? AND dependency_hash=? ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        return dict(row) if row is not None else None
