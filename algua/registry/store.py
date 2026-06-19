from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.metadata import canonicalize_tags, dump_tags, load_tags
from algua.registry.repository import (
    FdrGateOutcome,
    FdrStreamState,
    FunnelFloor,
    StrategyExists,
    StrategyNotFound,
    StrategyRecord,
)
from algua.research.gates import MIN_FUNNEL_FLOOR_STRATEGIES

__all__ = [
    "SqliteStrategyRepository",
    "StrategyExists",
    "StrategyNotFound",
    "StrategyRecord",
]


def _pool_trial_sharpe_var(triples: list[tuple[int, float, float]]) -> float | None:
    """Exact pooled SAMPLE variance (ddof=1) of trial Sharpes from ``(count, mean, var)`` triples.
    ``None`` for empty input; ``0.0`` for total count <= 1. Callers must pre-validate each triple
    (finite mean/var, count >= 1, var >= 0); this helper assumes clean triples."""
    if not triples:
        return None
    total_n = sum(n for n, _, _ in triples)
    if total_n <= 1:
        return 0.0
    grand_mean = sum(n * m for n, m, _ in triples) / total_n
    sse = sum((n - 1) * v + n * (m - grand_mean) ** 2 for n, m, v in triples)
    return sse / (total_n - 1)


def _validated_triples(rows) -> list[tuple[int, float, float]] | None:
    """Validate raw (n, mean, var) DB rows. Returns None (fail closed) if ANY row has a
    NULL/NaN/inf/negative stat — NULL rows are NEVER silently skipped."""
    triples: list[tuple[int, float, float]] = []
    for r in rows:
        n, mean, var = r["n"], r["mean"], r["var"]
        if n is None or mean is None or var is None:
            return None
        if not (math.isfinite(mean) and math.isfinite(var)) or int(n) < 1 or var < 0.0:
            return None
        triples.append((int(n), float(mean), float(var)))
    return triples


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
        # One dict drives both the clause list and the param list, so a column and its value can
        # never drift out of lockstep — adding a field is a single line, not two parallel edits.
        updates: dict[str, object] = {}
        if family is not None:
            updates["family"] = family
        if author is not None:
            updates["author"] = author.value
        if hypothesis_status is not None:
            updates["hypothesis_status"] = hypothesis_status.value
        if derived_from is not None:
            updates["derived_from"] = derived_from
        if description is not None:
            updates["description"] = description
        if add_tags or remove_tags:
            tags = set(rec.tags)
            tags |= set(canonicalize_tags(add_tags or []))
            tags -= set(canonicalize_tags(remove_tags or []))
            updates["tags"] = dump_tags(tags)
        if updates:
            updates["updated_at"] = _now()
            clauses = ", ".join(f"{col} = ?" for col in updates)
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {clauses} WHERE id = ?",
                    [*updates.values(), rec.id],
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
        # Each clause carries its OWN params as a co-located tuple, so a multi-placeholder clause
        # (e.g. the COALESCE pairs below) can never fall out of sync with a separate param list.
        clauses: list[tuple[str, tuple[object, ...]]] = []
        if stage is not None:
            clauses.append(("stage = ?", (stage.value,)))
        if family is not None:
            clauses.append(("family = ?", (family,)))
        if author is not None:
            # COALESCE so legacy NULL rows (pre-metadata schema) match the default 'agent'.
            clauses.append(("COALESCE(author, ?) = ?", (Author.AGENT.value, author.value)))
        if hypothesis_status is not None:
            # Same NULL-legacy treatment; hypothesis_status defaults to 'untested'.
            clauses.append((
                "COALESCE(hypothesis_status, ?) = ?",
                (HypothesisStatus.UNTESTED.value, hypothesis_status.value),
            ))
        for tag in canonicalize_tags(tags or []):
            clauses.append((
                "EXISTS (SELECT 1 FROM json_each("
                "CASE WHEN json_valid(tags) THEN tags ELSE '[]' END"
                ") WHERE value = ?)",
                (tag,),
            ))
        where = f" WHERE {' AND '.join(c for c, _ in clauses)}" if clauses else ""
        params = [p for _, clause_params in clauses for p in clause_params]
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

    def record_search_trial(
        self, strategy_name: str, n_combos: int, grid_json: str,
        *, trial_sharpe_count: int | None = None,
        trial_sharpe_mean: float | None = None,
        trial_sharpe_var_ann: float | None = None,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at,"
                " trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann)"
                " VALUES (?,?,?,?,?,?,?)",
                (strategy_name, n_combos, grid_json, _now(),
                 trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann),
            )
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid

    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        """Exact pooled SAMPLE variance (ddof=1) of the strategy's trial Sharpes across all its
        search_trials rows. Returns None (fail closed) if there are no rows OR any contributing
        row has a NULL/NaN/inf/negative count|mean|var. NULL rows are NEVER silently skipped."""
        rows = self._conn.execute(
            "SELECT trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name=?",
            (strategy_name,),
        ).fetchall()
        if not rows:
            return None
        triples = _validated_triples(rows)
        if triples is None:
            return None
        return _pool_trial_sharpe_var(triples)

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
        # Single-use identity is the OOS INTERVAL [holdout_start, holdout_end] for the strategy,
        # PROVENANCE-INDEPENDENT (#205): the same OOS calendar window is burn-once regardless of how
        # the bars were reached (snapshot S, a different snapshot S2, or a provider P). data_source/
        # snapshot_id are persisted as EVIDENCE only, never matched on (was: a per-provenance
        # bucket, which let the same physical window be burned twice across provenance — #205).
        # Defensive (GATE-1): an inverted incoming interval (start > end) would slip both the NULL
        # branch and the overlap test below and fail OPEN, so reject it. holdout_window yields a
        # well-formed interval (idx[train_n] <= idx[-1]); this guards the primitive vs. a caller.
        if holdout_start > holdout_end:
            raise ValueError(
                f"invalid holdout interval: start {holdout_start!r} > end {holdout_end!r}")
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
                "SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                " AND (holdout_start IS NULL OR holdout_end IS NULL"
                "      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, holdout_end, holdout_start),
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

    def finalize_holdout_reservation(
        self, reservation_id: int, *, config_hash: str, strategy_id: int
    ) -> None:
        with self._conn:  # UPDATE + guard commit together or roll back
            cur = self._conn.execute(
                "UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?"
                " WHERE id = ? AND strategy_id = ? AND committed_at IS NULL",
                (_now(), config_hash, reservation_id, strategy_id),
            )
            if cur.rowcount != 1:
                # Raise INSIDE the with so the mismatch rolls back (mirrors apply_transition's
                # gate-consume guard). Guards a double-finalize or a vanished/released row. Raise,
                # not assert — asserts strip under python -O.
                raise ValueError(
                    f"holdout reservation {reservation_id} is missing, already committed, "
                    f"or strategy_id mismatch")

    def release_holdout_reservation(self, reservation_id: int) -> None:
        with self._conn:
            # No guard: a release after a finalize/crash is a harmless no-op (rowcount 0). Never
            # touches a committed burn (committed_at IS NULL filter).
            self._conn.execute(
                "DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL",
                (reservation_id,),
            )

    def record_holdout_returns(
        self, holdout_evaluation_id: int, strategy_id: int, *,
        holdout_start: str, holdout_end: str,
        returns: list[float], bar_dates: list[str],
    ) -> int:
        """Persist ONE OOS return vector for a committed holdout burn (#221 Slice 1). Separate
        transaction from the burn (the burn committed at on_peek). IDEMPOTENT on identical content:
        if a row already exists with the same (strategy_id, holdout_start, holdout_end, n_bars,
        returns_blob, bar_dates_blob), the existing row id is returned without error — enabling
        the "returns row exists, gate row missing → re-run" reconciliation path. A row that exists
        with DIFFERENT content raises ValueError (genuine conflicting double-write).
        Validation is fail-closed, INSIDE the transaction so check+insert are atomic (TOCTOU)."""
        n_bars = len(returns)
        if n_bars != len(bar_dates):
            raise ValueError(
                f"holdout_returns length mismatch: {n_bars} returns vs {len(bar_dates)} dates")
        if n_bars == 0:
            raise ValueError("holdout_returns must have at least one bar")
        # Encoding is CPU-only — do it OUTSIDE the transaction so we don't hold a lock during it.
        returns_blob = np.asarray(returns, dtype="<f8").tobytes()
        bar_dates_blob = "\n".join(bar_dates).encode("utf-8")
        try:
            with self._conn:
                # Concurrency safety here rests on UNIQUE(holdout_evaluation_id) + the caught
                # IntegrityError below + the idempotency check, NOT on these SELECTs: Python's
                # sqlite3 opens the transaction only on the first DML (the INSERT), so a read-only
                # validation runs in autocommit. That is acceptable — no supported API mutates a
                # committed burn's strategy_id/committed_at, and concurrent inserts of the same
                # holdout_evaluation_id are caught by the UNIQUE constraint. The checks are
                # caller-bug defense (missing / uncommitted / mismatched-strategy burn).
                row = self._conn.execute(
                    "SELECT strategy_id, committed_at FROM holdout_evaluations WHERE id = ?",
                    (holdout_evaluation_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"holdout_evaluation {holdout_evaluation_id} does not exist")
                if row["committed_at"] is None:
                    raise ValueError(
                        f"holdout_evaluation {holdout_evaluation_id} is not a committed burn")
                if int(row["strategy_id"]) != int(strategy_id):
                    raise ValueError(
                        f"strategy_id {strategy_id} does not match holdout_evaluation "
                        f"{holdout_evaluation_id} (strategy_id {row['strategy_id']})")
                existing = self._conn.execute(
                    "SELECT id, strategy_id, holdout_start, holdout_end, n_bars,"
                    " returns_blob, bar_dates_blob"
                    " FROM holdout_returns WHERE holdout_evaluation_id = ?",
                    (holdout_evaluation_id,),
                ).fetchone()
                if existing is not None:
                    if (int(existing["strategy_id"]) == int(strategy_id)
                            and existing["holdout_start"] == holdout_start
                            and existing["holdout_end"] == holdout_end
                            and int(existing["n_bars"]) == n_bars
                            and bytes(existing["returns_blob"]) == returns_blob
                            and bytes(existing["bar_dates_blob"]) == bar_dates_blob):
                        # idempotent reconciliation: identical content already written
                        return int(existing["id"])
                    raise ValueError(
                        f"holdout_returns already written for holdout_evaluation "
                        f"{holdout_evaluation_id} with different content")
                cur = self._conn.execute(
                    "INSERT INTO holdout_returns"
                    "(holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,"
                    " returns_blob, bar_dates_blob, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,
                     returns_blob, bar_dates_blob, _now()),
                )
                rowid = cur.lastrowid
        except sqlite3.IntegrityError as e:
            # Concurrent insert raced between our existence check and INSERT — conflicting write.
            raise ValueError(
                f"holdout_returns already written for holdout_evaluation "
                f"{holdout_evaluation_id} (concurrent insert)") from e
        assert rowid is not None
        return int(rowid)

    def overlapping_holdout_return_streams(
        self, strategy_id: int, holdout_start: str, holdout_end: str, window_days: int
    ) -> list[tuple[list[float], list[str]]]:
        """SIBLING-ONLY cross-strategy read (#221 Slice 1 access control).

        Returns OTHER strategies' OOS return vectors whose holdout interval overlaps
        [holdout_start, holdout_end], burned within the trailing window_days. NEVER
        returns the requesting strategy's own vector. This is the ONLY method that reads
        returns_blob. The caller (promotion.run_gate) is trusted to pass the correct
        strategy_id so self-exclusion holds.

        NOTE: holdout_returns keys by strategy_id (FK to strategies.id) while
        search_trials keys by strategy_name — the asymmetry is intentional: returns
        vectors belong to a specific registered strategy row, whereas search trials are
        recorded before registration.
        """
        # Window filter uses he.created_at (burn time via holdout_evaluations.created_at),
        # NOT hr.created_at (write time of the vector row), so a late-written vector for an
        # out-of-window burn is correctly excluded.
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        rows = self._conn.execute(
            "SELECT hr.n_bars AS n_bars, hr.returns_blob AS rb, hr.bar_dates_blob AS db "
            "FROM holdout_returns hr JOIN holdout_evaluations he"
            " ON hr.holdout_evaluation_id = he.id "
            "WHERE hr.strategy_id != ? AND he.created_at >= ?"
            "  AND hr.holdout_start <= ? AND ? <= hr.holdout_end",
            (strategy_id, cutoff, holdout_end, holdout_start),
        ).fetchall()
        out: list[tuple[list[float], list[str]]] = []
        for r in rows:
            vec = np.frombuffer(r["rb"], dtype="<f8")
            if len(vec) != int(r["n_bars"]):
                raise ValueError(
                    f"corrupt holdout_returns blob: {len(vec)} floats != n_bars {r['n_bars']}")
            dates = r["db"].decode("utf-8").split("\n")
            if len(dates) != int(r["n_bars"]):
                raise ValueError(
                    f"corrupt holdout_returns bar_dates_blob: "
                    f"{len(dates)} dates != n_bars {r['n_bars']}")
            out.append(([float(x) for x in vec], dates))
        return out

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

    def funnel_trial_sharpe_var(self, window_days: int) -> FunnelFloor:
        """Per-strategy pooling FIRST (anti-gaming: one vote per strategy regardless of combo
        count), then MEAN across strategies with at least one search_trials row in the trailing
        ``window_days``. A selected strategy pools ALL its rows (the window selects strategies, it
        does NOT slice rows). A strategy with any NULL/NaN/inf stat row is excluded. Returns
        FunnelFloor(None, ...) when fewer than _MIN_FUNNEL_FLOOR_STRATEGIES finite variances exist
        (fail-open -> Phase-1 behavior). ISO-8601 UTC timestamps sort lexically, so a string `>=`
        on created_at is chronological."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        # SELECT all rows of every strategy that has at least one in-window row. The window
        # filters which STRATEGIES are eligible; pooling then uses every row of each.
        rows = self._conn.execute(
            "SELECT strategy_name AS name, trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name IN"
            " (SELECT DISTINCT strategy_name FROM search_trials WHERE created_at >= ?)",
            (cutoff,),
        ).fetchall()
        by_strategy: dict[str, list] = {}
        for r in rows:
            by_strategy.setdefault(r["name"], []).append(r)
        per_strategy_vars: list[float] = []
        total_rows = 0
        for name_rows in by_strategy.values():
            triples = _validated_triples(name_rows)
            if triples is None:
                continue  # excluded: a NULL/non-finite stat in any of this strategy's rows
            var_s = _pool_trial_sharpe_var(triples)
            if var_s is None or not math.isfinite(var_s):
                continue
            per_strategy_vars.append(var_s)
            total_rows += len(name_rows)
        n_strategies = len(per_strategy_vars)
        if n_strategies < MIN_FUNNEL_FLOOR_STRATEGIES:
            return FunnelFloor(None, n_strategies, total_rows)
        return FunnelFloor(sum(per_strategy_vars) / n_strategies, n_strategies, total_rows)

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
        family_id: int | None = None,
        family_lifetime_effective: int | None = None,
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
                " consumed, created_at, family_id, family_lifetime_effective)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)",
                (strategy_id, int(passed), n_funnel, own_lifetime_combos, windowed_total_combos,
                 funnel_window_days, breadth_provenance, int(pit_ok), int(pit_override),
                 holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,
                 data_source, snapshot_id, period_start, period_end, holdout_frac, actor,
                 decision_json, _now(), family_id, family_lifetime_effective),
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

    def fdr_stream_state(self) -> FdrStreamState | None:
        """Read the global LORD++ FDR stream from ``gate_evaluations WHERE fdr_binding=1``.

        Rows with ``fdr_binding`` NULL or 0 are excluded (legacy-safe). The stream is read in
        ``id`` (insertion) order so the position counter t matches the order in which evaluations
        were serialized. Integrity is validated fail-closed: any binding row with NULL/non-finite
        p-value or alpha-level, ``fdr_rejected`` not in {0, 1}, NULL or non-positive
        ``fdr_test_index``, or non-contiguous indices returns None."""
        rows = self._conn.execute(
            "SELECT fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index"
            " FROM gate_evaluations WHERE fdr_binding=1 ORDER BY id",
        ).fetchall()
        if not rows:
            return FdrStreamState(t=0, discovery_indices=[])
        discovery_indices: list[int] = []
        for pos, r in enumerate(rows, start=1):
            p, alpha, rejected, idx = (
                r["fdr_p_value"], r["fdr_alpha_level"], r["fdr_rejected"], r["fdr_test_index"]
            )
            if p is None or alpha is None or not math.isfinite(p) or not math.isfinite(alpha):
                return None
            if rejected is None or int(rejected) not in (0, 1):
                return None
            if idx is None or int(idx) < 1:
                return None
            if int(idx) != pos:
                return None
            if int(rejected) == 1:
                discovery_indices.append(int(idx))
        return FdrStreamState(t=len(rows), discovery_indices=discovery_indices)

    def record_gate_with_fdr_and_maybe_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        p_value: float | None,
        level_fn: Callable[[int, list[int]], float],
        actor: Actor,
        reason: str | None = None,
    ) -> FdrGateOutcome:
        # TOP-LEVEL ONLY — mirrors reserve_holdout's contract. A manual BEGIN IMMEDIATE inside an
        # already-open transaction would raise "cannot start a transaction within a transaction";
        # catching that instead of pre-checking would leave the caller's surrounding tx open in a
        # rolled-back state. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "record_gate_with_fdr_and_maybe_promote must be called at top level,"
                " not inside an open transaction")

        provisional_passed = bool(gate_row.get("passed"))
        fdr_binding = p_value is not None and math.isfinite(p_value)

        t_next: int | None = None
        alpha_t: float | None = None
        fdr_rejected: bool | None = None
        final_passed: bool

        # BEGIN IMMEDIATE takes the write lock up front so the stream-state SELECT, the gate row
        # INSERT, and the optional stage CAS are one atomic critical section — two concurrent
        # binding evaluations can't both read t=0 and both write fdr_test_index=1.
        # BaseException (not Exception) so KeyboardInterrupt/SystemExit still rolls back.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            if fdr_binding:
                stream = self.fdr_stream_state()
                if stream is None:
                    raise RuntimeError("FDR stream integrity failure — cannot compute alpha_t")
                t_next = stream.t + 1
                alpha_t = level_fn(t_next, stream.discovery_indices)
                assert p_value is not None
                fdr_rejected = p_value <= alpha_t
                final_passed = provisional_passed and fdr_rejected
            else:
                final_passed = provisional_passed

            # Patch decision_json so the stored audit record reflects final_passed and the FDR
            # outcome — both are only known after the stream read inside this transaction.
            raw_decision = json.loads(gate_row.get("decision_json") or "{}")
            raw_decision["passed"] = final_passed
            if fdr_binding:
                raw_decision["fdr_binding"] = True
                raw_decision["fdr_p_value"] = p_value
                raw_decision["fdr_alpha_level"] = alpha_t
                raw_decision["fdr_rejected"] = bool(fdr_rejected)
                raw_decision["fdr_test_index"] = t_next
                raw_decision["checks"] = raw_decision.get("checks", []) + [{
                    "name": "fdr_evidence",
                    "value": p_value,
                    "threshold": alpha_t,
                    "op": "<=",
                    "passed": bool(fdr_rejected),
                }]
            decision_json = json.dumps(raw_decision)

            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " consumed, created_at,"
                " fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index,"
                " family_id, family_lifetime_effective)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                # Agent passing rows are born-consumed: the stage has already advanced inside
                # this transaction, so the token is spent. Leaving consumed=0 would let a
                # future `registry transition --to candidate --actor agent` reuse the old row
                # after a back-step, bypassing the gate re-run requirement.
                (rec.id, int(final_passed),
                 gate_row["n_funnel"], gate_row["own_lifetime_combos"],
                 gate_row["windowed_total_combos"], gate_row["funnel_window_days"],
                 gate_row["breadth_provenance"], int(gate_row["pit_ok"]),
                 int(gate_row.get("pit_override", False)),
                 gate_row["holdout_n_bars"], gate_row["min_holdout_observations"],
                 gate_row["code_hash"], gate_row["config_hash"], gate_row.get("dependency_hash"),
                 gate_row["data_source"], gate_row.get("snapshot_id"),
                 gate_row["period_start"], gate_row["period_end"], gate_row["holdout_frac"],
                 actor.value, decision_json,
                 int(actor is Actor.AGENT and final_passed), _now(),
                 1 if fdr_binding else None,
                 p_value if fdr_binding else None,
                 alpha_t if fdr_binding else None,
                 int(fdr_rejected) if fdr_rejected is not None else None,
                 t_next if fdr_binding else None,
                 gate_row.get("family_id"), gate_row.get("family_lifetime_effective")),
            )
            gate_id = cur.lastrowid
            assert gate_id is not None

            updated_rec: StrategyRecord | None = None
            if final_passed:
                updated_rec = self._apply_transition_locked(
                    rec, Stage.CANDIDATE, actor, reason,
                    code_hash=gate_row["code_hash"], config_hash=gate_row["config_hash"],
                    dependency_hash=gate_row.get("dependency_hash"),
                    consume_gate_id=None, consume_forward_gate_id=None, now=_now())

            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

        return FdrGateOutcome(
            gate_id=gate_id,
            fdr_binding=fdr_binding,
            fdr_test_index=t_next,
            fdr_p_value=p_value if fdr_binding else None,
            fdr_alpha_level=alpha_t,
            fdr_rejected=fdr_rejected,
            final_passed=final_passed,
            updated_rec=updated_rec,
        )

    # -------------------------------------------------------------------------
    # Factor-evaluation ledger (#219, slice E of #140)
    # -------------------------------------------------------------------------

    def record_factor_evaluation(
        self,
        *,
        factor_name: str,
        import_path: str,
        code_hash: str,
        hypothesis_hash: str,
        period_start: str,
        period_end: str,
        horizon: int,
        params_json: str,
        construction: str,
        construction_params_json: str,
        n_obs: int | None,
        mean_ic: float | None,
        ic_ir: float | None,
        t_stat: float | None,
        ic_skew: float | None,
        ic_kurtosis: float | None,
        n_dependents: int,
        data_source: str,
        snapshot_id: str | None,
        actor: str,
        created_at: str,
    ) -> int:
        """Persist one factor evaluation (correction cols NULL until finalize). Returns row id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO factor_evaluations"
                "(factor_name, import_path, code_hash, hypothesis_hash,"
                " period_start, period_end, horizon, params_json, construction,"
                " construction_params_json, n_obs, mean_ic, ic_ir, t_stat,"
                " ic_skew, ic_kurtosis, n_dependents, data_source, snapshot_id,"
                " actor, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    factor_name, import_path, code_hash, hypothesis_hash,
                    period_start, period_end, horizon, params_json, construction,
                    construction_params_json, n_obs, mean_ic, ic_ir, t_stat,
                    ic_skew, ic_kurtosis, n_dependents, data_source, snapshot_id,
                    actor, created_at,
                ),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def factor_hypothesis_breadth(
        self, factor_name: str, window_days: int
    ) -> tuple[int, int]:
        """(own_lifetime, windowed_total) distinct hypothesis_hash counts.

        own_lifetime: all-time DISTINCT hypothesis_hash for this factor.
        windowed_total: DISTINCT hypothesis_hash across ALL factors within the
        trailing window_days (funnel-wide breadth — mirrors windowed_search_combos).
        ISO-8601 UTC timestamps compare lexicographically in chronological order.
        """
        own_row = self._conn.execute(
            "SELECT COUNT(DISTINCT hypothesis_hash) AS cnt"
            " FROM factor_evaluations WHERE factor_name=?",
            (factor_name,),
        ).fetchone()
        own = int(own_row["cnt"])

        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        win_row = self._conn.execute(
            "SELECT COUNT(DISTINCT hypothesis_hash) AS cnt"
            " FROM factor_evaluations WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
        windowed = int(win_row["cnt"])

        return own, windowed

    def windowed_factor_irs(self, window_days: int) -> list[float]:
        """Latest finite ic_ir per distinct hypothesis_hash within the trailing window.

        Deduplicated: for each hypothesis_hash, only the most-recent row's ic_ir is
        considered (a re-run that updates the IR doesn't double-count dispersion).
        Non-finite ic_ir values (NULL, NaN, inf) are excluded — fail-closed.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        rows = self._conn.execute(
            "SELECT ic_ir FROM ("
            "  SELECT ic_ir, ROW_NUMBER() OVER ("
            "    PARTITION BY hypothesis_hash ORDER BY created_at DESC, id DESC"
            "  ) AS rn"
            "  FROM factor_evaluations WHERE created_at >= ?"
            ") WHERE rn = 1 AND ic_ir IS NOT NULL",
            (cutoff,),
        ).fetchall()
        return [
            float(row["ic_ir"])
            for row in rows
            if row["ic_ir"] is not None and math.isfinite(float(row["ic_ir"]))
        ]

    def finalize_factor_evaluation(
        self,
        row_id: int,
        n_hypotheses: int,
        dsr_confidence: float | None,
        significant: bool,
    ) -> None:
        """Write the correction columns to the factor evaluation row."""
        with self._conn:
            self._conn.execute(
                "UPDATE factor_evaluations SET n_hypotheses=?, dsr_confidence=?, significant=?"
                " WHERE id=?",
                (n_hypotheses, dsr_confidence, int(significant), row_id),
            )

    # -------------------------------------------------------------------------
    # Backtest returns persistence (#222, Task 7)
    # -------------------------------------------------------------------------

    def persist_backtest_returns(
        self,
        strategy_name: str,
        period_start: str,
        period_end: str,
        returns: pd.Series,
    ) -> int:
        """Persist a backtest return series as JSON [[date_str, float], ...]. Returns row id."""
        pairs = [
            [idx.isoformat() if hasattr(idx, "isoformat") else str(idx), float(v)]
            for idx, v in returns.items()
        ]
        blob = json.dumps(pairs)
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO backtest_returns"
                " (strategy_name, period_start, period_end, returns_json, created_at)"
                " VALUES (?,?,?,?,?)",
                (strategy_name, period_start, period_end, blob, now),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def load_backtest_returns(self, strategy_name: str) -> pd.Series | None:
        """Load the most recent return series for a strategy, or None."""
        import pandas as pd

        row = self._conn.execute(
            "SELECT returns_json FROM backtest_returns WHERE strategy_name = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (strategy_name,),
        ).fetchone()
        if row is None:
            return None
        pairs = json.loads(row["returns_json"])
        if not pairs:
            return None
        dates, values = zip(*pairs, strict=True)
        return pd.Series(values, index=pd.to_datetime(dates), dtype=float)

    # -------------------------------------------------------------------------
    # Family registry + parentage DAG (#222)
    # -------------------------------------------------------------------------

    def create_family(
        self, name: str, actor: str, created_by_strategy: str | None = None
    ) -> int:
        """Create a new family and record the family_created event. Return the new family id."""
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO families(name, created_at, created_by_actor, created_by_strategy)"
                " VALUES (?,?,?,?)",
                (name, now, actor, created_by_strategy),
            )
            family_id = cur.lastrowid
            assert family_id is not None
            self._conn.execute(
                "INSERT INTO family_events(event_type, family_id, actor, created_at)"
                " VALUES (?,?,?,?)",
                ("family_created", family_id, actor, now),
            )
        return int(family_id)

    def assign_strategy_to_family(
        self,
        strategy_name: str,
        family_id: int,
        actor: str,
        *,
        verdict: str,
        similarity_score: float,
        clustering_version: str,
        clustering_config_json: str,
        axis_json: str,
        matched_family_id: int | None = None,
    ) -> None:
        """Assign a strategy to a family (append-only: old row gets removed_at set)."""
        now = _now()
        event_type = "strategy_merged" if verdict == "MERGE" else "strategy_assigned"
        with self._conn:
            # If an active membership row exists, soft-delete it first.
            self._conn.execute(
                "UPDATE family_members SET removed_at=?"
                " WHERE strategy_name=? AND removed_at IS NULL",
                (now, strategy_name),
            )
            self._conn.execute(
                "INSERT INTO family_members"
                "(family_id, strategy_name, joined_at, joined_by_actor, removed_at)"
                " VALUES (?,?,?,?,NULL)",
                (family_id, strategy_name, now, actor),
            )
            self._conn.execute(
                "INSERT INTO family_events"
                "(event_type, family_id, strategy_name, actor,"
                " clustering_verdict, similarity_score, clustering_version,"
                " clustering_config_json, axis_json, matched_family_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event_type, family_id, strategy_name, actor,
                 verdict, similarity_score, clustering_version,
                 clustering_config_json, axis_json, matched_family_id, now),
            )

    def strategy_family(self, strategy_name: str) -> int | None:
        """Return the current (active) family_id for the strategy, or None."""
        row = self._conn.execute(
            "SELECT family_id FROM family_members WHERE strategy_name=? AND removed_at IS NULL",
            (strategy_name,),
        ).fetchone()
        return int(row["family_id"]) if row is not None else None

    def family_ancestry(self, family_id: int) -> list[int]:
        """BFS-transitive list of all ancestor family_ids (cycle-safe via visited set)."""
        visited: set[int] = {family_id}
        queue: deque[int] = deque([family_id])
        ancestors: list[int] = []
        while queue:
            current = queue.popleft()
            rows = self._conn.execute(
                "SELECT parent_family_id FROM family_parents WHERE child_family_id=?",
                (current,),
            ).fetchall()
            for row in rows:
                pid = int(row[0])
                if pid not in visited:
                    visited.add(pid)
                    ancestors.append(pid)
                    queue.append(pid)
        return ancestors

    def add_parent_edge(self, child_family_id: int, parent_family_id: int) -> None:
        """Atomically add a parent edge (cycle-guarded, BEGIN IMMEDIATE, top-level-only)."""
        if self._conn.in_transaction:
            raise RuntimeError(
                "add_parent_edge must be called at top level, not inside an open transaction"
            )
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # Cycle guard: adding edge (child, parent) creates a cycle iff parent_family_id is
            # already an ancestor of child_family_id, OR parent == child (self-edge).
            # Cycle means: following PARENT edges from parent_family_id upward reaches
            # child_family_id (closing a loop: child -> parent -> ... -> child).
            # Equivalently: child_family_id must not already be an ancestor-or-self of
            # parent_family_id when we add child as parent's ancestor.
            # Correct check: BFS from parent_family_id following PARENT links (going up the
            # ancestry). If child_family_id appears, the new edge would form a cycle.
            if parent_family_id == child_family_id:
                raise ValueError(
                    f"cycle detected: cannot add self-edge {child_family_id} -> {parent_family_id}"
                )
            visited: set[int] = {parent_family_id}
            queue: deque[int] = deque([parent_family_id])
            while queue:
                current = queue.popleft()
                rows = self._conn.execute(
                    "SELECT parent_family_id FROM family_parents WHERE child_family_id=?",
                    (current,),
                ).fetchall()
                for row in rows:
                    pid = int(row[0])
                    if pid not in visited:
                        visited.add(pid)
                        queue.append(pid)
            if child_family_id in visited:
                raise ValueError(
                    f"cycle detected: adding edge {child_family_id} -> {parent_family_id}"
                    f" would create a cycle"
                )
            now = _now()
            self._conn.execute(
                "INSERT INTO family_parents(child_family_id, parent_family_id)"
                " VALUES (?,?)",
                (child_family_id, parent_family_id),
            )
            self._conn.execute(
                "INSERT INTO family_events"
                "(event_type, family_id, actor, created_at)"
                " VALUES (?,?,?,?)",
                ("parent_edge_added", child_family_id, "system", now),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def all_families_with_member_profiles(self) -> list[tuple[int, list[dict]]]:
        """Return [(family_id, members_list)] for all families that have active members.

        Each member dict: {"code_hash": str, "factors": set[str]}.
        The code_hash is looked up via compute_artifact_hashes; strategies whose module
        cannot be loaded silently get code_hash='' and factors=set() (fail-closed: they
        will not match unless the new strategy also fails to load, which is extremely
        unlikely and also fails closed elsewhere).
        """
        from algua.registry.approvals import compute_artifact_hashes
        from algua.registry.lineage import factors_used_by

        rows = self._conn.execute(
            "SELECT DISTINCT family_id, strategy_name"
            " FROM family_members"
            " WHERE removed_at IS NULL"
            " ORDER BY family_id"
        ).fetchall()
        # Group by family_id
        family_map: dict[int, list[dict]] = {}
        for row in rows:
            fid = int(row["family_id"])
            sname = row["strategy_name"]
            try:
                identity = compute_artifact_hashes(sname)
                code_hash = identity.code_hash
            except Exception:  # noqa: BLE001
                code_hash = ""
            try:
                factor_specs = factors_used_by(sname)
                # factors_used_by returns list[FactorSpec]; get the name string
                factors: set[str] = {
                    f.name if hasattr(f, "name") else str(f) for f in factor_specs
                }
            except Exception:  # noqa: BLE001
                factors = set()
            family_map.setdefault(fid, []).append({
                "code_hash": code_hash, "factors": factors, "name": sname,
            })
        return list(family_map.items())

    def _family_member_strategies(self, family_id: int) -> list[str]:
        """DISTINCT strategy names for a family and all its transitive ancestors."""
        ancestor_ids = [family_id] + self.family_ancestry(family_id)
        placeholders = ",".join("?" * len(ancestor_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT fm.strategy_name FROM family_members fm"
            f" WHERE fm.family_id IN ({placeholders})",
            ancestor_ids,
        ).fetchall()
        return [row[0] for row in rows]

    def windowed_family_combos(self, family_id: int, window_days: int) -> int:
        """Windowed search combos for a family + transitive ancestors.

        Like family_lifetime_combos but filtered to search_trials created within
        the trailing window_days. Used for informational output and gate_evaluations
        audit field; NOT used in the 3-way max (which uses lifetime).
        """
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        member_strategies = self._family_member_strategies(family_id)
        if not member_strategies:
            return 0
        placeholders = ",".join("?" * len(member_strategies))
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
            f" WHERE st.created_at >= ? AND st.strategy_name IN ({placeholders})",
            [cutoff, *member_strategies],
        ).fetchone()
        return int(row[0])

    def family_lifetime_combos(self, family_id: int) -> int:
        """Lifetime search combos across this family + all transitive ancestors.

        Uses a DISTINCT subquery so a strategy that was reassigned between two ancestor
        families (leaving a removed_at row in one and an active row in another) is counted
        exactly once — not once per matching family_members row.
        """
        member_strategies = self._family_member_strategies(family_id)
        if not member_strategies:
            return 0
        placeholders = ",".join("?" * len(member_strategies))
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
            f" WHERE st.strategy_name IN ({placeholders})",
            member_strategies,
        ).fetchone()
        return int(row[0])
