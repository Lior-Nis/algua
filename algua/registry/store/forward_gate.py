"""``ForwardGateLedger`` — forward-test gate evaluations, the atomic paper -> forward_tested
promotion, and the live-wall certificate selection (#124)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from algua.contracts.lifecycle import Actor, Stage
from algua.registry.repository import StrategyRecord
from algua.registry.store._util import _now
from algua.registry.store.base import TransitionMixin


class ForwardGateMixin(TransitionMixin):
    _conn: sqlite3.Connection

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
