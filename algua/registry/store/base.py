"""Shared base mixin for algua/registry/store/ — holds the one private helper genuinely called
from more than one Protocol's domain (crud.py's apply_transition + intake_candidate_to_paper,
gate.py's record_gate_with_fdr_and_maybe_promote, forward_gate.py's
record_forward_pass_and_promote). Each of those mixins inherits this one, so the real
implementation here is the one that runs at composition time via the facade's MRO."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.types import PendingLiveAuthorization
from algua.registry.repository import StrategyRecord


class TransitionMixin:
    _conn: sqlite3.Connection

    if TYPE_CHECKING:
        # Provided by crud.py's CrudMixin (StrategyReader). Declared TYPE_CHECKING-only (unlike
        # algua/data/store/bars.py, whose stubs point at the FACADE and so can never lose the
        # MRO) because these stubs point at SIBLING mixins: a runtime stub would shadow the real
        # implementation whenever the providing mixin sits later in the facade's base list.
        def get(self, name: str) -> StrategyRecord: ...

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
        live_authorization: PendingLiveAuthorization | None = None,
    ) -> StrategyRecord:
        """``apply_transition``'s body, WITHOUT opening a transaction: the caller owns the
        ``with self._conn:`` scope, so a composite write (e.g.
        ``record_forward_pass_and_promote``) can put extra statements in the SAME transaction
        as the token consume + stage CAS + transition INSERT."""
        from_stage = rec.stage
        if live_authorization is not None:
            # Go-live signature path (#254): consume the challenge and write the live_authorizations
            # row in THIS transaction with the stage CAS below, so a raced/failed transition rolls
            # back BOTH — never burning the nonce or leaving an orphan authorization. The signature
            # was already verified (no DB writes) in live_gate.verify_pending. We deliberately do
            # NOT store the challenge text — trade-time verification rebuilds the signed payload
            # from the recomputed identity, never from agent-writable bytes (codex CRITICAL).
            # The consume re-asserts the FULL pending-challenge predicate (strategy + recomputed
            # identity + unexpired + unconsumed) at consume time — mirroring the forward-gate
            # consume — so a signature verified just before expiry, or against a drifted identity,
            # cannot be applied here (closing the validate-then-consume gap; codex #254 review).
            cur = self._conn.execute(
                "UPDATE live_challenges SET consumed_at=?"
                " WHERE nonce=? AND consumed_at IS NULL AND strategy_id=?"
                " AND code_hash=? AND config_hash=? AND dependency_hash IS ? AND expires_at > ?",
                (now, live_authorization.nonce, rec.id, code_hash, config_hash, dependency_hash,
                 now),
            )
            if cur.rowcount != 1:
                raise TransitionError(
                    "go-live challenge is not consumable for this strategy+identity (already "
                    "consumed, missing, identity-drifted, or expired); request a fresh challenge "
                    "and re-sign")
            self._conn.execute(
                "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash,"
                " dependency_hash, nonce, expires_at, signature, principal, authorized_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, code_hash, config_hash, dependency_hash, live_authorization.nonce,
                 live_authorization.expires_at, live_authorization.signature_b64,
                 live_authorization.principal, now),
            )
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
