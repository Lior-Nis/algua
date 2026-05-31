from __future__ import annotations

import sqlite3
from collections.abc import Callable

from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry import store

ApprovalVerifier = Callable[[sqlite3.Connection, int, str, str], bool]


def transition_strategy(
    conn: sqlite3.Connection,
    name: str,
    to: Stage | str,
    actor: Actor | str,
    reason: str | None = None,
    code_hash: str | None = None,
    config_hash: str | None = None,
    approval_verifier: ApprovalVerifier | None = None,
) -> store.StrategyRecord:
    target = Stage(to)
    transition_actor = Actor(actor)
    rec = store.get_strategy(conn, name)
    validate_transition(rec.stage, target)
    if target == Stage.LIVE:
        _validate_live_gate(
            conn=conn,
            strategy_id=rec.id,
            actor=transition_actor,
            code_hash=code_hash,
            config_hash=config_hash,
            approval_verifier=approval_verifier,
        )
    return store.apply_transition(
        conn=conn,
        rec=rec,
        to=target,
        actor=transition_actor,
        reason=reason,
        code_hash=code_hash,
        config_hash=config_hash,
    )


def _validate_live_gate(
    *,
    conn: sqlite3.Connection,
    strategy_id: int,
    actor: Actor,
    code_hash: str | None,
    config_hash: str | None,
    approval_verifier: ApprovalVerifier | None,
) -> None:
    verifier = approval_verifier or _default_approval_verifier()
    if actor is not Actor.HUMAN:
        raise TransitionError("transition to live requires a human actor")
    if code_hash is None or config_hash is None:
        raise TransitionError("transition to live requires code_hash and config_hash")
    if not code_hash.strip() or not config_hash.strip():
        raise TransitionError("transition to live requires non-empty code_hash and config_hash")
    if not verifier(conn, strategy_id, code_hash, config_hash):
        raise TransitionError("no matching human approval for this code+config")


def _default_approval_verifier() -> ApprovalVerifier:
    from algua.registry.approvals import has_valid_approval

    return has_valid_approval
