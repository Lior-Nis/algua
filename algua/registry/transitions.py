from __future__ import annotations

from collections.abc import Callable

from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry.approvals import compute_artifact_hashes, has_valid_approval
from algua.registry.repository import StrategyRecord, StrategyRepository

ApprovalVerifier = Callable[[StrategyRepository, int, str, str], bool]


def transition_strategy(
    repo: StrategyRepository,
    name: str,
    to: Stage | str,
    actor: Actor | str,
    reason: str | None = None,
    approval_verifier: ApprovalVerifier | None = None,
) -> StrategyRecord:
    target = Stage(to)
    transition_actor = Actor(actor)
    rec = repo.get(name)
    validate_transition(rec.stage, target)
    code_hash: str | None = None
    config_hash: str | None = None
    if target == Stage.LIVE:
        code_hash, config_hash = _validate_live_gate(
            repo=repo,
            name=name,
            strategy_id=rec.id,
            actor=transition_actor,
            approval_verifier=approval_verifier,
        )
    return repo.apply_transition(
        rec=rec,
        to=target,
        actor=transition_actor,
        reason=reason,
        code_hash=code_hash,
        config_hash=config_hash,
    )


def _validate_live_gate(
    *,
    repo: StrategyRepository,
    name: str,
    strategy_id: int,
    actor: Actor,
    approval_verifier: ApprovalVerifier | None,
) -> tuple[str, str]:
    """Enforce the human-only live wall against the *recomputed* artifact identity.

    Returns the hashes actually pinned (recomputed from source), so they land in the transition
    history rather than any caller-supplied value.
    """
    if actor is not Actor.HUMAN:
        raise TransitionError("transition to live requires a human actor")
    code_hash, config_hash = compute_artifact_hashes(name)
    verifier = approval_verifier or has_valid_approval
    if not verifier(repo, strategy_id, code_hash, config_hash):
        raise TransitionError("no matching human approval for this code+config")
    return code_hash, config_hash
