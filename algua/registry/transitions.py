from __future__ import annotations

from collections.abc import Callable

from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry.repository import ArtifactIdentity, StrategyRecord, StrategyRepository

ApprovalVerifier = Callable[[StrategyRepository, int, str, str, str | None], bool]


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
    dependency_hash: str | None = None
    consume_gate_id: int | None = None
    if target == Stage.LIVE:
        identity = _validate_live_gate(
            repo=repo,
            name=name,
            strategy_id=rec.id,
            actor=transition_actor,
            approval_verifier=approval_verifier,
        )
        code_hash, config_hash, dependency_hash = identity
    elif target == Stage.CANDIDATE and transition_actor is not Actor.HUMAN:
        # Wall D: an agent reaches candidate ONLY by consuming a fresh, identity-matched,
        # single-use gate token (minted by `research promote`). Humans are exempt.
        consume_gate_id = _validate_shortlist_gate(repo=repo, name=name, strategy_id=rec.id)
    return repo.apply_transition(
        rec=rec,
        to=target,
        actor=transition_actor,
        reason=reason,
        code_hash=code_hash,
        config_hash=config_hash,
        dependency_hash=dependency_hash,
        consume_gate_id=consume_gate_id,
    )


def _validate_live_gate(
    *,
    repo: StrategyRepository,
    name: str,
    strategy_id: int,
    actor: Actor,
    approval_verifier: ApprovalVerifier | None,
) -> ArtifactIdentity:
    """Enforce the human-only live wall against the *recomputed* artifact identity.

    Returns the identity actually pinned (recomputed from source, config, and the lockfile), so
    it lands in the transition history rather than any caller-supplied value. The gate trusts an
    approval only when code, config, AND dependency hashes all match an unrevoked row.
    """
    if actor is not Actor.HUMAN:
        raise TransitionError("transition to live requires a human actor")
    identity = _compute_hashes(name)
    verifier = approval_verifier or _default_approval_verifier()
    if not verifier(
        repo,
        strategy_id,
        identity.code_hash,
        identity.config_hash,
        identity.dependency_hash,
    ):
        raise TransitionError("no matching human approval for this code+config+dependency")
    return identity


def _validate_shortlist_gate(*, repo: StrategyRepository, name: str, strategy_id: int) -> int:
    """Return the id of a fresh passing AGENT gate token whose recomputed identity matches the
    strategy's current code+config+dependency, or raise. Consumption itself happens inside
    apply_transition, in one transaction with the stage change."""
    identity = _compute_hashes(name)
    gate_id = repo.find_consumable_gate_evaluation(
        strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash)
    if gate_id is None:
        raise TransitionError(
            "transition to candidate requires a fresh passing gate evaluation for the current "
            "code+config+dependency; run `algua research promote` (no matching gate record found)"
        )
    return gate_id


def _compute_hashes(name: str) -> ArtifactIdentity:
    from algua.registry.approvals import compute_artifact_hashes

    return compute_artifact_hashes(name)


def _default_approval_verifier() -> ApprovalVerifier:
    from algua.registry.approvals import has_valid_approval

    return has_valid_approval
