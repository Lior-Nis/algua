from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

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
    consume_forward_gate_id: int | None = None
    if target == Stage.LIVE:
        identity = _validate_live_gate(
            repo=repo,
            name=name,
            strategy_id=rec.id,
            actor=transition_actor,
            approval_verifier=approval_verifier,
        )
        code_hash, config_hash, dependency_hash = identity
    elif (
        rec.stage is Stage.BACKTESTED
        and target == Stage.CANDIDATE
        and transition_actor is not Actor.HUMAN
    ):
        # Wall D, scoped to the FORWARD edge: an agent reaches candidate from below ONLY by
        # consuming a fresh, identity-matched, single-use gate token (minted by `research
        # promote`). Humans are exempt. The PAPER -> CANDIDATE back-step is free for any actor —
        # re-entry to candidate from below always re-runs the research gate.
        consume_gate_id = _validate_shortlist_gate(repo=repo, name=name, strategy_id=rec.id)
    elif rec.stage is Stage.PAPER and target == Stage.FORWARD_TESTED:
        # Identity is pinned for BOTH actors: the agent's token consume re-checks it inside
        # apply_transition, and a human raw transition records it for audit (#124).
        identity = _compute_hashes(name)
        code_hash, config_hash, dependency_hash = identity
        if transition_actor is not Actor.HUMAN:
            consume_forward_gate_id = _validate_forward_gate(
                repo=repo, strategy_id=rec.id, identity=identity)
    return repo.apply_transition(
        rec=rec,
        to=target,
        actor=transition_actor,
        reason=reason,
        code_hash=code_hash,
        config_hash=config_hash,
        dependency_hash=dependency_hash,
        consume_gate_id=consume_gate_id,
        consume_forward_gate_id=consume_forward_gate_id,
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


def _validate_forward_gate(
    *, repo: StrategyRepository, strategy_id: int, identity: ArtifactIdentity
) -> int:
    """Return the id of the newest fresh passing AGENT forward token matching the current
    identity, or raise. Consumption (with the full predicate recheck) happens inside
    apply_transition, in one transaction with the stage change."""
    from algua.research.forward_gates import FORWARD_TOKEN_TTL_DAYS

    gate_id = repo.find_consumable_forward_gate_evaluation(
        strategy_id,
        identity.code_hash,
        identity.config_hash,
        identity.dependency_hash,
        now=datetime.now(UTC).isoformat(),
        ttl_days=FORWARD_TOKEN_TTL_DAYS,
    )
    if gate_id is None:
        raise TransitionError(
            "transition to forward_tested requires a fresh passing forward-gate evaluation for"
            " the current code+config+dependency; run `algua paper promote`")
    return gate_id


def _compute_hashes(name: str) -> ArtifactIdentity:
    from algua.registry.approvals import compute_artifact_hashes

    return compute_artifact_hashes(name)


def _default_approval_verifier() -> ApprovalVerifier:
    from algua.registry.approvals import has_valid_approval

    return has_valid_approval
