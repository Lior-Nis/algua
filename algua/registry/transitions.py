from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.contracts.types import ExitLaneGuard, PendingLiveAuthorization
from algua.registry.repository import ArtifactIdentity, StrategyRecord, StrategyRepository

# (repo, strategy_id, code_hash, config_hash, dependency_hash) -> approval result. Truthy =
# approved. The signed-challenge (CLI) path returns a ``PendingLiveAuthorization`` to be recorded
# ATOMICALLY with the stage CAS (#254); the legacy approvals-row path (``has_valid_approval``)
# returns a bool (True = approved with nothing to persist). Falsy (False/None) = denied.
ApprovalVerifier = Callable[
    [StrategyRepository, int, str, str, str | None], "PendingLiveAuthorization | bool"]
# (repo, name, strategy_id, identity) -> certificate summary; raises TransitionError on any
# refusal. ``identity`` is the live gate's ONE recomputed identity — the verifier judges against
# it instead of recomputing, so the certificate and approval checks can never drift (#124 GATE-2).
ForwardCertificateVerifier = Callable[
    [StrategyRepository, str, int, ArtifactIdentity], dict[str, Any]]

# Every book-exit / lane-crossing edge that must SHED the strategy's capital reservation as part of
# the transition (#497). Leaving a strategy allocated after it leaves its operating book orphans the
# reservation: run-all only iterates the source lane, so nothing would ever wind the position down
# or free the capital. The revoke rides ATOMICALLY with the stage CAS (and, on forward_tested->live,
# the go-live authorization) inside apply_transition, mirroring the original live->dormant wind-down
# (#125/#247). Edges kept OUT of the set deliberately retain the allocation: paper->forward_tested
# and forward_tested->paper stay WITHIN the paper book, so the tenant keeps its slice.
_REVOKE_ON_EXIT: frozenset[tuple[Stage, Stage]] = frozenset({
    (Stage.PAPER, Stage.DORMANT), (Stage.PAPER, Stage.RETIRED), (Stage.PAPER, Stage.CANDIDATE),
    (Stage.FORWARD_TESTED, Stage.RETIRED), (Stage.FORWARD_TESTED, Stage.LIVE),
    (Stage.LIVE, Stage.PAPER), (Stage.LIVE, Stage.DORMANT), (Stage.LIVE, Stage.RETIRED)})


def transition_strategy(
    repo: StrategyRepository,
    name: str,
    to: Stage | str,
    actor: Actor | str,
    reason: str | None = None,
    approval_verifier: ApprovalVerifier | None = None,
    forward_certificate_verifier: ForwardCertificateVerifier | None = None,
    exit_guard: ExitLaneGuard | None = None,
) -> StrategyRecord:
    target = Stage(to)
    transition_actor = Actor(actor)
    rec = repo.get(name)
    validate_transition(rec.stage, target)
    if target is Stage.DORMANT and not (reason and reason.strip()):
        raise TransitionError("transition to dormant requires a non-empty reason")
    code_hash: str | None = None
    config_hash: str | None = None
    dependency_hash: str | None = None
    consume_gate_id: int | None = None
    consume_forward_gate_id: int | None = None
    # One membership test drives the wind-down for EVERY book-exit / lane-crossing edge (#497),
    # replacing the single hand-wired live->dormant branch. forward_tested->live is IN the set, so
    # go-live carries BOTH live_authorization (target==LIVE branch below) AND revoke_allocation.
    revoke_allocation = (rec.stage, target) in _REVOKE_ON_EXIT
    live_authorization: PendingLiveAuthorization | None = None
    if target == Stage.LIVE:
        identity, live_authorization = _validate_live_gate(
            repo=repo,
            name=name,
            strategy_id=rec.id,
            actor=transition_actor,
            approval_verifier=approval_verifier,
            forward_certificate_verifier=forward_certificate_verifier,
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
        revoke_allocation=revoke_allocation,
        live_authorization=live_authorization,
        # The source-lane open-order drain only applies to a book-exit edge that sheds the
        # allocation (#497 F2/H1); forwarding it on any other edge would trip the store-layer
        # "exit_guard is only valid on a revoke_allocation transition" guard. The CLI only builds a
        # guard for a live-source revoke edge, so in practice this is a belt-and-suspenders filter.
        exit_guard=exit_guard if revoke_allocation else None,
    )


def _validate_live_gate(
    *,
    repo: StrategyRepository,
    name: str,
    strategy_id: int,
    actor: Actor,
    approval_verifier: ApprovalVerifier | None,
    forward_certificate_verifier: ForwardCertificateVerifier | None,
) -> tuple[ArtifactIdentity, PendingLiveAuthorization | None]:
    """Enforce the human-only live wall against the *recomputed* artifact identity.

    Wall ordering (#124): actor -> forward certificate -> approval. The certificate is the
    EVIDENCE precondition in front of the signature — a fresh, identity-matched, strategy-bound
    passing forward-gate evaluation with a clean record since. Not waivable in-band: there is
    deliberately no flag.

    Returns the identity actually pinned (recomputed from source, config, and the lockfile), so
    it lands in the transition history rather than any caller-supplied value. The gate trusts an
    approval only when code, config, AND dependency hashes all match an unrevoked row.
    """
    if actor is not Actor.HUMAN:
        raise TransitionError("transition to live requires a human actor")
    # ONE identity computation feeds both walls below: a per-wall recompute would open a drift
    # window between the certificate's identity and the approval's (#124 GATE-2).
    identity = _compute_hashes(name)
    (forward_certificate_verifier or _default_forward_certificate_verifier())(
        repo, name, strategy_id, identity)
    verifier = approval_verifier or _default_approval_verifier()
    result = verifier(
        repo,
        strategy_id,
        identity.code_hash,
        identity.config_hash,
        identity.dependency_hash,
    )
    if not result:  # False / None => denied
        raise TransitionError("no matching human approval for this code+config+dependency")
    # The signed-challenge path returns the authorization to persist atomically with the stage CAS;
    # the legacy approvals-row path returns True (approved, nothing to persist) (#254).
    pending = result if isinstance(result, PendingLiveAuthorization) else None
    return identity, pending


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


def _default_forward_certificate_verifier() -> ForwardCertificateVerifier:
    """Real wiring for the live wall's certificate check, built lazily (mirrors
    ``_default_approval_verifier``). The CLI's challenge-issuance path uses this same builder
    (module-attribute access — also the single monkeypatch seam for tests). Every missing
    dependency FAILS CLOSED with an actionable ``TransitionError``."""

    def verify(
        repo: StrategyRepository, name: str, strategy_id: int, identity: ArtifactIdentity,
    ) -> dict[str, Any]:
        from algua.calendar.market_calendar import MarketCalendar
        from algua.config.settings import get_settings
        from algua.execution.alpaca_broker import AlpacaPaperBroker
        from algua.registry.forward_promotion import verify_forward_certificate

        # The Protocol stays I/O-agnostic; only the sqlite store exposes `connection`.
        conn = getattr(repo, "connection", None)
        if conn is None:
            raise TransitionError(
                "forward-certificate verification needs a sqlite-backed repository or an "
                "injected verifier")
        settings = get_settings()
        if not settings.alpaca_api_key or not settings.alpaca_api_secret:
            raise TransitionError(
                "go-live re-verifies account hygiene since certification and needs Alpaca "
                "paper credentials; set ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET")
        broker = AlpacaPaperBroker(api_key=settings.alpaca_api_key,
                                   api_secret=settings.alpaca_api_secret,
                                   base_url=settings.alpaca_paper_url)
        return verify_forward_certificate(
            repo, conn, name=name, strategy_id=strategy_id, identity=identity,
            calendar=MarketCalendar(), now=datetime.now(UTC),
            activities_fetch=broker.account_activities_window,
            # Account continuity: the certificate's account_id must equal the account these
            # credentials resolve to NOW, or the since-certification hygiene re-check would
            # silently inspect the wrong account (#124).
            account_id_fetch=lambda: broker.account().account_id)

    return verify
