"""Forward-test promotion orchestration (#124): guard -> preflight -> run the forward gate.

The protected orchestration layer for the ``paper -> forward_tested`` gate, mirroring
``registry/promotion.py`` for the shortlist gate. Evidence assembly (DB + broker ->
``ForwardEvidence``) lives in ``registry/forward_evidence.py``; live-wall certificate
re-verification lives in ``registry/live_certificate.py``. This module owns the promotion identity
chokepoint (Story 1.3d §5), actor/relaxation guarding, stage-legality preflight, and the
transactional record-and-promote write path around
``algua.research.forward_gates.evaluate_forward_gate`` — it decides, but only using evidence
``forward_evidence.py`` already assembled.

CODEOWNERS-protected: every clause here is a wall against an autonomous agent fabricating
forward evidence (back-dated ticks, identity drift, sibling contamination, manual fills,
external capital). Each helper FAILS CLOSED — ambiguity is never resolved in the
strategy's favor.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.deployment import DeploymentError
from algua.registry.forward_evidence import (
    ActivitiesFetch,
    AssembledEvidence,
    SessionCalendar,
    assemble_forward_evidence,
)
from algua.registry.frozen_runtime import FrozenContentVerifier, recorded_descriptor
from algua.registry.repository import ArtifactIdentity, StrategyRecord, StrategyRepository
from algua.registry.store import DeploymentRecord, SqliteStrategyRepository
from algua.research.forward_gates import (
    ForwardGateCriteria,
    ForwardGateDecision,
    evaluate_forward_gate,
)


def guard_forward_relaxations(actor: Actor, criteria: ForwardGateCriteria) -> None:
    """Each threshold has a strict direction; an agent may only move it stricter (#124).
    Mirrors ``guard_agent_relaxations``: the agent only ever sees the strict gate."""
    # forward_sharpe_confidence feeds NormalDist.inv_cdf (blows up at the 0/1 edges) and the
    # tighten-only comparison below (nan < 0.95 is False, so a non-finite value would slip through
    # the relaxation guard un-flagged). A confidence outside the open unit interval is nonsensical
    # for ANY actor — fail closed here at the boundary rather than pass it silently downstream.
    conf = criteria.forward_sharpe_confidence
    if not (math.isfinite(conf) and 0.0 < conf < 1.0):
        raise ValueError(
            "forward_sharpe_confidence must be a finite probability in (0, 1), got "
            f"{conf!r}")
    if actor is Actor.HUMAN:
        return
    defaults = ForwardGateCriteria()
    higher_is_stricter = ("min_forward_observations", "min_session_coverage",
                          "degradation_factor", "sharpe_floor", "min_forward_vol",
                          "forward_sharpe_confidence")
    lower_is_stricter = ("max_forward_drawdown", "max_staleness_sessions")
    relaxed = [f for f in higher_is_stricter if getattr(criteria, f) < getattr(defaults, f)]
    relaxed += [f for f in lower_is_stricter if getattr(criteria, f) > getattr(defaults, f)]
    if relaxed:
        raise ValueError(
            "forward-gate relaxation requires --actor human: " + ", ".join(sorted(relaxed)))


PromotionIdentity = tuple[DeploymentRecord, ArtifactIdentity]


def frozen_promotion_identity(
    conn: sqlite3.Connection, rec: StrategyRecord, *, data_dir: Path,
) -> PromotionIdentity | None:
    """The frozen branch of :func:`promotion_identity`; ``None`` for any other strategy.

    Reads the deployment ledger first, so ``paper promote`` calls it in the slot Story 1.3c's
    refusal held: a working-tree or legacy strategy passes untouched, with no hashing. A FROZEN
    deployment's identity is its recorded descriptor, never the checkout: the descriptor is parsed
    and its config strictly decoded (as the tick path does), then its bundle and environment are
    verified offline by a FRESH verifier (no verdict cached from another promotion). Any failure
    refuses with ``frozen_content_unavailable``/``frozen_content_unsupported`` before any
    evaluation row, look count, token or stage change."""
    deployment = SqliteStrategyRepository(conn).active_deployment(rec.id)
    if deployment is None or deployment.source_kind != "frozen":
        return None
    manifest, _config = recorded_descriptor(deployment, rec.name)
    verifier = FrozenContentVerifier(data_dir)
    verifier.bundle(manifest.bundle, deployment_id=deployment.id)
    verifier.environment(manifest.environment, deployment_id=deployment.id)
    return deployment, ArtifactIdentity(
        manifest.code_hash, manifest.config_hash, manifest.dependency_hash)


def promotion_identity(
    conn: sqlite3.Connection, rec: StrategyRecord, *, data_dir: Path,
) -> PromotionIdentity:
    """The ONE identity chokepoint of forward promotion (Story 1.3d §5): the active deployment
    epoch and the identity the evidence, the evaluation row and the promotion are judged against.
    Frozen: :func:`frozen_promotion_identity`. Working-tree: exactly the pre-1.3d checkout hash
    and deployment-hash match; a legacy strategy has no epoch to evaluate."""
    frozen = frozen_promotion_identity(conn, rec, data_dir=data_dir)
    if frozen is not None:
        return frozen
    identity = compute_artifact_hashes(rec.name)
    deployment = SqliteStrategyRepository(conn).require_tick_deployment(rec.id)
    if deployment is None:
        raise DeploymentError("forward promotion requires one active deployment epoch")
    if (
        deployment.code_hash != identity.code_hash
        or deployment.config_hash != identity.config_hash
        or deployment.dependency_hash != identity.dependency_hash
    ):
        raise DeploymentError("active deployment identity does not match the working tree")
    return deployment, identity


def forward_promotion_preflight(
    repo: StrategyRepository, name: str, *, actor: Actor, criteria: ForwardGateCriteria,
) -> StrategyRecord:
    """Pre-work refusals (mirrors ``promotion_preflight``): actor legality, relaxation guard,
    stage legality. FORWARD_TESTED is legal because a re-evaluation refreshes the live-wall
    certificate without a stage change (#124)."""
    # SYSTEM would pass as "not human" (strict) yet mint a row it can never consume.
    if actor not in (Actor.AGENT, Actor.HUMAN):
        raise ValueError(f"paper promote requires --actor agent or human, got {actor.value}")
    guard_forward_relaxations(actor, criteria)
    rec = repo.get(name)
    if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
        raise TransitionError(
            f"paper promote requires stage paper or forward_tested, got {rec.stage.value}")
    if rec.stage is Stage.PAPER:
        validate_transition(rec.stage, Stage.FORWARD_TESTED)
    return rec


@dataclass
class ForwardPromotionOutcome:
    decision: ForwardGateDecision
    promoted: bool
    assembled: AssembledEvidence


def run_forward_gate(
    repo: StrategyRepository,
    conn: sqlite3.Connection,
    *,
    name: str,
    actor: Actor,
    criteria: ForwardGateCriteria,
    calendar: SessionCalendar,
    now: datetime,
    activities_fetch: ActivitiesFetch,
    deployment: DeploymentRecord,
    identity: ArtifactIdentity,
) -> ForwardPromotionOutcome:
    """Assemble evidence -> evaluate -> record (pass AND fail) -> on pass from PAPER record AND
    promote in one transaction. At FORWARD_TESTED a passing run is the certificate refresh: a
    new row, no stage change.

    ``(deployment, identity)`` come from :func:`promotion_identity`, resolved ONCE per promotion
    (a frozen deployment's content is not re-verified here), and feed the evidence admissibility
    filter, the evaluation row AND the transition's pinned hashes — they can never disagree. The
    cheap re-check below binds them to this strategy's deployment row, so a direct caller can never
    pair a frozen deployment with a checkout identity; the epoch must still be active."""
    rec = repo.get(name)
    if deployment.strategy_id != rec.id or tuple(identity) != (
        deployment.code_hash, deployment.config_hash, deployment.dependency_hash
    ):
        raise DeploymentError("promotion identity does not match the strategy's deployment")
    deployment_id = deployment.id
    asm = assemble_forward_evidence(
        conn, strategy_id=rec.id, name=name, deployment_id=deployment_id,
        identity=identity, calendar=calendar, now=now,
        activities_fetch=activities_fetch)
    decision = evaluate_forward_gate(asm.evidence, criteria)
    gate_row: dict[str, Any] = {
        "passed": decision.passed,
        "n_forward_observations": asm.evidence.n_return_observations,
        "min_forward_observations": criteria.min_forward_observations,
        "session_coverage": asm.evidence.session_coverage,
        "realized_sharpe": asm.evidence.realized_sharpe,
        "holdout_sharpe": asm.evidence.holdout_sharpe,
        "degradation_factor": criteria.degradation_factor,
        "sharpe_floor": criteria.sharpe_floor,
        "realized_vol": asm.evidence.realized_vol,
        "min_forward_vol": criteria.min_forward_vol,
        "realized_max_drawdown": asm.evidence.realized_max_drawdown,
        "max_forward_drawdown": criteria.max_forward_drawdown,
        "first_tick_id": asm.first_tick_id, "last_tick_id": asm.last_tick_id,
        "first_tick_ts": asm.first_tick_ts, "last_tick_ts": asm.last_tick_ts,
        "max_staleness_sessions": criteria.max_staleness_sessions,
        "n_reconcile_failures": asm.evidence.n_reconcile_failures,
        "n_concurrent_forward": asm.n_concurrent_forward,
        "account_id": asm.account_id,
        "code_hash": identity.code_hash, "config_hash": identity.config_hash,
        "dependency_hash": identity.dependency_hash,
        "deployment_id": deployment_id,
        "decision_json": json.dumps(decision.to_dict(), sort_keys=True),
    }
    promoted = False
    if decision.passed and rec.stage is Stage.PAPER:
        # "Record passing row + stage CAS + transition row" is ONE sqlite transaction (#124
        # GATE-2): the old record-then-transition shape committed a consumable token first, so
        # a raced/failed transition banked a consumed=0 pass an agent could spend within the
        # TTL after a later demotion — re-entry without a fresh gate run. Going through the
        # repository instead of ``transitions.transition_strategy`` drops exactly two policy
        # steps, both deliberately: ``validate_transition`` (paper -> forward_tested is a
        # statically legal edge by construction here — preflight checked it — and the CAS's
        # from_stage=paper predicate enforces the stage atomically) and the consumable-token
        # lookup (we promote on the very evidence row we insert, in the same transaction —
        # there is no token to find or consume; the row is born spent). The standalone token
        # path in ``transitions`` remains for tokens minted by earlier runs.
        repo.record_forward_pass_and_promote(
            rec, gate_row=gate_row, actor=actor, reason=_forward_gate_reason(decision))
        promoted = True
    else:
        repo.record_forward_gate_evaluation(
            rec.id, **gate_row, actor=actor.value,
            # A refresh at forward_tested must refresh the live certificate WITHOUT minting a
            # re-entry token (#124 GATE-2): only a run FROM paper writes a consumable row, so a
            # demote-then-re-promote can never bank a refresh — it always re-runs the full gate.
            consumable=rec.stage is Stage.PAPER)
    return ForwardPromotionOutcome(decision=decision, promoted=promoted, assembled=asm)


def _forward_gate_reason(decision: ForwardGateDecision) -> str:
    """Human-readable gate summary (mirrors ``promotion._gate_reason``). Metric checks render
    value/op/threshold; boolean checks render name=pass|fail."""
    parts: list[str] = []
    for c in decision.checks:
        if "value" in c and c.get("value") is not None and c.get("threshold") is not None:
            parts.append(f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}")
        else:
            parts.append(f"{c['name']}={'pass' if c['passed'] else 'fail'}")
    return "forward gate pass: " + ", ".join(parts)
