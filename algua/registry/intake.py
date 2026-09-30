"""`paper intake` (#317): the per-strategy capital slice, the deterministic FIFO ordering of
candidates, and the admission loop that offers them to the atomic primitive.

The admission DECISION is NOT made here — it is made transactionally, one candidate at a time, by
``StrategyRepository.intake_candidate_to_paper`` (which re-checks the count cap and the Σ≤equity
capital bound UNDER the write lock). This module computes the fixed slice and the stable order in
which candidates are offered to that primitive, then drives ``run_intake``'s loop over them one at
a time. Every new admission is FROZEN (Story 1.3c): before the primitive, each candidate is
prepared (Story 1.3b) and its recorded descriptor verified offline, outside any write transaction.

``run_intake`` DOES pre-check the same bounds (``slc <= 0.0 or count >= max_concurrent``) before
offering each candidate, against a locally-incremented count. That is a fail-SAFE filter, not a
second authority: it can only stop early, never admit something the in-transaction re-check would
refuse. The authoritative decision stays under the write lock, so a drift between the two can cost
an admission that would have succeeded — never one that should have failed.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

from algua.audit.log import append as audit_append
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import allocations, artifact_errors
from algua.registry.allocations import (
    AllocationError,
    CountCapReached,
    active_allocation,
    active_paper_lane_count,
)
from algua.registry.artifact_preparation import prepare_frozen_artifact
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_verification import verify_frozen_artifact
from algua.registry.deployment import DeploymentManifest
from algua.registry.store import SqliteStrategyRepository

# Story 1.3b's typed preparation/verification refusals and their stable codes. The registry must
# not import the CLI error registry; tests pin every code equal to its ``error_code``. Anything
# else raised while preparing (SQLite faults, interrupts, untyped bugs) propagates.
_REFUSAL_CODES: tuple[tuple[type[Exception], str], ...] = (
    (artifact_errors.FrozenEnvironmentUnavailable, 'frozen_environment_unavailable'),
    (artifact_errors.FrozenSourceInvalid, 'frozen_source_invalid'),
    (artifact_errors.FrozenSourceDrift, 'frozen_source_drift'),
    (artifact_errors.FrozenAssetsUnsupported, 'frozen_assets_unsupported'),
    (artifact_errors.FrozenBundleCorrupt, 'frozen_bundle_corrupt'),
    (artifact_errors.FrozenEnvironmentIncompatible, 'frozen_environment_incompatible'),
    (artifact_errors.FrozenEnvironmentCorrupt, 'frozen_environment_corrupt'),
    (artifact_errors.FrozenDescriptorConflict, 'frozen_descriptor_conflict'),
    (artifact_errors.ArtifactNotFound, 'artifact_not_found'),
)
_REFUSALS = tuple(typ for typ, _ in _REFUSAL_CODES)
# The only retryable 1.3b refusal: admission stops and the rest stay queued for the next intake.
_RETRYABLE_REFUSAL = 'frozen_environment_unavailable'


@dataclass(frozen=True)
class Candidate:
    """A candidate strategy awaiting paper-intake admission.

    ``entry_id`` is the monotonic ``stage_transitions.id`` of the row that moved the strategy into
    its CURRENT ``candidate`` episode — the FIFO ordering key. A DB autoincrement id is a true,
    gap-free, clock-independent insertion order, unlike a wall-clock ``created_at`` string (which
    can collide at sub-second resolution or move backwards under a clock adjustment). ``sid`` is the
    strategy id, a deterministic tie-break when two candidates somehow share an ``entry_id``.
    """

    name: str
    entry_id: int
    sid: int


@dataclass(frozen=True)
class FrozenAdmission:
    """The exact recorded, offline-verified frozen descriptor one admission binds to."""

    manifest: DeploymentManifest
    research_gate_id: int


# ``(repo, name, *, repo_root, store_root) -> FrozenAdmission``; injectable so tests need no uv/Git.
PrepareAndVerify = Callable[..., FrozenAdmission]


def prepare_and_verify_frozen(
    repo: SqliteStrategyRepository, name: str, *, repo_root: Path, store_root: Path,
) -> FrozenAdmission:
    """Story 1.3b preparation for the current clean ``HEAD`` (qualify, export, publish, record),
    then offline verification of THAT recorded descriptor. The verifier must confirm exactly the
    strategy, artifact row and descriptor preparation recorded; the admission binds that row."""
    prepared = prepare_frozen_artifact(repo, name, repo_root=repo_root, store_root=store_root)
    verified = verify_frozen_artifact(repo, prepared.manifest.digest, store_root=store_root)
    if (verified.strategy, verified.artifact_id, verified.manifest) != (
            name, prepared.artifact_id, prepared.manifest):
        raise artifact_errors.FrozenDescriptorConflict()
    return FrozenAdmission(
        frozen_deployment_manifest(verified.manifest), prepared.research_gate_id)


def _to_cents(dollars: float) -> int:
    """Whole integer cents in ``dollars``, FLOORED (never rounded up).

    Uses ``Decimal`` so the floor is taken on the exact decimal value the operator means (e.g.
    ``0.019`` → ``1`` cent, not ``2``) rather than on a binary-float artifact — ``round(0.019*100)``
    would give ``2`` cents (``$0.02``), OVER-counting a sub-cent equity and violating this
    function's own never-rounds-up contract (and letting ``slice_capital`` return a slice larger
    than ``equity``). ``Decimal(str(x))`` reads the shortest decimal repr, so ``0.29`` floors to
    ``29`` cents, not ``28`` from ``0.29*100 == 28.9999…``.
    """
    return int((Decimal(str(dollars)) * 100).to_integral_value(rounding=ROUND_FLOOR))


def slice_capital(equity: float, max_concurrent: int) -> float:
    """Per-strategy capital slice in dollars, floored to whole cents.

    The floor (never rounding up) is computed in INTEGER CENTS — ``floor(equity_cents) //
    max_concurrent`` — so it is exact at the cent boundary and free of the binary-float rounding
    that could let ``k`` slices sum to a hair OVER ``equity`` and mis-admit the ``k``-th. Flooring
    guarantees ``k`` slices sum to ``<= equity`` for any ``k <= max_concurrent``. If ``equity <=
    0`` the result is ``<= 0`` (nothing is admissible).
    """
    if max_concurrent <= 0:
        raise ValueError('max_concurrent must be positive')
    slice_cents = _to_cents(equity) // max_concurrent  # floor division (toward -inf if equity < 0)
    return slice_cents / 100


def order_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Candidates in deterministic FIFO admission order: ascending ``entry_id`` (older candidate
    episode first), tie-broken by ascending strategy ``sid`` for a total, stable order."""
    return sorted(candidates, key=lambda c: (c.entry_id, c.sid))


def _stage_entry_id(repo: SqliteStrategyRepository, name: str, stage: Stage) -> int:
    """The monotonic ``stage_transitions.id`` of the row that most recently moved this strategy
    into ``stage`` — the FIFO ordering key (#317, finding #5). ``list_transitions`` returns rows
    ordered by ``id``, so the last matching row is the current episode. A DB autoincrement id is a
    true clock-independent insertion order (see intake.Candidate); defensively falls back to ``0``
    if — impossibly — no such transition is recorded."""
    entered = [t['id'] for t in repo.list_transitions(name) if t['to_stage'] == stage.value]
    return entered[-1] if entered else 0


def _candidate_entry_id(repo: SqliteStrategyRepository, name: str) -> int:
    """The FIFO key for a candidate awaiting admission."""
    return _stage_entry_id(repo, name, Stage.CANDIDATE)


# The book stages a strategy can hold a paper allocation at, mirroring
# ``allocations.active_paper_lane_count``'s tenancy definition and ``paper allocate``'s lane scope.
_BOOK_STAGES = (Stage.PAPER, Stage.FORWARD_TESTED)


def _unallocated_book_tenants(
    conn: sqlite3.Connection, repo: SqliteStrategyRepository,
) -> list[Candidate]:
    """Book-stage strategies holding NO active allocation, in FIFO order by book entry.

    This state is a broken invariant, not a queue: `paper -> dormant` and `live -> paper` REVOKE
    the allocation atomically, but the return edges (`dormant -> paper`, `live -> paper`) restore
    only the STAGE. The strategy then sits at a book stage with no slice — `paper run-all` skips it
    as unallocated, it never ticks, and `fleet health` alerts on it forever. Re-admitting it is the
    intake's job because the intake is what owns capital budgeting, the count cap and the FIFO."""
    return order_candidates(
        Candidate(name=r.name, entry_id=_stage_entry_id(repo, r.name, stage), sid=r.id)
        for stage in _BOOK_STAGES
        for r in repo.list_strategies(stage)
        if active_allocation(conn, r.id) is None)


def run_intake(
    conn: sqlite3.Connection, *, equity: float, max_concurrent: int, actor: Actor,
    prepare_and_verify: PrepareAndVerify | None = None,
    repo_root: Path | None = None, store_root: Path | None = None,
) -> dict:
    """The FIFO book-admission loop over ONE registry connection — shared by the ``paper intake``
    command and the ``paper merge-back`` driver (#485) so there is exactly one admit path, never a
    dual one.

    Two populations, re-entrants first:

    **RE-ADMISSION** (``readmitted``) restores the book slice of a strategy already AT a book stage
    that holds no active allocation. That state is a broken invariant, not a queue: `paper ->
    dormant` and `live -> paper` revoke the allocation atomically while the return edges restore
    only the stage, leaving a strategy that `paper run-all` skips as unallocated, that never ticks,
    and that `fleet health` alerts on forever. Re-entrants go FIRST — they were admitted before the
    queued candidates existed — and are funded through ``allocations.allocate_in_lane``, which
    re-reads the stage under the SAME write lock and enforces the SAME Σ ≤ equity and count-cap
    bounds. No stage changes, and an ALREADY-allocated tenant is never touched (intake is not a
    rebalancer).

    **ADMISSION** (``admitted``) takes each candidate in FIFO order (candidate-entry
    ``stage_transitions.id``, tie-break strategy id). Outside any write transaction it runs
    ``prepare_and_verify`` (default :func:`prepare_and_verify_frozen` over this checkout and
    ``Settings.data_dir``), then offers that exact frozen descriptor to the ATOMIC
    ``intake_candidate_to_paper`` primitive, which under ONE write lock re-checks the count cap,
    allocates an equal slice = floor(equity / max_concurrent to cents) (Σ allocations + slice ≤
    ``equity``), opens the deployment epoch on the byte-verified descriptor row and CASes
    candidate→paper — commit-or-rollback together. A 1.3b refusal leaves the candidate untouched and
    is reported in ``refused`` as ``{strategy, code}``: ``frozen_environment_unavailable``
    (retryable) stops admission and queues the rest; any other refusal passes over it. On either
    hard bound (book full / no capital headroom) the remaining candidates are left queued; a
    candidate raced out of ``candidate`` before the CAS is reported in ``skipped_stale``.

    The caller is responsible for reading ``equity`` READ-ONLY from the broker BEFORE opening
    ``conn`` (no trading) and for validating ``max_concurrent`` > 0."""
    repo = SqliteStrategyRepository(conn)
    occupied = active_paper_lane_count(conn)
    slc = slice_capital(equity, max_concurrent)
    admitted: list[dict] = []
    readmitted: list[dict] = []
    queued: list[str] = []
    skipped_stale: list[str] = []
    refused: list[dict] = []
    count = occupied
    step = prepare_and_verify or prepare_and_verify_frozen
    root = repo_root or Path(__file__).resolve().parents[2]

    # ---- RE-ADMISSION first (see docstring): a newcomer must not take the slot out from under a
    # tenant that was already admitted once; no stage changes, the SAME bounds under the lock.
    for tenant in _unallocated_book_tenants(conn, repo):
        if slc <= 0.0 or count >= max_concurrent:
            queued.append(tenant.name)
            continue
        try:
            allocations.allocate_in_lane(
                conn, tenant.sid, capital=slc, actor=actor.value, account_equity=equity,
                allowed_stages=frozenset(s.value for s in _BOOK_STAGES),
                max_concurrent=max_concurrent)
        except CountCapReached:
            queued.append(tenant.name)
            continue
        except AllocationError:
            # No capital headroom, or the tenant left the book lane between selection and the
            # write. Either way it is not fundable now; leave it and keep going — a LATER tenant
            # may still be (the slice is uniform, but a concurrent revoke can free headroom).
            queued.append(tenant.name)
            continue
        audit_append(conn, actor=actor.value, action='paper_readmit',
                     reason=f'slice {slc} (restored book allocation)', strategy=tenant.name)
        readmitted.append({'strategy': tenant.name, 'capital': slc})
        count += 1

    # ---- ADMISSION: the FIFO candidate -> paper queue.
    ordered = order_candidates(
        Candidate(name=r.name, entry_id=_candidate_entry_id(repo, r.name), sid=r.id)
        for r in repo.list_strategies(Stage.CANDIDATE))
    for i, cand in enumerate(ordered):
        if slc <= 0.0 or count >= max_concurrent:
            # Slice unfundable, or count cap already reached: queue the rest and stop.
            queued.extend(c.name for c in ordered[i:])
            break
        try:
            # Slow Git/uv/filesystem work, before (never inside) the admit's write transaction.
            admission = step(repo, cand.name, repo_root=root,
                             store_root=store_root or get_settings().data_dir)
        except _REFUSALS as exc:
            code = next(c for typ, c in _REFUSAL_CODES if isinstance(exc, typ))
            audit_append(conn, actor=actor.value, action='paper_intake_refused', reason=code,
                         strategy=cand.name)
            refused.append({'strategy': cand.name, 'code': code})
            if code == _RETRYABLE_REFUSAL:
                queued.extend(c.name for c in ordered[i + 1:])
                break
            continue
        try:
            repo.intake_candidate_to_paper(
                repo.get(cand.name), capital=slc, actor=actor,
                account_equity=equity, max_concurrent=max_concurrent,
                deployment_manifest=admission.manifest,
                research_gate_id=admission.research_gate_id)
        except (CountCapReached, AllocationError):
            # Hard bound in-txn (book full or no capital headroom): queue the rest, stop.
            queued.extend(c.name for c in ordered[i:])
            break
        except TransitionError:
            # Stale selection: a concurrent transition moved this candidate out of `candidate`
            # before the CAS. Already handled elsewhere — pass over it, keep admitting.
            skipped_stale.append(cand.name)
            continue
        audit_append(conn, actor=actor.value, action='paper_intake',
                     reason=f'slice {slc}', strategy=cand.name)
        admitted.append({'strategy': cand.name, 'capital': slc})
        count += 1
    return {'admitted': admitted, 'readmitted': readmitted, 'queued': queued,
            'skipped_stale': skipped_stale, 'refused': refused, 'equity': equity, 'slice': slc,
            'occupied_before': occupied, 'max_concurrent': max_concurrent}
