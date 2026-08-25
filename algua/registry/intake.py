"""`paper intake` (#317): the per-strategy capital slice, the deterministic FIFO ordering of
candidates, and the admission loop that offers them to the atomic primitive.

The admission DECISION is NOT made here — it is made transactionally, one candidate at a time, by
``StrategyRepository.intake_candidate_to_paper`` (which re-checks the count cap and the Σ≤equity
capital bound UNDER the write lock). This module computes the fixed slice and the stable order in
which candidates are offered to that primitive, then drives ``run_intake``'s loop over them one at
a time — it never does its own bounds-check, so there is no second one to drift from the
authoritative in-transaction one.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from algua.audit.log import append as audit_append
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import allocations
from algua.registry.allocations import (
    AllocationError,
    CountCapReached,
    active_allocation,
    active_paper_lane_count,
)
from algua.registry.store import SqliteStrategyRepository


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

    **ADMISSION** (``admitted``) offers each candidate, in FIFO order (candidate-entry
    ``stage_transitions.id``, tie-break
    strategy id), to the ATOMIC ``intake_candidate_to_paper`` primitive, which under ONE write lock
    re-checks the ``max_concurrent`` count cap, cap-checks + allocates an equal slice =
    floor(equity / max_concurrent to cents) (Σ allocations + slice ≤ ``equity``), and CASes
    candidate→paper — commit-or-rollback together, so there is no reachable transitioned-but-
    unallocated state. On either hard bound (book full / no capital headroom) the remaining
    candidates are left queued; a candidate raced out of ``candidate`` between selection and the txn
    is reported in ``skipped_stale`` and passed over. Returns the intake envelope dict.

    The caller is responsible for reading ``equity`` READ-ONLY from the broker BEFORE opening
    ``conn`` (no trading) and for validating ``max_concurrent`` > 0."""
    repo = SqliteStrategyRepository(conn)
    occupied = active_paper_lane_count(conn)
    slc = slice_capital(equity, max_concurrent)
    admitted: list[dict] = []
    readmitted: list[dict] = []
    queued: list[str] = []
    skipped_stale: list[str] = []
    count = occupied

    # ---- RE-ADMISSION first: restore the book slice of an already-admitted tenant that lost it.
    # These strategies are ALREADY at a book stage — they passed the candidate gates and were
    # admitted once; a bench/demotion round-trip revoked their allocation and the return edge
    # restored only the stage. A newcomer must not take the slot out from under one, so they go to
    # the front of the FIFO. No stage changes here: `allocate_in_lane` re-reads the stage under the
    # SAME write lock and applies the SAME Σ ≤ equity and count-cap bounds intake applies.
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
            repo.intake_candidate_to_paper(
                repo.get(cand.name), capital=slc, actor=actor,
                account_equity=equity, max_concurrent=max_concurrent)
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
            'skipped_stale': skipped_stale, 'equity': equity, 'slice': slc,
            'occupied_before': occupied, 'max_concurrent': max_concurrent}
