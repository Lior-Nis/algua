from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A candidate strategy awaiting paper-intake admission.

    ``entry_ts`` is the ISO-8601 candidate-entry timestamp (the moment the strategy
    reached the ``candidate`` stage). It is the FIFO ordering key. ``sid`` is the
    strategy id, used only as a deterministic tie-break when two candidates share an
    ``entry_ts``.
    """

    name: str
    entry_ts: str
    sid: int


@dataclass(frozen=True)
class IntakePlan:
    """The deterministic outcome of an intake planning pass.

    ``admit`` lists the names admitted THIS pass, in admission (FIFO) order; ``queued``
    lists the names left waiting, in evaluation order. ``occupied_before`` and
    ``max_concurrent`` are echoed for auditability.
    """

    slice_capital: float
    admit: list[str]
    queued: list[str]
    occupied_before: int
    max_concurrent: int


def slice_capital(equity: float, max_concurrent: int) -> float:
    """Per-strategy capital slice, floored to whole cents.

    Flooring (never rounding up) guarantees that ``k`` slices sum to ``<= equity`` for
    any ``k <= max_concurrent``, so admitting up to the concurrency cap can never
    over-allocate the book. If ``equity <= 0`` the result is ``<= 0``.
    """
    if max_concurrent <= 0:
        raise ValueError('max_concurrent must be positive')
    return math.floor(equity / max_concurrent * 100) / 100


def plan_intake(
    candidates: Sequence[Candidate],
    *,
    occupied_slots: int,
    total_allocated: float,
    equity: float,
    max_concurrent: int,
) -> IntakePlan:
    """Plan which queued candidates to admit to paper trading.

    Candidates are evaluated in FIFO order by ``(entry_ts, sid)``. A candidate is
    admitted iff the per-strategy slice is positive AND a concurrency slot is free AND
    the book still has equity headroom for one more slice; otherwise it is queued.

    The walk is uniform: hitting a bound queues the current candidate but does NOT stop
    evaluation of later ones. In practice a later equal-slice candidate can never fit
    once an earlier one didn't, but the loop stays uniform for clarity and to keep the
    ``queued`` list in a stable evaluation order.
    """
    slc = slice_capital(equity, max_concurrent)

    ordered = sorted(candidates, key=lambda c: (c.entry_ts, c.sid))

    admit: list[str] = []
    queued: list[str] = []
    slots = occupied_slots
    running_total = total_allocated

    for c in ordered:
        if slc > 0 and slots < max_concurrent and running_total + slc <= equity:
            admit.append(c.name)
            slots += 1
            running_total += slc
        else:
            queued.append(c.name)

    return IntakePlan(
        slice_capital=slc,
        admit=admit,
        queued=queued,
        occupied_before=occupied_slots,
        max_concurrent=max_concurrent,
    )
