"""Pure helpers for `paper intake` (#317): the per-strategy capital slice and the deterministic
FIFO ordering of candidates.

The admission DECISION is NOT made here — it is made transactionally, one candidate at a time, by
``StrategyRepository.intake_candidate_to_paper`` (which re-checks the count cap and the Σ≤equity
capital bound UNDER the write lock). This module only computes the fixed slice and the stable order
in which candidates are offered to that primitive, so there is no second bounds-check to drift from
the authoritative in-transaction one.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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
    """Round a dollar amount to whole integer cents (nearest)."""
    return round(dollars * 100)


def slice_capital(equity: float, max_concurrent: int) -> float:
    """Per-strategy capital slice in dollars, floored to whole cents.

    The floor (never rounding up) is computed in INTEGER CENTS — ``equity_cents //
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
