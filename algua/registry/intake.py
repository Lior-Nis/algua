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
from decimal import ROUND_FLOOR, Decimal


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
