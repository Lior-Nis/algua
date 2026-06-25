from __future__ import annotations

from typing import Protocol


class FamilyBudgetLedger(Protocol):
    """Stratum B: partition of global LORD++ alpha wealth across families.

    Real implementation deferred to Phase 2 (#220). p = 1 - dsr_confidence (Phase 1 convention).
    reserve() is called in run_gate AFTER evaluate_gate computes DSR confidence; it is NOT called
    in promotion_preflight. gate_eval_id is NOT in the signature (R2-F1: the gate row doesn't
    exist at reserve() time).
    """

    def global_cap(self) -> float:
        """Total alpha wealth W_global."""
        ...

    def family_wealth(self, family_id: int) -> float:
        """Current unallocated alpha wealth for family_id (α_f · W_global remaining)."""
        ...

    def reserve(self, family_id: int, p_value: float, actor: str) -> str | None:
        """Allocate p_value of alpha from family_id's budget.

        Returns the reservation key (a str) on success; None if budget exhausted
        (promotion blocked). Pass the returned key to settle() when the strategy is
        retired/dormant. Only called when dsr_confidence is not None.
        """
        ...

    def settle(self, family_id: int, reservation_key: str) -> None:
        """Release the reservation (strategy retired/dormant); return alpha wealth to family."""
        ...


class InMemoryFamilyBudgetLedger:
    """Fake implementation for tests and Stratum A use. No persistence. Single global wealth pool.
    All families share the same global cap (no per-family allocation until Phase 2 ships)."""

    def __init__(self, global_cap: float = 1.0) -> None:
        self._cap = global_cap
        self._counter = 0
        self._reservations: dict[str, float] = {}  # reservation_key -> p_value

    def global_cap(self) -> float:
        return self._cap

    def family_wealth(self, family_id: int) -> float:
        allocated = sum(self._reservations.values())
        return max(0.0, self._cap - allocated)

    def reserve(self, family_id: int, p_value: float, actor: str) -> str | None:
        allocated = sum(self._reservations.values())
        if allocated + p_value > self._cap + 1e-10:
            return None
        self._counter += 1
        key = str(self._counter)
        self._reservations[key] = p_value
        return key

    def settle(self, family_id: int, reservation_key: str) -> None:
        self._reservations.pop(reservation_key, None)
