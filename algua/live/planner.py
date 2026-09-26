"""Public stateless planner facade; all inputs are data and carry no operational authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

from algua.live.planner_contract import (
    EarlyPlannerInput,
    EarlyPlannerResult,
    LatePlannerInput,
    LatePlannerResult,
)
from algua.live.planner_decision import (
    PlannerInput,
    PlannerResult,
    build_intents,
    decide,
    plan,
)
from algua.live.planner_early import phase_a as _phase_a
from algua.live.planner_early import prepare_early
from algua.live.planner_late import phase_b_impl

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy

__all__ = [
    "PlannerInput",
    "PlannerResult",
    "build_intents",
    "decide",
    "phase_a",
    "phase_a_closed_bars",
    "phase_b",
    "plan",
]


def phase_a(strategy: LoadedStrategy, early: EarlyPlannerInput) -> EarlyPlannerResult:
    return _phase_a(strategy, early)


def phase_a_closed_bars(strategy: LoadedStrategy, early: EarlyPlannerInput):
    """Return Phase A's canonical closed union frame for supervisor-owned valuation."""
    return prepare_early(strategy, early).bars


def phase_b(strategy: LoadedStrategy, late: LatePlannerInput) -> LatePlannerResult:
    # Resolve facade globals on every invocation: Phase B recomputes Phase A without retaining
    # state, and both public seams remain independently observable by the future dispatcher.
    return phase_b_impl(strategy, late, phase_a_fn=phase_a, plan_fn=plan)
