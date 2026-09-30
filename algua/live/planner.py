"""Public stateless planner facade; all inputs are data and carry no operational authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
    import pandas as pd

    from algua.strategies.base import LoadedStrategy

__all__ = [
    "InProcessPlanner",
    "PlannerInput",
    "PlannerPort",
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


class PlannerPort(Protocol):
    """The three planner calls `run_tick` makes, and the seam a frozen dispatcher replaces.

    `run_tick` owns the venue-belief handshake: `phase_b` is called with a pending belief, must
    answer `VenueBeliefRequired`, and is called again with the resolved belief.
    """

    def phase_a(self, early: EarlyPlannerInput) -> EarlyPlannerResult: ...

    def closed_bars(self, early: EarlyPlannerInput) -> pd.DataFrame: ...

    def phase_b(self, late: LatePlannerInput) -> LatePlannerResult: ...


@dataclass(frozen=True)
class InProcessPlanner:
    """The default port: today's in-process facade calls on the checkout-loaded strategy."""

    strategy: LoadedStrategy

    # The bodies name the module-level facade functions, resolved at call time, so a caller that
    # observes or patches the facade sees exactly the calls `run_tick` made before the port.
    def phase_a(self, early: EarlyPlannerInput) -> EarlyPlannerResult:
        return phase_a(self.strategy, early)

    def closed_bars(self, early: EarlyPlannerInput) -> pd.DataFrame:
        return phase_a_closed_bars(self.strategy, early)

    def phase_b(self, late: LatePlannerInput) -> LatePlannerResult:
        return phase_b(self.strategy, late)
