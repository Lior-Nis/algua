"""Immutable logical values crossing the two-phase in-process planner boundary.

These values deliberately carry data, never acquisition or execution authority.  The pandas frame
is treated as an immutable captured value by the API; Story 1.3c will give it a wire encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from algua.contracts.types import OrderIntent

BOUNDARY_VERSION = 1


@dataclass(frozen=True)
class EarlyPlannerInput:
    boundary_version: int
    request_id: str
    strategy_name: str
    deployment_id: int | None
    artifact_id: int | None
    manifest_digest: str | None
    config_hash: str
    resolved_config_json: str
    now: datetime
    timeframe: str
    calendar_code: str
    raw_bars: pd.DataFrame
    early_positions: Mapping[str, float]
    gate_universe: tuple[str, ...]
    max_drawdown: float | None


@dataclass(frozen=True)
class PlannerState:
    decision_ts: datetime | None
    target_weights: tuple[tuple[str, float], ...]
    positions_before: tuple[tuple[str, float], ...]
    equity: float
    peak_equity: float | None
    reconcile_ok: bool
    realized_gross: float


@dataclass(frozen=True)
class EarlyNoDecision:
    reason: Literal["no_bars", "warming"]
    state: PlannerState


@dataclass(frozen=True)
class PlannerRiskFailure:
    kind: str
    detail: str
    is_dark_feed: bool


@dataclass(frozen=True)
class PlannerInputFailure:
    code: str
    detail: str


@dataclass(frozen=True)
class SnapshotRequired:
    decision_ts: datetime | None
    warming: bool
    phase_a_binding: str


type EarlyPlannerResult = (
    EarlyNoDecision | PlannerRiskFailure | PlannerInputFailure | SnapshotRequired
)


@dataclass(frozen=True)
class VenueBeliefDisabled:
    tag: Literal["disabled"] = "disabled"


@dataclass(frozen=True)
class VenueBeliefEnabled:
    quantities: Mapping[str, float]
    tag: Literal["enabled"] = "enabled"


@dataclass(frozen=True)
class VenueBeliefPending:
    """Late state captured before the supervisor reads the venue-belief hook."""

    tag: Literal["pending"] = "pending"


type VenueBelief = VenueBeliefPending | VenueBeliefDisabled | VenueBeliefEnabled


@dataclass(frozen=True)
class CapturedStrategyState:
    request_id: str
    sizing_equity: float
    drawdown_equity: float
    quantities: Mapping[str, float]
    market_values: Mapping[str, float]
    persisted_peak_equity: float | None
    venue_belief: VenueBelief


@dataclass(frozen=True)
class LatePlannerInput:
    early: EarlyPlannerInput
    phase_a_binding: str
    captured: CapturedStrategyState


@dataclass(frozen=True)
class PhaseBindingFailure:
    code: str
    detail: str


@dataclass(frozen=True)
class VenueBeliefRequired:
    """Equity and drawdown passed; rerun Phase B after capturing venue belief."""


@dataclass(frozen=True)
class LateNoDecision:
    reason: Literal["warming"]
    state: PlannerState


@dataclass(frozen=True)
class Decision:
    state: PlannerState
    ordered_intents: tuple[OrderIntent, ...]


type LatePlannerResult = (
    PhaseBindingFailure
    | PlannerRiskFailure
    | PlannerInputFailure
    | VenueBeliefRequired
    | LateNoDecision
    | Decision
)
