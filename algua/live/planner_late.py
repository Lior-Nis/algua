"""Phase B: binding verification, the strategy-free late verdict and the decision output.

`verified_phase_a` and `late_verdict` are Phase B up to the strategy's weights and need no strategy
code: the in-process planner and a frozen tick's supervisor both compute them (Story 1.3c contract
§7), so every captured-state wall -- equity, drawdown, venue reconcile, realized gross and the
marks re-check -- has one implementation, owned by the current supervisor.
"""

from __future__ import annotations

import hmac
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.calendar.market_calendar import MarketCalendar
from algua.live.planner_contract import (
    CapturedStrategyState,
    Decision,
    EarlyPlannerInput,
    EarlyPlannerResult,
    LateNoDecision,
    LatePlannerInput,
    LatePlannerResult,
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.live.planner_decision import PlannerInput, PlannerResult
from algua.live.planner_early import (
    assert_marks_usable,
    closed_universe_bars,
    latest_values,
    risk_failure,
)
from algua.live.planner_validation import input_failure
from algua.risk.limits import WEIGHT_TOL, RiskBreach, check_drawdown

if TYPE_CHECKING:
    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import LoadedStrategy

_RECONCILE_TOL = 1e-6
_BINDING = re.compile(r"[0-9a-f]{64}\Z")
type PhaseAFn = Callable[[LoadedStrategy, EarlyPlannerInput], EarlyPlannerResult]
type PlanFn = Callable[[LoadedStrategy, PlannerInput], PlannerResult]


def _check_realized_gross(gross: float, max_gross: float) -> None:
    if gross > max_gross + WEIGHT_TOL:
        raise RiskBreach(
            "gross_exposure_realized",
            f"realized gross exposure {gross:.4f} exceeds max_gross_exposure {max_gross:.4f}",
        )


def _late_state(
    captured: CapturedStrategyState, decision_ts: datetime | None
) -> tuple[PlannerState, dict[str, float]]:
    if type(captured.sizing_equity) not in (int, float):
        raise ValueError("sizing_equity must be numeric")
    sizing_equity = float(captured.sizing_equity)
    if not (math.isfinite(sizing_equity) and sizing_equity > 0.0):
        raise RiskBreach(
            "non_positive_equity",
            f"sizing equity {sizing_equity} is not a usable (positive, finite) "
            "denominator - refusing to trade before it divides by zero, inverts weights, or "
            "NaN-poisons",
        )
    drawdown_equity = _finite_number(captured.drawdown_equity, "drawdown_equity")
    persisted_peak = (
        None
        if captured.persisted_peak_equity is None
        else _finite_number(captured.persisted_peak_equity, "persisted_peak_equity")
    )
    quantities = _numeric_mapping(captured.quantities, "quantities")
    market_values = _numeric_mapping(captured.market_values, "market_values")
    quantity_symbols = {symbol for symbol, value in quantities.items() if value != 0.0}
    value_symbols = {symbol for symbol, value in market_values.items() if value != 0.0}
    if quantity_symbols != value_symbols:
        raise ValueError("nonzero quantity and market-value symbol sets must match")
    peak = drawdown_equity if persisted_peak is None else max(persisted_peak, drawdown_equity)
    positions = {symbol: qty for symbol, qty in quantities.items() if qty != 0.0}
    current_weights = {
        symbol: value / sizing_equity for symbol, value in market_values.items() if value != 0.0
    }
    return PlannerState(
        decision_ts,
        (),
        tuple(positions.items()),
        drawdown_equity,
        float(peak),
        True,
        # Sum in symbol order: float addition is order-sensitive and a frozen child sees mappings
        # sorted, while captured values arrive in broker order (Story 1.3c parity).
        sum(abs(current_weights[symbol]) for symbol in sorted(current_weights)),
    ), current_weights


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    assert isinstance(value, (int, float))
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _numeric_mapping(value: object, label: str) -> dict[str, float]:
    from collections.abc import Mapping

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for symbol, number in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{label} keys must be non-empty strings")
        normalized = unicodedata.normalize("NFC", symbol)
        if normalized in result:
            raise ValueError(f"{label} keys collide after Unicode normalization")
        result[normalized] = _finite_number(number, f"{label}[{symbol!r}]")
    return result


@dataclass(frozen=True)
class LateProceed:
    """Every strategy-free wall passed: the strategy plans on ``view`` from ``current_weights``."""

    state: PlannerState
    current_weights: dict[str, float]
    view: pd.DataFrame
    decision_ts: datetime


type LateVerdict = (
    PlannerRiskFailure | PlannerInputFailure | VenueBeliefRequired | LateNoDecision | LateProceed
)


def verified_phase_a(
    late: LatePlannerInput, recomputed: EarlyPlannerResult
) -> SnapshotRequired | PhaseBindingFailure | PlannerInputFailure:
    """Phase A recomputed from ``late.early``, if the late input is bound to it."""
    if not isinstance(recomputed, SnapshotRequired):
        return PhaseBindingFailure(
            "phase_a_outcome_mismatch", "recomputed Phase A no longer requires a snapshot"
        )
    if not isinstance(late.phase_a_binding, str) or not _BINDING.fullmatch(late.phase_a_binding):
        return PhaseBindingFailure(
            "invalid_phase_a_binding", "binding must be 64 lowercase hex chars"
        )
    if not hmac.compare_digest(late.phase_a_binding, recomputed.phase_a_binding):
        return PhaseBindingFailure("phase_a_binding_mismatch", "Phase A binding does not match")
    if late.captured.request_id != late.early.request_id:
        return input_failure("request_id_mismatch", "captured state belongs to another request")
    return recomputed


def late_verdict(
    early: EarlyPlannerInput,
    captured: CapturedStrategyState,
    phase_a: SnapshotRequired,
    execution: ExecutionContract,
) -> LateVerdict:
    """Phase B's outcome before the strategy plans: a pending belief is asked for once equity and
    drawdown pass; a resolved one is reconciled, then realized gross, warm-up and every consumed
    mark are checked."""
    try:
        state, current_weights = _late_state(captured, phase_a.decision_ts)
        assert state.peak_equity is not None
        check_drawdown(state.equity, state.peak_equity, early.max_drawdown)
        if isinstance(captured.venue_belief, VenueBeliefPending):
            if captured.venue_belief.tag != "pending":
                return input_failure("invalid_venue_belief", "pending belief has an invalid tag")
            return VenueBeliefRequired()
        if isinstance(captured.venue_belief, VenueBeliefEnabled):
            if captured.venue_belief.tag != "enabled":
                return input_failure("invalid_venue_belief", "enabled belief has an invalid tag")
            belief = {
                symbol: qty
                for symbol, qty in _numeric_mapping(
                    captured.venue_belief.quantities, "venue_belief.quantities"
                ).items()
                if qty != 0.0
            }
            positions = dict(state.positions_before)
            drift = [
                symbol
                for symbol in set(belief) | set(positions)
                if abs(belief.get(symbol, 0.0) - positions.get(symbol, 0.0)) > _RECONCILE_TOL
            ]
            if drift:
                raise RiskBreach(
                    "reconcile",
                    f"venue belief {dict(sorted(belief.items()))} disagrees with positions_before "
                    f"{dict(sorted(positions.items()))} before "
                    "tick - refusing to trade on inconsistent state",
                )
        elif not isinstance(captured.venue_belief, VenueBeliefDisabled):
            return input_failure("invalid_venue_belief", "unknown venue-belief variant")
        elif captured.venue_belief.tag != "disabled":
            return input_failure("invalid_venue_belief", "disabled belief has an invalid tag")
        _check_realized_gross(state.realized_gross, execution.max_gross_exposure)
        if phase_a.warming:
            return LateNoDecision("warming", state)
        closed = closed_universe_bars(early.raw_bars, early.now, early.gate_universe)
        latest_ts, latest_close = latest_values(closed.bars)
        assert_marks_usable(
            set(dict(state.positions_before)) | set(early.gate_universe),
            latest_ts,
            latest_close,
            early.now,
            MarketCalendar(early.calendar_code),
        )
        if phase_a.decision_ts is None:
            return input_failure("missing_decision_time", "decision path has no timestamp")
        view = closed.universe_bars.loc[: phase_a.decision_ts]
        return LateProceed(state, current_weights, view, phase_a.decision_ts)
    except RiskBreach as exc:
        return risk_failure(exc)
    except (TypeError, ValueError) as exc:
        return input_failure("invalid_captured_state", str(exc))


def phase_b_impl(
    strategy: LoadedStrategy, late: LatePlannerInput, *, phase_a_fn: PhaseAFn, plan_fn: PlanFn
) -> LatePlannerResult:
    """Recompute Phase A, verify its binding, run the late verdict, then plan."""
    phase_a = verified_phase_a(late, phase_a_fn(strategy, late.early))
    if not isinstance(phase_a, SnapshotRequired):
        return phase_a
    verdict = late_verdict(late.early, late.captured, phase_a, strategy.execution)
    if not isinstance(verdict, LateProceed):
        return verdict
    try:
        result = plan_fn(
            strategy, PlannerInput(verdict.view, verdict.current_weights, verdict.decision_ts)
        )
        state = verdict.state
        final = PlannerState(
            state.decision_ts,
            tuple((str(symbol), float(weight)) for symbol, weight in result.weights.items()),
            state.positions_before,
            state.equity,
            state.peak_equity,
            state.reconcile_ok,
            state.realized_gross,
        )
        return Decision(final, tuple(result.intents))
    except RiskBreach as exc:
        return risk_failure(exc)
    except (TypeError, ValueError) as exc:
        return input_failure("invalid_captured_state", str(exc))
