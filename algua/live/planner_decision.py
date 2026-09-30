"""Pure target-weight validation and intent construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.contracts.types import OrderIntent, Side
from algua.risk.limits import WEIGHT_TOL, check_finite_weights, validate_decision_weights

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy


@dataclass(frozen=True)
class PlannerInput:
    view: pd.DataFrame
    current_weights: dict[str, float]
    decision_ts: datetime
    protocol_version: int = PLANNER_PROTOCOL_VERSION


@dataclass(frozen=True)
class PlannerResult:
    weights: pd.Series
    intents: list[OrderIntent]
    protocol_version: int = PLANNER_PROTOCOL_VERSION


def build_intents(
    weights: pd.Series, current_weights: dict[str, float], decision_ts: datetime
) -> list[OrderIntent]:
    """Emit sorted target-weight deltas, including zero targets for dropped holdings."""
    intents: list[OrderIntent] = []
    for symbol in sorted(set(weights.index) | set(current_weights)):
        target = float(weights.get(symbol, 0.0))
        current = float(current_weights.get(symbol, 0.0))
        if abs(target - current) > WEIGHT_TOL:
            side = Side.BUY if target > current else Side.SELL
            intents.append(OrderIntent(symbol, side, target, decision_ts))
    return intents


def canonical_weights(weights: pd.Series, strategy_name: str) -> pd.Series:
    """The strategy's weights as float64 in symbol order: the form a frozen tick's supervisor
    re-validates them in (Story 1.3c §7), so a float32 weight at a cap, or a gross summed in
    another order, cannot pass one side and breach the other. The dtype guards run on the
    strategy's own Series first, so a bool or string weight still breaches instead of coercing."""
    check_finite_weights(weights, strategy_name)
    return weights.astype("float64").sort_index(key=lambda index: index.map(str))


def plan(strategy: LoadedStrategy, inputs: PlannerInput) -> PlannerResult:
    if (
        type(inputs.protocol_version) is not int
        or inputs.protocol_version != PLANNER_PROTOCOL_VERSION
    ):
        raise ValueError(f"unsupported planner protocol: {inputs.protocol_version!r}")
    weights = canonical_weights(strategy.target_weights(inputs.view), strategy.name)
    validate_decision_weights(
        weights, strategy.execution, strategy.name, allowed_symbols=strategy.universe
    )
    return PlannerResult(
        weights, build_intents(weights, inputs.current_weights, inputs.decision_ts)
    )


def decide(
    strategy: LoadedStrategy,
    view: pd.DataFrame,
    current_weights: dict[str, float],
    decision_ts: datetime,
) -> tuple[pd.Series, list[OrderIntent]]:
    result = plan(strategy, PlannerInput(view, current_weights, decision_ts))
    return result.weights, result.intents
