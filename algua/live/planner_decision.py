"""Pure target-weight validation and intent construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.contracts.types import OrderIntent, Side
from algua.risk.limits import WEIGHT_TOL, validate_decision_weights

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


def plan(strategy: LoadedStrategy, inputs: PlannerInput) -> PlannerResult:
    if (
        type(inputs.protocol_version) is not int
        or inputs.protocol_version != PLANNER_PROTOCOL_VERSION
    ):
        raise ValueError(f"unsupported planner protocol: {inputs.protocol_version!r}")
    weights = strategy.target_weights(inputs.view)
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
