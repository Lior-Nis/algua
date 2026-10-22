"""Pure target-weight validation and intent construction."""

from __future__ import annotations

import math
import unicodedata
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


def _finite_number(value: object, label: str) -> float:
    """The shared finite-number contract for captured state and current holding weights."""
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    assert isinstance(value, (int, float))
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _numeric_mapping(value: object, label: str) -> dict[str, float]:
    """Non-empty NFC string keys and finite values. One copy, shared with the late planner."""
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


def build_intents(
    weights: pd.Series, current_weights: dict[str, float], decision_ts: datetime
) -> list[OrderIntent]:
    """Emit sorted target-weight deltas, including zero targets for dropped holdings.

    ``current_weights`` uses the same contract as captured late-planner mappings: non-empty
    string keys and finite values. A violation raises before any intent is built.
    """
    holdings = _numeric_mapping(current_weights, "current_weights")
    intents: list[OrderIntent] = []
    for symbol in sorted(set(weights.index) | set(holdings)):
        target = float(weights.get(symbol, 0.0))
        current = float(holdings.get(symbol, 0.0))
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
