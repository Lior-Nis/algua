"""Phase A input validation: the checks that make an early input well formed and, given the
strategy, bind it to that strategy.

`validate_early(early, strategy)` is the in-process planner's check. Given no strategy it runs only
the strategy-free checks, in the same order: a frozen tick's supervisor validates its own input
with it before computing the strategy-free verdict itself (Story 1.3c contract §7), so a malformed
input is refused exactly as the planner refuses it and is never read as a breach.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.calendar.market_calendar import MarketCalendar
from algua.live.planner_binding import phase_a_binding
from algua.live.planner_contract import BOUNDARY_VERSION, EarlyPlannerInput, PlannerInputFailure

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy

_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH_32 = re.compile(r"[0-9a-f]{32}\Z")
_HASH_64 = re.compile(r"[0-9a-f]{64}\Z")


def input_failure(code: str, detail: str) -> PlannerInputFailure:
    return PlannerInputFailure(code, detail)


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalized_symbols(values: Collection[str], label: str) -> tuple[str, ...]:
    normalized_values = tuple(
        normalized(value) for value in values if isinstance(value, str) and value
    )
    if len(normalized_values) != len(values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(normalized_values)) != len(normalized_values):
        raise ValueError(f"{label} collides after Unicode normalization")
    return normalized_values


def validate_early(
    early: EarlyPlannerInput, strategy: LoadedStrategy | None
) -> PlannerInputFailure | None:
    """The first failure, or ``None``; without a strategy, the strategy-free checks only."""
    if type(early.boundary_version) is not int or early.boundary_version != BOUNDARY_VERSION:
        return input_failure("unsupported_boundary_version", str(early.boundary_version))
    if not isinstance(early.request_id, str) or not _REQUEST_ID.fullmatch(early.request_id):
        return input_failure("invalid_request_id", "request_id must be 32 lowercase hex chars")
    if strategy is not None and (
        not isinstance(early.strategy_name, str)
        or not early.strategy_name
        or early.strategy_name != strategy.name
    ):
        return input_failure("strategy_identity_mismatch", "strategy name does not match")
    deployment_values = (early.deployment_id, early.artifact_id, early.manifest_digest)
    if not all(value is None for value in deployment_values) and (
        type(early.deployment_id) is not int
        or early.deployment_id <= 0
        or type(early.artifact_id) is not int
        or early.artifact_id <= 0
        or not isinstance(early.manifest_digest, str)
        or not _HASH_64.fullmatch(early.manifest_digest)
    ):
        return input_failure("invalid_deployment_identity", "deployment identity is inconsistent")
    if not isinstance(early.config_hash, str) or not _HASH_32.fullmatch(early.config_hash):
        return input_failure("invalid_config_hash", "config_hash must be 32 lowercase hex chars")
    if not isinstance(early.resolved_config_json, str):
        return input_failure("invalid_resolved_config", "resolved_config_json must be a string")
    if not isinstance(early.now, datetime) or str(pd.Timestamp(early.now).tz) != "UTC":
        return input_failure("invalid_now", "now must be a UTC datetime")
    if not isinstance(early.timeframe, str) or early.timeframe != "1d":
        return input_failure(
            "unsupported_timeframe",
            f"mark-freshness wall supports only 1d bars; got {early.timeframe!r}",
        )
    if not isinstance(early.calendar_code, str) or not early.calendar_code:
        return input_failure("invalid_calendar", "calendar_code must be non-empty")
    if not isinstance(early.raw_bars, pd.DataFrame):
        return input_failure("invalid_raw_bars", "raw_bars must be a pandas DataFrame")
    if not isinstance(early.early_positions, Mapping) or not isinstance(early.gate_universe, tuple):
        return input_failure(
            "invalid_early_input", "positions must be a mapping and universe a tuple"
        )
    if early.max_drawdown is not None and (
        type(early.max_drawdown) not in (int, float)
        or not math.isfinite(float(early.max_drawdown))
        or not 0.0 <= float(early.max_drawdown) <= 1.0
    ):
        return input_failure("invalid_max_drawdown", "max_drawdown must be finite in [0, 1]")
    try:
        gate_universe = normalized_symbols(early.gate_universe, "gate_universe")
        strategy_universe = (
            None
            if strategy is None
            else normalized_symbols(strategy.universe, "strategy universe")
        )
    except ValueError as exc:
        return input_failure("invalid_gate_universe", str(exc))
    if strategy_universe is not None and gate_universe != strategy_universe:
        return input_failure("strategy_identity_mismatch", "gate universe does not match strategy")
    try:
        from algua.strategies.base import config_hash

        if strategy is not None and hasattr(strategy, "config"):
            resolved_data = json.loads(early.resolved_config_json)
            if not isinstance(resolved_data, dict) or not isinstance(
                resolved_data.get("universe"), list
            ):
                return input_failure(
                    "invalid_resolved_config", "resolved config must contain a universe list"
                )
            identity_config = strategy.config.model_copy(
                update={"universe": resolved_data["universe"]}
            )
            identity_strategy = replace(strategy, config=identity_config)
            if config_hash(identity_strategy) != early.config_hash:
                return input_failure(
                    "strategy_identity_mismatch", "config hash does not match strategy"
                )
            expected_effective = dict(resolved_data)
            expected_effective["universe"] = list(gate_universe)
            if expected_effective != strategy.config.model_dump(mode="json"):
                return input_failure(
                    "strategy_identity_mismatch",
                    "resolved config does not match effective strategy",
                )
        phase_a_binding(early, decision_ts=None, warming=False)
        MarketCalendar(early.calendar_code)
    except (TypeError, ValueError) as exc:
        return input_failure("invalid_early_input", str(exc))
    return None
