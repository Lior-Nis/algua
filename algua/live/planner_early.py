"""Phase A validation, timing, freshness and integrity binding."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.calendar.market_calendar import MarketCalendar
from algua.live.planner_binding import phase_a_binding
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    EarlyNoDecision,
    EarlyPlannerInput,
    EarlyPlannerResult,
    PlannerInputFailure,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
)
from algua.risk.limits import MAX_STALE_SESSIONS, RiskBreach, check_mark_freshness

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy

_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH_32 = re.compile(r"[0-9a-f]{32}\Z")
_HASH_64 = re.compile(r"[0-9a-f]{64}\Z")


def input_failure(code: str, detail: str) -> PlannerInputFailure:
    return PlannerInputFailure(code, detail)


def risk_failure(exc: RiskBreach) -> PlannerRiskFailure:
    return PlannerRiskFailure(exc.kind, exc.detail, exc.is_dark_feed)


def _empty_state(early: EarlyPlannerInput, decision_ts: datetime | None = None) -> PlannerState:
    return PlannerState(
        decision_ts,
        (),
        tuple(
            sorted(
                (_normalized(symbol), float(qty)) for symbol, qty in early.early_positions.items()
            )
        ),
        0.0,
        None,
        True,
        0.0,
    )


def _validate_early(
    strategy: LoadedStrategy, early: EarlyPlannerInput
) -> PlannerInputFailure | None:
    if type(early.boundary_version) is not int or early.boundary_version != BOUNDARY_VERSION:
        return input_failure("unsupported_boundary_version", str(early.boundary_version))
    if not isinstance(early.request_id, str) or not _REQUEST_ID.fullmatch(early.request_id):
        return input_failure("invalid_request_id", "request_id must be 32 lowercase hex chars")
    if (
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
        gate_universe = _normalized_symbols(early.gate_universe, "gate_universe")
        strategy_universe = _normalized_symbols(strategy.universe, "strategy universe")
    except ValueError as exc:
        return input_failure("invalid_gate_universe", str(exc))
    if gate_universe != strategy_universe:
        return input_failure("strategy_identity_mismatch", "gate universe does not match strategy")
    try:
        from algua.strategies.base import config_hash

        if hasattr(strategy, "config"):
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


def _normalized(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", value)


def _normalized_symbols(values: Collection[str], label: str) -> tuple[str, ...]:
    normalized = tuple(_normalized(value) for value in values if isinstance(value, str) and value)
    if len(normalized) != len(values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} collides after Unicode normalization")
    return normalized


def _latest_values(bars: pd.DataFrame) -> tuple[dict[str, datetime], dict[str, float]]:
    tail = bars.iloc[0:0] if bars.empty else bars.groupby("symbol", sort=False).tail(1)
    timestamps = {
        str(symbol): timestamp for timestamp, symbol in zip(tail.index, tail["symbol"], strict=True)
    }
    closes = {
        str(symbol): float(close)
        for symbol, close in zip(tail["symbol"], tail["close"], strict=True)
    }
    return timestamps, closes


def assert_marks_usable(
    symbols: Collection[str],
    latest_ts: Mapping[str, datetime],
    latest_close: Mapping[str, float],
    now: datetime,
    calendar: MarketCalendar,
) -> None:
    unvaluable = sorted(
        symbol
        for symbol in symbols
        if symbol in latest_close
        and not (math.isfinite(latest_close[symbol]) and latest_close[symbol] > 0.0)
    )
    if unvaluable:
        raise RiskBreach(
            "unvaluable_marks",
            f"held/consumed symbols have a non-positive / non-finite mark: {unvaluable} — "
            "refusing to value/size the book off an unvaluable feed",
        )
    stale: dict[str, float] = {}
    for symbol in sorted(symbols):  # symbol order, so an unmappable mark is named deterministically
        timestamp = latest_ts.get(symbol)
        if timestamp is None:
            stale[symbol] = math.inf
            continue
        try:
            stale[symbol] = float(calendar.sessions_stale(timestamp, now))
        except Exception as exc:
            raise RiskBreach(
                "stale_marks",
                f"cannot map {symbol} mark {timestamp} to an exchange session ({exc!r}) — "
                "refusing to establish risk state off an unmappable timestamp",
            ) from exc
    check_mark_freshness(stale, MAX_STALE_SESSIONS)


@dataclass(frozen=True)
class ClosedBars:
    bars: pd.DataFrame
    universe_bars: pd.DataFrame
    decision_ts: datetime | None


def closed_universe_bars(
    raw_bars: pd.DataFrame, now: datetime, gate_universe: Collection[str]
) -> ClosedBars:
    """Phase A's strategy-free closed-bar selection: NFC symbols, float64 values, a stable
    timestamp sort and only sessions dated before `now`; the decision time is the latest closed
    gate-universe bar. A frozen tick's supervisor recomputes this to check the child's result."""
    bars = raw_bars.copy()
    bars["symbol"] = bars["symbol"].map(_normalized)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        bars[column] = bars[column].astype("float64")
    bars = bars.sort_index(kind="stable")
    if not bars.empty:
        bars = bars[[timestamp.date() < now.date() for timestamp in bars.index]]
    universe_bars = bars[bars["symbol"].isin(_normalized_symbols(gate_universe, "gate_universe"))]
    decision_ts = universe_bars.index.max() if not universe_bars.empty else None
    return ClosedBars(bars, universe_bars, decision_ts)


@dataclass(frozen=True)
class PreparedEarly:
    bars: pd.DataFrame
    universe_bars: pd.DataFrame
    latest_ts: Mapping[str, datetime]
    latest_close: Mapping[str, float]
    decision_ts: datetime | None
    warming: bool


def prepare_early(strategy: LoadedStrategy, early: EarlyPlannerInput) -> PreparedEarly:
    closed = closed_universe_bars(early.raw_bars, early.now, early.gate_universe)
    warming = closed.universe_bars.index.nunique() <= strategy.execution.warmup_bars
    latest_ts, latest_close = _latest_values(closed.bars)
    return PreparedEarly(
        closed.bars, closed.universe_bars, latest_ts, latest_close, closed.decision_ts, warming
    )


def phase_a(strategy: LoadedStrategy, early: EarlyPlannerInput) -> EarlyPlannerResult:
    failure = _validate_early(strategy, early)
    if failure is not None:
        return failure
    prepared = prepare_early(strategy, early)
    held = {_normalized(symbol) for symbol, qty in early.early_positions.items() if qty != 0.0}
    try:
        if held:
            assert_marks_usable(
                held,
                prepared.latest_ts,
                prepared.latest_close,
                early.now,
                MarketCalendar(early.calendar_code),
            )
    except RiskBreach as exc:
        return risk_failure(exc)
    if prepared.bars.empty:
        return EarlyNoDecision("no_bars", _empty_state(early))
    if prepared.warming and not held:
        return EarlyNoDecision("warming", _empty_state(early, prepared.decision_ts))
    binding = phase_a_binding(early, decision_ts=prepared.decision_ts, warming=prepared.warming)
    return SnapshotRequired(prepared.decision_ts, prepared.warming, binding)
