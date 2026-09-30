"""Phase A: closed-bar selection, mark freshness and the strategy-free early verdict.

`early_verdict` is Phase A after validation, and needs no strategy code: held-name marks, then no
bars, then a flat warm-up, else the snapshot. The in-process planner and a frozen tick's supervisor
both compute it (Story 1.3c contract §7), so a strategy-free wall has one implementation and the
current supervisor, not a frozen bundle's copy, owns it.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from algua.calendar.market_calendar import MarketCalendar
from algua.live.planner_binding import phase_a_binding
from algua.live.planner_contract import (
    EarlyNoDecision,
    EarlyPlannerInput,
    EarlyPlannerResult,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
)
from algua.live.planner_validation import normalized, normalized_symbols, validate_early
from algua.risk.limits import MAX_STALE_SESSIONS, RiskBreach, check_mark_freshness

if TYPE_CHECKING:
    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import LoadedStrategy

type EarlyVerdict = PlannerRiskFailure | EarlyNoDecision | SnapshotRequired


def risk_failure(exc: RiskBreach) -> PlannerRiskFailure:
    return PlannerRiskFailure(exc.kind, exc.detail, exc.is_dark_feed)


def _empty_state(early: EarlyPlannerInput, decision_ts: datetime | None = None) -> PlannerState:
    return PlannerState(
        decision_ts,
        (),
        tuple(
            sorted(
                (normalized(symbol), float(qty)) for symbol, qty in early.early_positions.items()
            )
        ),
        0.0,
        None,
        True,
        0.0,
    )


def latest_values(bars: pd.DataFrame) -> tuple[dict[str, datetime], dict[str, float]]:
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
            f"held/consumed symbols have a non-positive / non-finite mark: {unvaluable} - "
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
                f"cannot map {symbol} mark {timestamp} to an exchange session ({exc!r}) - "
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
    bars["symbol"] = bars["symbol"].map(normalized)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        bars[column] = bars[column].astype("float64")
    bars = bars.sort_index(kind="stable")
    if not bars.empty:
        bars = bars[[timestamp.date() < now.date() for timestamp in bars.index]]
    universe_bars = bars[bars["symbol"].isin(normalized_symbols(gate_universe, "gate_universe"))]
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
    return _prepared(early, strategy.execution.warmup_bars)


def _prepared(early: EarlyPlannerInput, warmup_bars: int) -> PreparedEarly:
    closed = closed_universe_bars(early.raw_bars, early.now, early.gate_universe)
    warming = closed.universe_bars.index.nunique() <= warmup_bars
    latest_ts, latest_close = latest_values(closed.bars)
    return PreparedEarly(
        closed.bars, closed.universe_bars, latest_ts, latest_close, closed.decision_ts, warming
    )


def phase_a(strategy: LoadedStrategy, early: EarlyPlannerInput) -> EarlyPlannerResult:
    failure = validate_early(early, strategy)
    if failure is not None:
        return failure
    return early_verdict(early, strategy.execution)


def early_verdict(early: EarlyPlannerInput, execution: ExecutionContract) -> EarlyVerdict:
    """Phase A's outcome for a validated early input, from the input and the execution contract
    alone: the in-process planner and the frozen supervisor both call this."""
    prepared = _prepared(early, execution.warmup_bars)
    held = {normalized(symbol) for symbol, qty in early.early_positions.items() if qty != 0.0}
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
