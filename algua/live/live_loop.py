from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from algua.calendar.factory import get_calendar
from algua.contracts.types import OrderIntent
from algua.execution.alpaca_broker import _AlpacaBroker
from algua.live import planner as decision_planner
from algua.live.planner import InProcessPlanner, PlannerPort
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    Decision,
    EarlyNoDecision,
    EarlyPlannerInput,
    LateNoDecision,
    LatePlannerInput,
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.risk.limits import (
    MAX_STALE_SESSIONS,
    RiskBreach,
    check_mark_freshness,
)
from algua.strategies.base import LoadedStrategy

_RECONCILE_TOL = 1e-6
decide = decision_planner.decide


def _positions(broker: _AlpacaBroker) -> dict[str, float]:
    """Current broker positions as {symbol: qty} — used only on early-return paths (no decision),
    where no sizing snapshot is taken."""
    return {s: float(q) for s, q in broker.get_positions().items()}


def _early_positions(hooks: TickHooks, broker: _AlpacaBroker) -> dict[str, float]:
    """Return ledger positions from the hook when supplied, else fall back to broker positions.
    Used on early-return paths (no-bars / warmup) so live reports the ledger view, not broker."""
    return hooks.live_positions() if hooks.live_positions is not None else _positions(broker)


def _latest_bar_ts(bars) -> dict[str, datetime]:
    tail = bars.iloc[0:0] if bars.empty else bars.groupby("symbol", sort=False).tail(1)
    return {str(sym): ts for ts, sym in zip(tail.index, tail["symbol"], strict=True)}


def _latest_marks(bars) -> dict[str, float]:
    tail = bars.iloc[0:0] if bars.empty else bars.groupby("symbol", sort=False).tail(1)
    return {
        str(sym): float(close)
        for sym, close in zip(tail["symbol"], tail["close"], strict=True)
    }


def assert_marks_usable(
    symbols: Collection[str],
    latest_ts: dict[str, datetime],
    latest_close: dict[str, float],
    now: datetime,
) -> None:
    """Fail closed (RiskBreach) if ANY consumed mark is absent (`no_mark`), stale (> MAX_STALE_
    SESSIONS completed sessions), unvaluable (latest close not a positive finite number — rejects
    <= 0 AND +inf / NaN), or future-dated (bar maps to a session after `now`). Establishing NAV /
    drawdown / gross exposure / the sizing denominator off any such mark is impossible, so the risk
    state cannot be trusted (#452). Exported (no leading underscore) so #389's
    `build_book_exposure` reuses the SAME wall over the account book. Raises
    `RiskBreach('unvaluable_marks' | 'stale_marks')` which the per-lane handlers route to
    HALT-WITHOUT-FLATTEN (a dark bar feed, broker still alive)."""
    unvaluable = sorted(
        s
        for s in symbols
        if s in latest_close and not (math.isfinite(latest_close[s]) and latest_close[s] > 0.0)
    )
    if unvaluable:
        raise RiskBreach(
            "unvaluable_marks",
            f"held/consumed symbols have a non-positive / non-finite mark: {unvaluable} — "
            f"refusing to value/size the book off an unvaluable feed",
        )
    cal = get_calendar()
    stale_by_symbol: dict[str, float] = {}
    for s in symbols:
        ts = latest_ts.get(s)
        if ts is None:
            stale_by_symbol[s] = math.inf  # no bar at all -> no_mark offender (finding 3)
            continue
        try:
            stale_by_symbol[s] = float(cal.sessions_stale(ts, now))
        except Exception as exc:  # MinuteOutOfBounds / unmappable ts (finding 5)
            raise RiskBreach(
                "stale_marks",
                f"cannot map {s} mark {ts} to an exchange session ({exc!r}) — "
                f"refusing to establish risk state off an unmappable timestamp",
            ) from exc
    check_mark_freshness(stale_by_symbol, MAX_STALE_SESSIONS)


@dataclass
class SubmittedOrder:
    symbol: str
    side: str
    target_weight: float
    order_id: str
    client_order_id: str
    decision_ts: datetime


@dataclass
class TickResult:
    decision_ts: datetime | None
    target_weights: dict[str, float]
    positions_before: dict[str, float]
    submitted: list[dict[str, Any]]
    equity: float = 0.0
    peak_equity: float | None = None
    reconcile_ok: bool = True
    realized_gross: float = 0.0


@dataclass(frozen=True)
class PlannerContext:
    """Supervisor-verified identity values copied into one planner request."""

    deployment_id: int | None
    artifact_id: int | None
    manifest_digest: str | None
    config_hash: str
    resolved_config_json: str
    calendar_code: str


def planner_context_for_deployment(deployment: Any, calendar_code: str) -> PlannerContext | None:
    """Copy an already-verified deployment record into the authority-free planner envelope."""
    if deployment is None:
        return None
    return PlannerContext(
        deployment.id,
        deployment.artifact_id,
        deployment.manifest_digest,
        deployment.config_hash,
        deployment.resolved_config_json,
        calendar_code,
    )


@dataclass
class TickHooks:
    """Side-effecting callbacks the orchestrator (the CLI) supplies so the loop itself stays free
    of DB and kill-switch wiring. All are optional; with none supplied the loop is a pure decide +
    submit pass over the injected broker.

    - `client_order_id_for(strategy, decision_ts, symbol) -> str`: the deterministic id sent to
      Alpaca so a retried/re-run submit is idempotent (#18, #24).
    - `on_submitted(SubmittedOrder)`: persist ONE accepted order immediately, so a mid-loop death
      can't leave Alpaca with an order the DB never recorded (#18).
    - `should_halt() -> bool`: re-checked right before the submit phase so an externally-tripped
      kill-switch aborts BEFORE any order is sent (#21).
    - `cancel() -> None`: how to cancel stale open orders before the submit phase. Defaults to the
      broker's ACCOUNT-WIDE cancel (paper); the live multi-strategy loop supplies a SCOPED cancel so
      a strategy never cancels a sibling's orders.
    - `peak_equity`: the persisted per-strategy peak (drawdown denominator across ticks, #27).
    """

    client_order_id_for: Callable[[str, datetime, str], str] | None = None
    on_submitted: Callable[[SubmittedOrder], None] | None = None
    should_halt: Callable[[], bool] | None = None
    cancel: Callable[[], None] | None = None
    peak_equity: float | None = None
    # lane-supplied per-strategy belief (paper_venue_fills); reconciled vs positions_before with
    # tolerance. None = no reconcile (live/sim).
    venue_belief: Callable[[], dict[str, float]] | None = None
    # live_snapshot(bars) -> (SizingSnapshot, nav): supplies the ledger-backed sizing snapshot + NAV
    # (live path). When set, sizing is off the snapshot equity and drawdown off NAV (not account
    # equity). Paper passes None -> broker.snapshot + equity for both (unchanged).
    live_snapshot: Callable[[Any], tuple[Any, float]] | None = None
    # live_positions() -> dict[str, float]: supplies ledger positions for the no-decision early-
    # return paths (empty bars / warmup). Paper passes None -> broker.get_positions().
    live_positions: Callable[[], dict[str, float]] | None = None
    # reserve_buy(symbol, notional) -> permitted_notional: the loop's buying-power reservation hook;
    # caps a BUY's notional to the shared per-cycle pool, returning 0 to skip the order entirely.
    # Sells are never consulted. None == no reservation (paper and any non-reserved path).
    reserve_buy: Callable[[str, float], float] | None = None
    # before_submit(intent, coid): fires IMMEDIATELY BEFORE broker.submit_sized for each intent so
    # the paper lane can record order intent in a crash-safe ledger before the broker call (#249).
    # Live/sim callers that do not supply this hook are unaffected (None -> skipped).
    before_submit: Callable[[OrderIntent, str | None], None] | None = None
    # on_noop(intent, coid): fires when submit_sized reports 'noop'/'skipped' (no order reached the
    # venue — both sentinels return before the POST) AFTER before_submit already recorded a durable
    # intent, so the paper lane can retract that phantom intent row (#311). None -> skipped.
    on_noop: Callable[[OrderIntent, str | None], None] | None = None
    planner_context: PlannerContext | None = None
    # The planner port every planner call goes through (Story 1.3c: a frozen deployment's
    # dispatcher). None -> InProcessPlanner(strategy), today's in-process facade calls.
    planner: PlannerPort | None = None


class TickHalted(RuntimeError):
    """The kill-switch tripped between cancel and submit; the tick aborted before sending orders."""


def _default_planner_context(strategy: LoadedStrategy) -> PlannerContext:
    """Compatibility identity for direct/legacy callers without a deployment record."""
    if hasattr(strategy, "config"):
        from algua.strategies.base import config_hash

        resolved = strategy.config.model_dump(mode="json")
        digest = config_hash(strategy)
    else:
        resolved = {"name": strategy.name, "universe": list(strategy.universe)}
        encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()[:32]
    resolved_json = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PlannerContext(None, None, None, digest, resolved_json, get_calendar().code)


def _tick_result(state, *, submitted=None) -> TickResult:
    return TickResult(
        decision_ts=state.decision_ts,
        target_weights=dict(state.target_weights),
        positions_before=dict(state.positions_before),
        submitted=[] if submitted is None else submitted,
        equity=state.equity,
        peak_equity=state.peak_equity,
        reconcile_ok=state.reconcile_ok,
        realized_gross=state.realized_gross,
    )


def _raise_planner_failure(result: object) -> None:
    if isinstance(result, PlannerRiskFailure):
        raise RiskBreach(result.kind, result.detail)
    if isinstance(result, (PlannerInputFailure, PhaseBindingFailure)):
        raise ValueError(result.detail)


def run_tick(
    strategy: LoadedStrategy,
    broker: _AlpacaBroker,
    provider: Any,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    now: datetime | None = None,
    hooks: TickHooks | None = None,
    max_drawdown: float | None = None,
) -> TickResult:
    """One wall-clock tick: decide on the latest closed session, submit market-order deltas to
    Alpaca (the source of truth). Pure over the injected broker + provider (`now` injected for
    testability); side effects (persistence, kill-switch checks) flow through `hooks`."""
    hooks = hooks or TickHooks()
    # Timeframe fail-closed (#452): the freshness math maps a bar by its session DATE and reasons in
    # exchange SESSIONS — daily-bar semantics. Intraday freshness (a minutes/hours bound) is an
    # explicit deferred scope, so the wall refuses any non-daily timeframe with a plain ValueError
    # (NOT a bare `assert`, which `-O` strips) at entry, before the fetch, so it governs the held-
    # book gate too. A timeframe mismatch is a static misconfiguration, not a runtime data failure,
    # so it fails the tick closed without flattening — no trade, no venue call.
    if timeframe != "1d":
        raise ValueError(f"mark-freshness wall supports only 1d bars; got {timeframe!r}")
    now = now or datetime.now(UTC)

    held_qtys = _early_positions(hooks, broker)
    held = {s for s, q in held_qtys.items() if q != 0.0}
    universe_and_held = sorted(set(strategy.universe) | held)
    bars = provider.get_bars(universe_and_held, start, end, timeframe)
    context = hooks.planner_context or _default_planner_context(strategy)
    early = EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id=secrets.token_hex(16),
        strategy_name=strategy.name,
        deployment_id=context.deployment_id,
        artifact_id=context.artifact_id,
        manifest_digest=context.manifest_digest,
        config_hash=context.config_hash,
        resolved_config_json=context.resolved_config_json,
        now=now,
        timeframe=timeframe,
        calendar_code=context.calendar_code,
        raw_bars=bars,
        early_positions=dict(held_qtys),
        gate_universe=tuple(strategy.universe),
        max_drawdown=max_drawdown,
    )
    planner = hooks.planner if hooks.planner is not None else InProcessPlanner(strategy)
    first = planner.phase_a(early)
    _raise_planner_failure(first)
    if isinstance(first, EarlyNoDecision):
        return _tick_result(first.state)
    if not isinstance(first, SnapshotRequired):
        raise RuntimeError("planner returned an unknown Phase A result")

    if hooks.live_snapshot is not None:
        snap, drawdown_equity = hooks.live_snapshot(planner.closed_bars(early))
    else:
        snap = broker.snapshot(strategy.universe)
        drawdown_equity = snap.equity
    captured = CapturedStrategyState(
        request_id=early.request_id,
        sizing_equity=float(snap.equity),
        drawdown_equity=float(drawdown_equity),
        quantities=dict(snap.qtys),
        market_values=dict(snap.market_values),
        persisted_peak_equity=hooks.peak_equity,
        venue_belief=VenueBeliefPending(),
    )
    late = LatePlannerInput(early, first.phase_a_binding, captured)
    second = planner.phase_b(late)
    _raise_planner_failure(second)
    if not isinstance(second, VenueBeliefRequired):
        raise RuntimeError("planner did not request venue belief after late risk validation")
    venue_belief = (
        VenueBeliefDisabled()
        if hooks.venue_belief is None
        else VenueBeliefEnabled(dict(hooks.venue_belief()))
    )
    second = planner.phase_b(replace(late, captured=replace(captured, venue_belief=venue_belief)))
    _raise_planner_failure(second)
    if isinstance(second, LateNoDecision):
        return _tick_result(second.state)
    if not isinstance(second, Decision):
        raise RuntimeError("planner returned an unknown Phase B result")
    t = second.state.decision_ts
    if t is None:
        raise RuntimeError("decision result has no decision timestamp")
    weights = dict(second.state.target_weights)
    intents = list(second.ordered_intents)

    if hooks.should_halt is not None and hooks.should_halt():
        raise TickHalted("kill-switch tripped before submit phase")

    (hooks.cancel or broker.cancel_open_orders)()

    # Re-check the kill-switch AFTER cancel and immediately before the submit loop (#21): if the
    # switch tripped while cancellation was in flight, abort before sending any order.
    if hooks.should_halt is not None and hooks.should_halt():
        raise TickHalted("kill-switch tripped before submit phase")

    submitted: list[dict[str, Any]] = []
    for intent in intents:
        # Re-check before EACH order so a halt / authorization-revoke mid-loop stops further orders.
        if hooks.should_halt is not None and hooks.should_halt():
            raise TickHalted("kill-switch tripped during submit phase")
        coid = (
            hooks.client_order_id_for(strategy.name, t, intent.symbol)
            if hooks.client_order_id_for is not None
            else None
        )
        if hooks.before_submit is not None:
            hooks.before_submit(intent, coid)
        order_id = broker.submit_sized(intent, snap, coid, reserve=hooks.reserve_buy)
        if order_id in ("noop", "skipped"):
            # No order reached the venue: let the lane retract the phantom before_submit row (#311).
            if hooks.on_noop is not None:
                hooks.on_noop(intent, coid)
            continue
        record = SubmittedOrder(
            symbol=intent.symbol,
            side=intent.side.value,
            target_weight=intent.target_weight,
            order_id=order_id,
            client_order_id=coid or "",
            decision_ts=t,
        )
        # Persist IMMEDIATELY (before the next submit) so a mid-loop death never loses this order.
        if hooks.on_submitted is not None:
            hooks.on_submitted(record)
        submitted.append(
            {
                "symbol": record.symbol,
                "side": record.side,
                "target_weight": record.target_weight,
                "order_id": record.order_id,
                "client_order_id": record.client_order_id,
            }
        )

    return TickResult(
        decision_ts=t,
        target_weights={s: float(w) for s, w in weights.items()},
        positions_before=dict(second.state.positions_before),
        submitted=submitted,
        equity=second.state.equity,
        peak_equity=second.state.peak_equity,
        reconcile_ok=second.state.reconcile_ok,
        realized_gross=second.state.realized_gross,
    )
