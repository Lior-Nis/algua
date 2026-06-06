from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from algua.execution.alpaca_broker import _AlpacaBroker
from algua.live.paper_loop import decide
from algua.risk.limits import WEIGHT_TOL, RiskBreach, check_drawdown
from algua.strategies.base import LoadedStrategy


def _positions(broker: _AlpacaBroker) -> dict[str, float]:
    """Current broker positions as {symbol: qty} — used only on early-return paths (no decision),
    where no sizing snapshot is taken."""
    return {s: float(q) for s, q in broker.get_positions().items()}


def _early_positions(hooks: TickHooks, broker: _AlpacaBroker) -> dict[str, float]:
    """Return ledger positions from the hook when supplied, else fall back to broker positions.
    Used on early-return paths (no-bars / warmup) so live reports the ledger view, not broker."""
    return hooks.live_positions() if hooks.live_positions is not None else _positions(broker)


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
    # None == hook not supplied (pure decide+submit path, no reconcile). A supplied dict — even an
    # EMPTY one — means "the DB says we hold this"; an empty DB against a held broker book is the
    # drift case we must catch, so reconcile is unconditional once the hook is present (#18).
    derived_positions: dict[str, float] | None = None
    # live_snapshot(bars) -> (SizingSnapshot, nav): supplies the ledger-backed sizing snapshot + NAV
    # (live path). When set, sizing is off the snapshot equity and drawdown off NAV (not account
    # equity). Paper passes None -> broker.snapshot + equity for both (unchanged).
    live_snapshot: Callable[[Any], tuple[Any, float]] | None = None
    # live_positions() -> dict[str, float]: supplies ledger positions for the no-decision early-
    # return paths (empty bars / warmup). Paper passes None -> broker.get_positions().
    live_positions: Callable[[], dict[str, float]] | None = None


class TickHalted(RuntimeError):
    """The kill-switch tripped between cancel and submit; the tick aborted before sending orders."""


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
    now = now or datetime.now(UTC)
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    if not bars.empty:
        # Only decide on fully-closed sessions: drop any bar dated on/after today so a partial
        # current-session bar can't drive the decision. (B2b's scheduler can use the exchange
        # calendar to admit today's bar once its session has closed.)
        cutoff = now.date()
        bars = bars[[ts.date() < cutoff for ts in bars.index]]
    if bars.empty:
        return TickResult(None, {}, _early_positions(hooks, broker), [])

    t = bars.index.max()
    # warmup_bars = N holds the first N closed sessions flat: refuse to decide until strictly MORE
    # than N distinct closed sessions are available, so the FIRST decision happens on session index
    # N (the bar that sees N+1 sessions of history) — identical to the backtest loop's
    # `if i < warmup: continue` and the paper loop's `if bars_seen <= warmup: continue`
    # (#1: reconcile the historical off-by-one, which decided one bar early at nunique() == N).
    if bars.index.nunique() <= strategy.execution.warmup_bars:
        return TickResult(t, {}, _early_positions(hooks, broker), [])  # warm-up not met

    # Snapshot equity + positions ONCE (1 account GET + 1 positions GET); reuse it as the fixed
    # sizing denominator AND as the deterministic position state for the report, reconcile, and the
    # symbol union, so nothing can drift between two network calls mid-tick (#20, #23).
    # When live_snapshot is supplied, it provides a ledger-backed SizingSnapshot (equity =
    # min(allocation, NAV)) and the NAV used as the drawdown basis — so sizing is off the virtual
    # subaccount and drawdown is off NAV (not account equity). Paper passes None -> broker path.
    snap: Any
    if hooks.live_snapshot is not None:
        snap, drawdown_equity = hooks.live_snapshot(bars)
    else:
        snap = broker.snapshot(strategy.universe)
        drawdown_equity = snap.equity

    # Drawdown against the persisted peak BEFORE trading: equity below the breaker threshold halts
    # the tick before any order goes out (#27). The peak ratchets up to this tick's drawdown basis.
    peak = (
        drawdown_equity if hooks.peak_equity is None else max(hooks.peak_equity, drawdown_equity)
    )
    check_drawdown(drawdown_equity, peak, max_drawdown)

    positions_before = {s: q for s, q in snap.qtys.items() if q != 0.0}
    # Realized current weight per held symbol from the SAME snapshot (market_value / equity), so the
    # shared decide() compares targets against what the broker actually holds, not a re-read (#23).
    current_weights = {s: mv / snap.equity for s, mv in snap.market_values.items() if mv != 0.0}

    # Reconcile DB-derived positions against the broker's pre-submit state (#18): a drift means a
    # prior tick's orders never persisted (or vice versa) — halt before compounding it. Reconcile
    # whenever the hook is supplied (derived_positions is not None), INCLUDING when it is empty: an
    # empty DB against a held broker book is exactly the drift we must catch.
    reconcile_ok = True
    if hooks.derived_positions is not None:
        derived = {s: q for s, q in hooks.derived_positions.items() if q != 0.0}
        reconcile_ok = derived == positions_before
        if not reconcile_ok:
            raise RiskBreach(
                "reconcile",
                f"DB-derived positions {hooks.derived_positions} disagree with broker "
                f"{positions_before} before tick — refusing to trade on inconsistent state",
            )

    # Validate REALIZED gross exposure from the snapshot BEFORE cancelling/submitting (#27): if the
    # broker book is already over the limit, trip/flatten before any NEW order goes out rather than
    # after. The target-weight gross check in decide() can't catch a book that drifted across ticks.
    realized_gross = sum(abs(w) for w in current_weights.values())
    check_gross_exposure_realized(realized_gross, strategy.execution.max_gross_exposure)

    weights, intents = decide(strategy, bars.loc[:t], current_weights, t)

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
            if hooks.client_order_id_for is not None else None
        )
        order_id = broker.submit_sized(intent, snap, coid)
        if order_id == "noop":
            continue
        record = SubmittedOrder(symbol=intent.symbol, side=intent.side.value,
                                target_weight=intent.target_weight, order_id=order_id,
                                client_order_id=coid or "", decision_ts=t)
        # Persist IMMEDIATELY (before the next submit) so a mid-loop death never loses this order.
        if hooks.on_submitted is not None:
            hooks.on_submitted(record)
        submitted.append({"symbol": record.symbol, "side": record.side,
                          "target_weight": record.target_weight, "order_id": record.order_id,
                          "client_order_id": record.client_order_id})

    return TickResult(
        decision_ts=t,
        target_weights={s: float(w) for s, w in weights.items()},
        positions_before=positions_before,
        submitted=submitted,
        equity=drawdown_equity,
        peak_equity=peak,
        reconcile_ok=reconcile_ok,
        realized_gross=realized_gross,
    )


def check_gross_exposure_realized(gross: float, max_gross: float) -> None:
    """Gross-exposure check on REALIZED (broker-held) weights rather than targets. Raises the same
    RiskBreach kind family so the CLI trips the kill-switch + flattens exactly as for a target
    breach; the detail names it as realized so the audit trail is unambiguous (#27)."""
    if gross > max_gross + WEIGHT_TOL:
        raise RiskBreach(
            "gross_exposure_realized",
            f"realized gross exposure {gross:.4f} exceeds max_gross_exposure {max_gross:.4f}",
        )
