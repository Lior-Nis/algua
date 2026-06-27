from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from algua.contracts.types import OrderIntent
from algua.execution.alpaca_broker import _AlpacaBroker
from algua.live.paper_loop import decide
from algua.risk.limits import WEIGHT_TOL, RiskBreach, check_drawdown
from algua.strategies.base import LoadedStrategy

_RECONCILE_TOL = 1e-6


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

    # snap.equity is the sizing denominator for the mv/equity weights below: a value that is not a
    # positive finite number ZeroDivisions (== 0), silently flips every current weight's sign (< 0)
    # so decide() trades against inverted holdings, or NaN-poisons every weight to a no-op (NaN, via
    # a bad mark) — refuse before any of those. `not (x > 0.0)` rejects NaN too (NaN > 0 is False),
    # where `x <= 0.0` would let it through. Trip BEFORE drawdown/reconcile/division (#162). The
    # live_snapshot path already fails closed in build_live_sizing_snapshot; this is the guard for
    # the paper-broker path (broker.snapshot) and defense-in-depth for any future sizing source.
    if not (snap.equity > 0.0):
        raise RiskBreach(
            "non_positive_equity",
            f"sizing equity {snap.equity} is not a usable (positive, finite) denominator — "
            f"refusing to trade before it divides by zero, inverts current weights, or NaN-poisons",
        )

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

    # Reconcile the lane-supplied venue belief against the broker's pre-submit snapshot (#249):
    # a drift means an attributed fill and a broker position diverge — halt before compounding it.
    # Reconcile whenever the hook is supplied (venue_belief is not None), INCLUDING when it returns
    # an empty dict: an empty belief against a held broker book is exactly the drift we must catch.
    # Tolerance (_RECONCILE_TOL) absorbs floating-point residuals from fill arithmetic so that
    # sub-nano differences don't trip false positives (#249).
    reconcile_ok = True
    if hooks.venue_belief is not None:
        belief = {s: q for s, q in hooks.venue_belief().items() if q != 0.0}
        all_symbols = set(belief) | set(positions_before)
        drift = [
            s for s in all_symbols
            if abs(belief.get(s, 0.0) - positions_before.get(s, 0.0)) > _RECONCILE_TOL
        ]
        if drift:
            reconcile_ok = False
            raise RiskBreach(
                "reconcile",
                f"venue belief {belief} disagrees with positions_before {positions_before} "
                f"before tick — refusing to trade on inconsistent state",
            )

    # Validate REALIZED gross exposure from the snapshot BEFORE cancelling/submitting (#27): if the
    # broker book is already over the limit, trip/flatten before any NEW order goes out rather than
    # after. The target-weight gross check in decide() can't catch a book that drifted across ticks.
    realized_gross = sum(abs(w) for w in current_weights.values())
    check_gross_exposure_realized(realized_gross, strategy.execution.max_gross_exposure)
    # NOTE (#251): only realized GROSS is re-checked here. The per-symbol concentration cap and
    # short policy are enforced on TARGET weights inside decide()/validate_decision_weights, not on
    # realized positions — so a held name that drifts past max_weight_per_symbol on a realized basis
    # while gross stays in-bounds is NOT tripped here. This is a DELIBERATE deferral ("Realized
    # per-symbol cap in live", in the risk-walls-concentration-cap-design spec under
    # docs/superpowers/specs/), not an oversight; add a realized check here if it becomes a hard
    # live invariant.

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
        if hooks.before_submit is not None:
            hooks.before_submit(intent, coid)
        order_id = broker.submit_sized(intent, snap, coid, reserve=hooks.reserve_buy)
        if order_id in ("noop", "skipped"):
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
