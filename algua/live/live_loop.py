from __future__ import annotations

import math
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.calendar.market_calendar import MarketCalendar
from algua.contracts.types import OrderIntent
from algua.execution.alpaca_broker import _AlpacaBroker
from algua.live.paper_loop import decide
from algua.risk.limits import (
    MAX_STALE_SESSIONS,
    WEIGHT_TOL,
    RiskBreach,
    check_drawdown,
    check_mark_freshness,
)
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


def _latest_rows(bars: pd.DataFrame) -> pd.DataFrame:
    """Each symbol's LATEST row selected by timestamp with NULLS PRESERVED. The frame is already
    `sort_index()`-ed by the caller, so `groupby('symbol').tail(1)` returns the newest row per
    symbol WITHOUT dropping a NaN close — unlike `groupby(...).last()`, which skips NaN and would
    backfill an older finite close paired with the newest timestamp, masking a NaN-latest mark
    from the `isfinite and > 0` wall (#452 Round-2b). `_latest_bar_ts` and `_latest_marks` both
    read from this same per-symbol row so the timestamp and close come from ONE atomic row."""
    if bars.empty or "symbol" not in bars.columns:
        return bars.iloc[0:0]
    return bars.groupby("symbol", sort=False).tail(1)


def _latest_bar_ts(bars: pd.DataFrame) -> dict[str, datetime]:
    """{symbol: latest kept bar timestamp} from the null-preserving latest-row selection."""
    tail = _latest_rows(bars)
    return {str(sym): ts for ts, sym in zip(tail.index, tail["symbol"], strict=True)}


def _latest_marks(bars: pd.DataFrame) -> dict[str, float]:
    """{symbol: latest kept close} from the null-preserving latest-row selection. A NaN/+inf close
    is PRESERVED (not skipped) so the usability wall can reject it as `unvaluable_marks`."""
    tail = _latest_rows(bars)
    return {str(sym): float(c) for sym, c in zip(tail["symbol"], tail["close"], strict=True)}


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
    `_build_book_exposure` reuses the SAME wall over the account book. Raises
    `RiskBreach('unvaluable_marks' | 'stale_marks')` which the per-lane handlers route to
    HALT-WITHOUT-FLATTEN (a dark bar feed, broker still alive)."""
    unvaluable = sorted(
        s for s in symbols
        if s in latest_close and not (math.isfinite(latest_close[s]) and latest_close[s] > 0.0)
    )
    if unvaluable:
        raise RiskBreach(
            "unvaluable_marks",
            f"held/consumed symbols have a non-positive / non-finite mark: {unvaluable} — "
            f"refusing to value/size the book off an unvaluable feed",
        )
    cal = MarketCalendar()
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
    # Timeframe fail-closed (#452): the freshness math maps a bar by its session DATE and reasons in
    # exchange SESSIONS — daily-bar semantics. Intraday freshness (a minutes/hours bound) is an
    # explicit deferred scope, so the wall refuses any non-daily timeframe with a plain ValueError
    # (NOT a bare `assert`, which `-O` strips) at entry, before the fetch, so it governs the held-
    # book gate too. A timeframe mismatch is a static misconfiguration, not a runtime data failure,
    # so it fails the tick closed without flattening — no trade, no venue call.
    if timeframe != "1d":
        raise ValueError(f"mark-freshness wall supports only 1d bars; got {timeframe!r}")
    now = now or datetime.now(UTC)

    # Discover the held book BEFORE the fetch (ledger/broker positions need no bars), then fetch
    # marks for the UNIVERSE ∪ HELD set — so a held INHERITED / out-of-universe symbol is actually
    # requested and gets a REAL mark, instead of falsely reading as `no_mark` merely because it was
    # never fetched (#452 Round-2c). Valuation reads this union frame; the DECISION reads a
    # universe-only view (below), so the widened fetch never pollutes target_weights() or timing.
    held_qtys = _early_positions(hooks, broker)
    held = {s for s, q in held_qtys.items() if q != 0.0}
    universe_and_held = sorted(set(strategy.universe) | held)
    bars = provider.get_bars(universe_and_held, start, end, timeframe).sort_index()
    if not bars.empty:
        # Only decide on fully-closed sessions: drop any bar dated on/after today so a partial
        # current-session bar can't drive the decision. (B2b's scheduler can use the exchange
        # calendar to admit today's bar once its session has closed.)
        cutoff = now.date()
        bars = bars[[ts.date() < cutoff for ts in bars.index]]

    # Latest mark + timestamp per symbol, read ATOMICALLY from the same (null-preserving) row.
    latest_ts = _latest_bar_ts(bars)
    latest_close = _latest_marks(bars)

    # (1) HELD-book gate — BEFORE the empty-bars / warm-up early returns and BEFORE sizing (#452,
    # findings 1 + CRITICAL). A book that holds ANY position must have a usable mark for every held
    # symbol; a dead / empty / stale feed on a held book must TRIP (RiskBreach), not slip out on a
    # no-op early return. A FLAT book has nothing at risk, so the early returns proceed unchanged.
    if held:
        assert_marks_usable(held, latest_ts, latest_close, now)

    if bars.empty:
        # Nothing fetched at all. If the book were held, the gate above already tripped; so a FLAT
        # book falls through here to the unchanged no-op early return (no marks, nothing to value).
        return TickResult(None, {}, held_qtys, [])

    # Decision TIMING + warm-up derive from the UNIVERSE-restricted history ONLY (#452 Round-2d): a
    # held out-of-universe symbol's longer / independent history must NOT make the universe look
    # warmed or move the decision timestamp. Valuation (below) still reads the full union frame.
    universe_bars = bars[bars["symbol"].isin(strategy.universe)]
    t = universe_bars.index.max() if not universe_bars.empty else None
    # warmup_bars = N holds the first N closed sessions flat: refuse to decide until strictly MORE
    # than N distinct closed sessions are available, so the FIRST decision happens on session index
    # N (the bar that sees N+1 sessions of history) — identical to the backtest loop's
    # `if i < warmup: continue` and the paper loop's `if bars_seen <= warmup: continue`
    # (#1: reconcile the historical off-by-one, which decided one bar early at nunique() == N).
    warming = universe_bars.index.nunique() <= strategy.execution.warmup_bars

    # (2) HELD-book risk VALUATION — runs whenever the book is held, warm-up or not (#452).
    # Warm-up is a DECISION gate, not a risk gate: it may suppress decide()/new orders, but it must
    # NEVER suppress valuation of a book that is already held. So the sizing snapshot + the
    # non-positive-equity guard + check_drawdown + reconcile + realized-gross run whenever
    # `held or not warming`.
    valued = held or not warming
    snap: Any = None
    drawdown_equity = 0.0
    peak: float | None = None
    positions_before: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    reconcile_ok = True
    realized_gross = 0.0
    if valued:
        # Snapshot equity + positions ONCE (1 account GET + 1 positions GET); reuse it as the fixed
        # sizing denominator AND as the deterministic position state for the report, reconcile, and
        # the symbol union, so nothing can drift between two network calls mid-tick (#20, #23). When
        # live_snapshot is supplied, it provides a ledger-backed SizingSnapshot (equity =
        # min(allocation, NAV)) and the NAV used as the drawdown basis. It reads the UNION frame so
        # a held out-of-universe mark is priced. Paper passes None -> broker path.
        if hooks.live_snapshot is not None:
            snap, drawdown_equity = hooks.live_snapshot(bars)
        else:
            snap = broker.snapshot(strategy.universe)
            drawdown_equity = snap.equity

        # snap.equity is the sizing denominator for the mv/equity weights below: a value that is not
        # a positive finite number ZeroDivisions (== 0), silently flips every current weight's sign
        # (< 0) so decide() trades against inverted holdings, or NaN-poisons every weight to a no-op
        # (NaN, via a bad mark) — refuse before any of those. `not (x > 0.0)` rejects NaN too. Trip
        # BEFORE drawdown/reconcile/division (#162). A non-positive NAV off trustworthy marks is a
        # genuine economic wipe, so this routes to the trip + flatten economic-breach handler.
        if not (snap.equity > 0.0):
            raise RiskBreach(
                "non_positive_equity",
                f"sizing equity {snap.equity} is not a usable (positive, finite) denominator — "
                f"refusing to trade before it divides by zero, inverts weights, or NaN-poisons",
            )

        # Drawdown against the persisted peak: equity below the breaker threshold halts the tick
        # before any order (#27). The peak ratchets up to this tick's drawdown basis. This runs for
        # a held-but-warming book too, so an inherited book down past the limit trips even while the
        # strategy is short of warm-up bars.
        peak = (
            drawdown_equity if hooks.peak_equity is None
            else max(hooks.peak_equity, drawdown_equity)
        )
        check_drawdown(drawdown_equity, peak, max_drawdown)

        positions_before = {s: q for s, q in snap.qtys.items() if q != 0.0}
        # Realized current weight per held symbol from the SAME snapshot (market_value / equity), so
        # the shared decide() compares targets against what the broker actually holds (#23).
        current_weights = {
            s: mv / snap.equity for s, mv in snap.market_values.items() if mv != 0.0
        }

        # Reconcile the lane-supplied venue belief against the broker's pre-submit snapshot (#249):
        # a drift means an attributed fill and a broker position diverge — halt before compounding
        # it. Reconcile whenever the hook is supplied (venue_belief is not None), INCLUDING when it
        # returns an empty dict: an empty belief against a held broker book is exactly the drift we
        # must catch. Tolerance (_RECONCILE_TOL) absorbs floating-point residuals from fill
        # arithmetic so sub-nano differences don't trip false positives (#249).
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

        # Validate REALIZED gross exposure from the snapshot BEFORE cancelling/submitting (#27): if
        # the broker book is already over the limit, trip/flatten before any NEW order goes out. The
        # target-weight gross check in decide() can't catch a book that drifted across ticks.
        realized_gross = sum(abs(w) for w in current_weights.values())
        check_gross_exposure_realized(realized_gross, strategy.execution.max_gross_exposure)
        # NOTE (#251): only realized GROSS is re-checked here. The per-symbol concentration cap and
        # short policy are enforced on TARGET weights inside decide()/validate_decision_weights, not
        # on realized positions — so a held name that drifts past max_weight_per_symbol on a
        # realized basis while gross stays in-bounds is NOT tripped here. This is a DELIBERATE
        # deferral ("Realized per-symbol cap in live", in the risk-walls-concentration-cap-design
        # spec), not an oversight; add a realized check here if it becomes a hard live invariant.

    if warming:
        # Warm-up not met: no decide()/submit. A HELD book has already been valued + breaker-checked
        # above, so return its drawdown state (equity/peak/realized_gross/positions_before) to
        # persist the peak and record the tick snapshot — keeping the breaker basis continuous
        # across warm-up (#452 CRITICAL). Only a FLAT warming book takes the unchanged no-op return.
        if valued:
            return TickResult(t, {}, positions_before, [], equity=drawdown_equity,
                              peak_equity=peak, reconcile_ok=reconcile_ok,
                              realized_gross=realized_gross)
        return TickResult(t, {}, held_qtys, [])  # FLAT + warming: unchanged no-op

    # (3) CONSUMED-set gate — DECISION path only (past warm-up), after the sizing snapshot, before
    # decide()/submit (#452). The consumed set widens to the valued book ∪ decision universe; the
    # same usability wall runs over all of it so no stale/absent-priced ranked target or order
    # reaches the venue. Past warm-up `valued` is always True (not warming), so snap is defined.
    assert snap is not None  # valued is True on the decision path (not warming)
    assert t is not None  # past warm-up universe_bars is non-empty, so t is a real timestamp
    consumed = {s for s, q in snap.qtys.items() if q != 0.0} | set(strategy.universe)
    assert_marks_usable(consumed, latest_ts, latest_close, now)

    # DECISION view is the UNIVERSE-only history up to t — the widened fetch (held out-of-universe
    # marks) NEVER reaches target_weights(); decide() sees exactly the universe bars it saw before.
    weights, intents = decide(strategy, universe_bars.loc[:t], current_weights, t)

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
            # No order reached the venue: let the lane retract the phantom before_submit row (#311).
            if hooks.on_noop is not None:
                hooks.on_noop(intent, coid)
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
