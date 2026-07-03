from __future__ import annotations

import math

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import breach_payload, ok, registry_conn, resolve_drawdown_breaker, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.contracts.types import LiveAuthorization, ScopedCancelBroker
from algua.execution import live_reconcile
from algua.execution.alpaca_broker import AlpacaLiveBroker, BrokerError
from algua.execution.flatten import flatten_strategy
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_broker_order_id,
    believed_positions,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
    record_live_order,
    recover_stranded_broker_order_ids,
)
from algua.execution.live_reservations import record_reservation
from algua.execution.live_sizing import LiveSizingError, build_live_sizing_snapshot
from algua.execution.order_state import (
    client_order_id,
    get_nav_peak,
    record_tick_snapshot,
    update_nav_peak,
)
from algua.execution.tick_clock import tick_clock
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.observability import (
    CycleCounters,
    configure_logging,
    correlation_context,
    get_logger,
)
from algua.registry import allocations
from algua.registry.allocations import active_allocation
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.live_gate import (
    ALLOWED_SIGNERS_PATH,
    LiveAuthorizationError,
    authorization_active,
    verify_live_authorization,
)
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.risk.book_breaker import BookBreach, BookBreakerLimits, evaluate_book_breaker
from algua.risk.book_equity import update_book_peak
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_tradable_strategy

live_app = typer.Typer(help="LIVE (real-money) trading — human-authorized strategies only",
                       no_args_is_help=True)
app.add_typer(live_app, name="live")

log = get_logger(__name__)


def _live_account_equity() -> float:
    """Read the live account equity (read-only; no go-live authorization needed — not trading)."""
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError("Alpaca LIVE credentials not configured")
    import requests
    from requests import RequestException
    try:
        resp = requests.get(
            f"{s.alpaca_live_url.rstrip('/')}/v2/account",
            # Host is pinned https by the alpaca_live_url settings validator; refuse to chase a
            # redirect so the APCA credential headers can never reach a foreign target (#394). A
            # 3xx then fails the status!=200 check below.
            allow_redirects=False,
            headers={"APCA-API-KEY-ID": s.alpaca_live_api_key,
                     "APCA-API-SECRET-KEY": s.alpaca_live_api_secret},
            timeout=30,
        )
    except RequestException as exc:
        raise ValueError(f"alpaca account equity request failed: {exc}") from exc
    if resp.status_code != 200:
        raise ValueError(f"alpaca {resp.status_code} reading account equity")
    return float(resp.json()["equity"])


@live_app.command("allocate")
@json_errors
def allocate(
    name: str,
    capital: float = typer.Option(..., "--capital", help="live capital base $"),
) -> None:
    """Set a strategy's live capital base (its fixed sizing denominator). Enforces that the sum of
    all live allocations does not exceed account equity."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.DORMANT:
            raise ValueError(
                f"cannot allocate live capital to dormant strategy {name!r}; a recovered "
                "strategy re-allocates only after re-climbing paper -> ... -> live")
        allocations.allocate(conn, rec.id, capital=capital, actor="human",
                             account_equity=_live_account_equity())
    emit(ok({"strategy": name, "capital": capital}))


def _alpaca_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker:
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError(
            "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
            "and ALGUA_ALPACA_LIVE_API_SECRET"
        )
    return AlpacaLiveBroker(authorization, s.alpaca_live_api_key, s.alpaca_live_api_secret,
                            base_url=s.alpaca_live_url)


def _still_live_allocated(conn, name: str) -> bool:
    """True iff `name` is still Stage.LIVE with an active allocation. Re-read at submit time so a
    `live -> dormant` bench committed MID-CYCLE (which atomically revokes the allocation, #247)
    aborts further orders instead of orphaning a position on a now-dormant strategy that run-all —
    iterating only Stage.LIVE — will never flatten (#281). Mirrors the #21 re-read-the-kill-switch-
    before-submit discipline; broader than dormant (any non-LIVE transition mid-cycle halts too)."""
    rec = SqliteStrategyRepository(conn).get(name)
    return rec.stage is Stage.LIVE and active_allocation(conn, rec.id) is not None


def _run_strategy_tick(  # noqa: PLR0913
    conn, name: str, authorization, broker, provider, max_drawdown,
    start: str = "2023-01-01", end: str = "2023-12-31", reserve_buy=None, cancel=None,
) -> dict:
    """Drive ONE strategy's live tick: hooks (incl. the scoped `cancel`), run_tick, breach handling
    (trip + scoped flatten), snapshot persistence. ALWAYS returns a per-strategy result dict — on
    TickHalted/RiskBreach it still performs the side-effects (trip + scoped flatten + audit) and
    returns a breach/halt marker (`{"ok": False, ...}`) instead of emitting+exiting, so run-all can
    surface the already-ticked siblings alongside the breaching strategy in one envelope (#270)."""
    strategy = load_tradable_strategy(name)

    rec = SqliteStrategyRepository(conn).get(name)
    alloc = active_allocation(conn, rec.id)
    if alloc is None:
        raise ValueError(f"{name} has no live allocation")
    allocation = float(alloc["capital"])
    identity = compute_artifact_hashes(name)
    # No buying-power preflight here: min(allocation, NAV) sizing already de-risks toward what the
    # account can fund, and a coarse allocation-vs-BP check would falsely refuse a fully-invested
    # strategy that only rebalances. The proper per-order BP reservation is C2 (codex C1 review).

    def _live_snap(bars):
        return build_live_sizing_snapshot(conn, name, allocation, bars, strategy.universe)

    def _persist(record: SubmittedOrder) -> None:
        # Record the order in the BOOKS immediately (client_order_id is the durable identity): this
        # is what lets fills attribute back to this strategy and lets scoped cancel find this
        # strategy's own open orders. Also audit it so a mid-loop crash still records what hit the
        # real-money venue (#18) — never batch after the loop.
        record_live_order(conn, name, record.symbol, record.side, None, record.client_order_id)
        backfill_broker_order_id(conn, record.client_order_id, record.order_id)
        audit_append(conn, actor="agent", action="live_order",
                     reason=f"{record.side} {record.symbol} {record.order_id}", strategy=name)

    hooks = TickHooks(
        client_order_id_for=client_order_id, on_submitted=_persist, cancel=cancel,
        live_snapshot=_live_snap,
        live_positions=lambda: believed_positions(conn, name, LedgerKind.LIVE),
        should_halt=lambda: (kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn)
                             or not authorization_active(conn, authorization)
                             or not _still_live_allocated(conn, name)),
        peak_equity=get_nav_peak(conn, name),
        reserve_buy=reserve_buy,
    )
    try:
        result = run_tick(strategy, broker, provider, utc(start), utc(end),
                          hooks=hooks, max_drawdown=max_drawdown)
    except TickHalted as exc:
        audit_append(conn, actor="system", action="live_trade_tick_halted",
                     reason=str(exc), strategy=name)
        log.info("tick_halted", extra={"fields": {"strategy": name, "lane": "live"}})
        return breach_payload(str(exc), strategy=name, halted=True)
    except RiskBreach as exc:
        trip_for_breach(conn, name, exc)
        log.error("breach", extra={"fields": {"strategy": name, "lane": "live",
                                              "kind": exc.kind}}, exc_info=True)
        # Scoped cancel (only our orders); ingest fills up to now, then offset every believed
        # position — single-sourced in the execution layer (#336). liquidation_submitted mirrors the
        # prior optimistic semantics: True unless the flatten loop errored.
        res = flatten_strategy(
            conn, broker, name, LedgerKind.LIVE, lane="live",
            cancel=lambda: _scoped_cancel(conn, broker, name),
            ingest=lambda: ingest_activities(
                conn, _broker_account_activities(broker, fill_cursor(conn, LedgerKind.LIVE)),
                LedgerKind.LIVE),
        )
        payload = breach_payload(exc.detail, strategy=name, kind=exc.kind,
                                  liquidation_submitted=res.flatten_error is None)
        if res.flatten_error is not None:
            payload["flatten_error"] = res.flatten_error
        return payload
    except LiveSizingError as exc:        # fail-closed mark -> skip this strategy, don't trade
        audit_append(conn, actor="system", action="live_sizing_skipped",
                     reason=str(exc), strategy=name)
        return {"strategy": name, "skipped": str(exc)}
    if result.peak_equity is not None:
        update_nav_peak(conn, name, result.peak_equity)
        tick_ts, clock_source = tick_clock(broker.clock)
        acct = broker.account()
        record_tick_snapshot(
            conn, name,
            tick_ts=tick_ts,
            decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
            equity=result.equity, peak_equity=result.peak_equity,
            positions=result.positions_before, n_submitted=len(result.submitted),
            reconcile_ok=result.reconcile_ok,
            lane="live", strategy_id=rec.id,
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=identity.dependency_hash,
            account_id=acct.account_id, cash=acct.cash,
            clock_source=clock_source,
        )
    audit_append(conn, actor="agent", action="live_trade_tick",
                 reason=f"{len(result.submitted)} live orders submitted", strategy=name)
    return {
        "strategy": name,
        "venue": "live",
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "submitted": result.submitted,
        "reconcile_ok": result.reconcile_ok,
    }



def _broker_account_activities(broker, after):
    return broker.account_activities(after=after)


def _recover_live_stranded(conn, broker) -> None:
    """#312: backfill broker_order_id onto any crash-stranded NULL live_orders row by asking the
    venue for the order carrying each row's client_order_id (never submits; symbol-verified).
    Audit-only side effects; a broker error propagates via the run-all json_errors handling."""
    outcome = recover_stranded_broker_order_ids(conn, broker, kind=LedgerKind.LIVE)
    if outcome.recovered:
        audit_append(conn, actor="system", action="stranded_order_recovered",
                     reason=f"{len(outcome.recovered)} backfilled: {outcome.recovered}",
                     strategy=None)
    if outcome.mismatched:
        audit_append(conn, actor="system", action="stranded_recovery_mismatch",
                     reason=f"{len(outcome.mismatched)} broker mismatch: {outcome.mismatched}",
                     strategy=None)


def _broker_net_positions(broker) -> dict:
    pos = broker.get_positions()  # pandas Series symbol->qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


def _broker_buying_power(broker) -> float:
    return float(broker.account().buying_power)


def _evaluate_book_loss_breaker(conn, broker):
    """Evaluate the book-level loss/drawdown circuit breaker (#390) for the whole account.

    Reads ONE ``broker.account()`` snapshot (equity + prior-session close). Returns a ``BookBreach``
    to halt+flatten, or None to proceed. Equity is validated BEFORE the high-water mark is ratcheted
    so a non-finite/non-positive read can never corrupt the peak (GATE-1): an unusable equity
    short-circuits to a breach without touching the peak. Otherwise the peak ratchets to include
    this cycle (a fresh all-time high => zero drawdown), and the daily-loss baseline is the broker's
    prior trading-session close (``account.last_equity``).

    A BrokerError reading / parsing the account (missing or malformed equity / last_equity) is
    itself a fail-closed breach: without a trustworthy account snapshot the book is unvaluable, so
    it must engage the persistent halt rather than fall through to a retryable JSON error (GATE-2).
    """
    try:
        account = broker.account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
    except BrokerError as exc:
        return BookBreach(
            "book_account_read_failed",
            f"could not read a trustworthy account snapshot for the book breaker ({exc}) — "
            "refusing to trade the shared book blind",
        )
    limits = BookBreakerLimits(
        max_drawdown=get_settings().book_max_drawdown,
        max_daily_loss=get_settings().book_max_daily_loss,
    )
    if not math.isfinite(equity) or equity <= 0.0:
        # Do NOT ratchet the peak on an unusable read; evaluate_book_breaker returns the
        # book_equity_unusable breach (peak value is irrelevant on this branch).
        return evaluate_book_breaker(equity, 0.0, last_equity, limits)
    peak = update_book_peak(conn, equity)
    return evaluate_book_breaker(equity, peak, last_equity, limits)




@live_app.command("run-all")
@json_errors
def run_all(
    snapshot: str = typer.Option(..., "--snapshot"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float | None = typer.Option(
        None, "--max-drawdown",
        help="per-strategy drawdown breaker fraction; omit to use the conservative default-ON "
             "bound (settings.strategy_max_drawdown_default)",
    ),
    disable_drawdown_breaker: bool = typer.Option(
        False, "--disable-drawdown-breaker",
        help="HUMAN-ONLY emergency: turn the per-strategy drawdown breaker fully OFF (audited)",
    ),
    grace_cycles: int = typer.Option(
        3, "--grace-cycles",
        help="cycles a reconcile mismatch may persist before halting",
    ),
    tolerance: float = typer.Option(1e-6, "--tolerance", help="reconcile share tolerance"),
) -> None:
    """One sequenced portfolio cycle over ALL live strategies: re-verify each, ingest fills,
    reconcile the account against the broker, then tick each (scoped cancel). Trades only when the
    account reconciles clean; a persistent unexplained drift engages the global halt. The book-level
    loss circuit breaker (#390) halts + flattens the WHOLE account on aggregate drawdown / daily
    loss before any strategy can order."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    max_drawdown = resolve_drawdown_breaker(max_drawdown, disable_drawdown_breaker)
    configure_logging()
    counters = CycleCounters()
    # One correlation id per cycle; golden_signals flushes in `finally` so the rollup survives
    # even when the cycle fails before/around the strategy loop (#346).
    with correlation_context():
        log.info("cycle_start", extra={"fields": {"lane": "live", "snapshot": snapshot}})
        try:
            with registry_conn() as conn:
                if disable_drawdown_breaker:
                    audit_append(conn, actor="human", action="drawdown_breaker_disabled",
                                 reason="live run-all invoked with --disable-drawdown-breaker",
                                 strategy=None)
                repo = SqliteStrategyRepository(conn)
                live = repo.list_strategies(Stage.LIVE)
                if not live:
                    emit(ok({"strategies": [], "note": "no live strategies"}))
                    return
                if global_halt.is_engaged(conn):
                    emit(breach_payload("global halt engaged", halted=True))
                    raise typer.Exit(1)
                # re-verify each; skip + flag failures, keep one authorization for the broker
                verified: list[tuple[str, LiveAuthorization]] = []
                skipped: list[dict] = []
                for rec in live:
                    try:
                        verified.append((
                            rec.name,
                            verify_live_authorization(conn, repo, rec.name, ALLOWED_SIGNERS_PATH),
                        ))
                    except LiveAuthorizationError as exc:
                        skipped.append({"strategy": rec.name, "reason": str(exc)})
                if not verified:
                    emit(ok({
                        "strategies": [],
                        "skipped": skipped,
                        "note": "no authorized live strategies",
                    }))
                    return
                broker = _alpaca_live_broker(verified[0][1])
                provider = _select_provider(False, snapshot)
                # ingest fills, then reconcile the account before trading
                cursor = fill_cursor(conn, LedgerKind.LIVE)
                ingest_activities(conn, _broker_account_activities(broker, cursor), LedgerKind.LIVE)
                # #312: resolve any crash-stranded NULL-broker_order_id live row (accepted-but-not-
                # backfilled) BEFORE reconcile, so its now-attributed fill no longer reads as drift.
                _recover_live_stranded(conn, broker)
                cycle = live_reconcile.next_cycle(conn)
                net_positions = _broker_net_positions(broker)
                recon = live_reconcile.reconcile(
                    conn, net_positions, cycle,
                    tolerance=tolerance, grace_cycles=grace_cycles,
                )
                recon_payload = {
                    "cycle": cycle,
                    "clean": recon.clean,
                    "halt": recon.halt,
                    "mismatches": recon.mismatches,
                }
                if recon.halt:
                    counters.reconcile_halted += 1
                    log.error("reconcile_halt",
                              extra={"fields": {"lane": "live", "mismatches": recon.mismatches}})
                    global_halt.engage(
                        conn, reason=f"reconcile drift {recon.mismatches}", actor="system"
                    )
                    emit({"ok": False, "reconcile": recon_payload, "skipped": skipped})
                    raise typer.Exit(1)
                if not recon.clean:
                    counters.reconcile_deferred += 1
                    log.info("reconcile_deferred", extra={"fields": {"lane": "live"}})
                    emit(ok({
                        "reconcile": recon_payload,
                        "skipped": skipped,
                        "note": "reconcile pending; deferring trades this cycle",
                        "strategies": [],
                    }))
                    return
                # BOOK-LEVEL LOSS CIRCUIT BREAKER (#390): before ANY strategy can order, check the
                # WHOLE-account equity against the account high-water mark (drawdown) and the prior
                # trading-session close (daily loss). A breach halts + flattens the ENTIRE account —
                # the per-strategy drawdown breaker can't see a correlated crash across the book.
                # ONE broker.account() snapshot feeds both equity and the daily-loss baseline.
                book_breach = _evaluate_book_loss_breaker(conn, broker)
                if book_breach is not None:
                    counters.breaches += 1
                    # Engage the persistent halt FIRST (fail-safe: no-trade even if the close then
                    # errors), audit, THEN flatten the whole account (cancel-all + close-all —
                    # reaches orphan/dormant/unverified holdings the per-strategy loop never would).
                    global_halt.engage(conn, reason=book_breach.detail, actor="system")
                    audit_append(conn, actor="system", action="book_circuit_breaker",
                                 reason=f"{book_breach.kind}: {book_breach.detail}", strategy=None)
                    log.error("book_circuit_breaker",
                              extra={"fields": {"lane": "live", "kind": book_breach.kind}})
                    payload = {"ok": False, "book_breach": {"kind": book_breach.kind,
                                                            "detail": book_breach.detail},
                               "global_halt": "set", "reconcile": recon_payload,
                               "skipped": skipped}
                    try:
                        broker.close_all_positions()
                    except Exception as exc:  # noqa: BLE001 — surface + persist halt, never swallow
                        counters.flatten_failures += 1
                        log.error("book_flatten_failed",
                                  extra={"fields": {"lane": "live"}}, exc_info=True)
                        emit({**payload, "liquidation_submitted": False, "flatten_error": str(exc)})
                        raise typer.Exit(1) from exc
                    emit({**payload, "liquidation_submitted": True})
                    raise typer.Exit(1)
                # BOOK-LEVEL aggregate risk (#389): the account-scoped aggregate gross / net /
                # single-name-concentration gate is being rewired from the retired per-BUY
                # `BookExposure.permit_buy` accumulator onto the whole-cycle, prefix-safe
                # `evaluate_book` PLAN -> AGGREGATE-GATE -> SETTLE -> APPLY path (design v4). Until
                # that batch wiring lands (a later slice of #389), this cycle applies only the
                # per-cycle buying-power pool below; the pure `evaluate_book` evaluator and its unit
                # coverage are in place (algua/risk/book_limits.py, tests/test_book_limits.py).
                pool = {"available": _broker_buying_power(broker)}

                def _reserve_for(strategy_name):
                    def _reserve(symbol: str, notional: float) -> float:
                        # Buying-power pool trim across strategies (strategy order). Audit any
                        # shortfall vs the intended notional.
                        permitted = min(notional, max(0.0, pool["available"]))
                        pool["available"] -= permitted
                        if permitted < notional:  # trimmed -> audit the shortfall
                            record_reservation(
                                conn, cycle, strategy_name, symbol, notional, permitted
                            )
                        return permitted
                    return _reserve

                results = []
                breached = False
                for name, authorization in verified:
                    result = _run_strategy_tick(
                        conn, name, authorization, broker, provider, max_drawdown,
                        start=start, end=end,
                        reserve_buy=_reserve_for(name),
                        cancel=lambda n=name: _scoped_cancel(conn, broker, n),
                    )
                    results.append(result)
                    counters.ticks += 1
                    if result.get("ok") is False:  # breach/halt marker: stop, keep prior results
                        counters.breaches += 1
                        if result.get("flatten_error") is not None:
                            counters.flatten_failures += 1
                        breached = True
                        break
            envelope = {"reconcile": recon_payload, "skipped": skipped, "strategies": results}
            if breached:
                # A strategy breached/halted (already tripped + scoped-flattened): surface the
                # breaching strategy AND every sibling already ticked this cycle in one envelope,
                # then exit non-zero (#270) — don't discard the prior results.
                emit({"ok": False, **envelope})
                raise typer.Exit(1)
            emit(ok(envelope))
        except typer.Exit:
            raise
        except Exception:
            log.error("cycle_failed", extra={"fields": {"lane": "live"}}, exc_info=True)
            raise
        finally:
            log.info("golden_signals", extra={"fields": counters.as_fields()})


def _scoped_cancel(conn, broker: ScopedCancelBroker, strategy: str) -> None:
    """Cancel only THIS strategy's open orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, strategy):
        broker.cancel_order(oid)
