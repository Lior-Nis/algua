from __future__ import annotations

from dataclasses import replace

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import (
    SYSTEMIC_SETUP_EXCEPTIONS,
    StrategySetupError,
    breach_payload,
    ok,
    resolve_drawdown_breaker,
    resolve_wall_clock_window,
)
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.cli.lane_refresh import build_cycle_plan, lane_symbols, refresh_lane_snapshot
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.contracts.types import LiveAuthorization, ScopedCancelBroker
from algua.evaluation.inputs import (
    select_provider as _select_provider,
)
from algua.execution import live_reconcile
from algua.execution.alpaca_broker import AlpacaLiveBroker
from algua.execution.broker_factory import BrokerKind, build_broker
from algua.execution.flatten import flatten_strategy
from algua.execution.lane_exit import build_live_broker
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
from algua.execution.sizing import MIN_NOTIONAL
from algua.execution.tick_clock import tick_clock
from algua.live.book_exposure import build_book_exposure as _build_book_exposure
from algua.live.live_loop import (
    SubmittedOrder,
    TickHalted,
    TickHooks,
    run_tick,
)
from algua.observability import (
    CycleCounters,
    configure_logging,
    correlation_context,
    get_logger,
)
from algua.primitives.timeparse import utc
from algua.registry import allocations
from algua.registry.allocations import active_allocation
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.db import registry_conn
from algua.registry.live_gate import (
    ALLOWED_SIGNERS_PATH,
    LiveAuthorizationError,
    authorization_active,
    verify_live_authorization,
)
from algua.registry.store import SqliteStrategyRepository
from algua.registry.universe_binding import SOURCE_CONFIG_LEGACY, resolve_operational_universe
from algua.risk import global_halt, kill_switch
from algua.risk.book_cycle import evaluate_book_loss_breaker as _evaluate_book_loss_breaker
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_tradable_strategy

live_app = typer.Typer(help="LIVE (real-money) trading — human-authorized strategies only",
                       no_args_is_help=True)
app.add_typer(live_app, name="live")

log = get_logger(__name__)


def _live_account_equity() -> float:
    """Live account equity (read-only; no go-live authorization needed). Delegates to the
    read-only live broker (bounded retry, stricter `account()` parse) rather than a hand-rolled
    HTTP call; kept as a module-level function so existing monkeypatch pins keep resolving."""
    return build_broker(BrokerKind.ALPACA_LIVE_READONLY).account().equity


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
        # LIVE-gate: refuse any non-live stage BEFORE the network account read (skip a doomed
        # equity fetch); allocate_in_lane re-checks authoritatively under its write lock.
        if rec.stage is not Stage.LIVE:
            raise ValueError(
                f"cannot allocate live capital to {name!r} at stage {rec.stage.value}; live "
                "allocate requires stage 'live' (a dormant/recovered/forward-tested strategy "
                "re-allocates only after reaching live)")
        allocations.allocate_in_lane(
            conn, rec.id, capital=capital, actor="human",
            account_equity=_live_account_equity(),
            allowed_stages=frozenset({Stage.LIVE.value}))
    emit(ok({"strategy": name, "capital": capital}))


def _alpaca_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker:
    # Single-sourced with the book-exit drain (#497) so the two never drift on how the real-money
    # broker is built from the settings credentials.
    return build_live_broker(authorization)


def _still_live_allocated(conn, name: str) -> bool:
    """True iff `name` is still Stage.LIVE with an active allocation. Re-read at submit time so a
    mid-cycle `live -> dormant` bench (#247) halts further orders instead of orphaning a position
    run-all would never flatten (#281); broader than dormant (any non-LIVE transition halts)."""
    rec = SqliteStrategyRepository(conn).get(name)
    return rec.stage is Stage.LIVE and active_allocation(conn, rec.id) is not None


def _run_strategy_tick(  # noqa: PLR0913
    conn, name: str, authorization, broker, provider, max_drawdown,
    start: str, end: str, reserve_buy=None, cancel=None, snapshot_id: str | None = None,
) -> dict:
    """Drive ONE strategy's live tick: hooks (incl. scoped `cancel`), run_tick, breach handling
    (trip + scoped flatten), snapshot persistence. Always returns a per-strategy result dict — on
    TickHalted/RiskBreach it returns a `{"ok": False, ...}` marker (after the side effects) instead
    of emitting+exiting, so run-all can surface ticked siblings alongside the breach (#270).

    Fault-isolation boundary (#374 GATE-2): the SETUP portion (strategy load, allocation lookup,
    identity, hook wiring) runs strictly BEFORE any broker/ledger side effect — a failure there is
    isolated as ``StrategySetupError`` UNLESS it's a :data:`SYSTEMIC_SETUP_EXCEPTIONS` member (a
    shared-infra fault), which propagates raw to abort the whole cycle. Everything from
    ``run_tick`` onward is unwrapped: any exception there is book-integrity-critical and aborts."""
    try:
        strategy = load_tradable_strategy(name)

        rec = SqliteStrategyRepository(conn).get(name)
        alloc = active_allocation(conn, rec.id)
        if alloc is None:
            raise ValueError(f"{name} has no live allocation")
        allocation = float(alloc["capital"])
        identity = compute_artifact_hashes(name)

        # #559/#601: bind this tick to the GATED universe, never CONFIG — the same wall paper
        # enforces (landed in paper first; live lagged until #601). A missing gate row raises
        # LookupError -> the caller's setup handling; a legacy row (universe_name NULL) falls
        # back to CONFIG with a warning. test_lane_parity.py asserts neither lane can lose this.
        resolved_universe, universe_source = resolve_operational_universe(
            conn, get_settings().data_dir, name, strategy.universe)
        if universe_source == SOURCE_CONFIG_LEGACY:
            log.warning("universe_binding_config_legacy", extra={"fields": {
                "strategy": name, "lane": "live",
                "note": "newest passing gate row has no universe_name (pre-#559); ticking on "
                        "CONFIG.universe — re-run research promote to bind the gate universe"}})
        if resolved_universe != strategy.universe:
            strategy = replace(
                strategy,
                config=strategy.config.model_copy(update={"universe": resolved_universe}))

        # No buying-power preflight: min(allocation, NAV) sizing already de-risks; a coarse
        # allocation-vs-BP check would falsely refuse a fully-invested rebalance-only strategy.

        def _live_snap(bars):
            return build_live_sizing_snapshot(conn, name, allocation, bars, strategy.universe)

        def _persist(record: SubmittedOrder) -> None:
            # Record in the BOOKS immediately (client_order_id is the durable identity) so fills
            # attribute back and scoped cancel finds it; audit too (#18) — never batch after loop.
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
    except (KeyboardInterrupt, SystemExit):
        raise
    except StrategySetupError:
        # A nested setup helper that already classified itself: propagate as-is so its original
        # ``code`` survives instead of being double-wrapped (defense-in-depth; unreachable today).
        raise
    except SYSTEMIC_SETUP_EXCEPTIONS:
        # Shared-infra fault (e.g. sqlite3.Error): propagate RAW so run-all aborts the cycle and
        # the top-level db_unavailable/retryable signal survives (#374 GATE-2).
        raise
    except Exception as exc:  # noqa: BLE001 - pre-side-effect setup fault: isolate ONE tenant
        raise StrategySetupError(name, exc) from exc
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
        if exc.is_dark_feed:
            # DARK BAR FEED (#452 HIGH#3): a stale/unvaluable mark means risk state cannot be
            # TRUSTED — flattening blind would dump the book at unknown prices. SYSTEMIC (shared
            # provider), so HALT the whole account and PRESERVE positions; the broker-truth
            # book-loss breaker still catches a real drawdown next cycle.
            global_halt.engage(conn, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="live_mark_freshness_halt",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            return breach_payload(exc.detail, strategy=name, kind=exc.kind, halted=True,
                                  global_halt="set", liquidation_submitted=False)
        # ECONOMIC/integrity breach: trip + scoped flatten (own orders only), offsetting every
        # believed position capped to the actually broker-held qty (Fork B #449, single-sourced
        # in the execution layer #336) so every offset is provably risk-reducing.
        res = flatten_strategy(
            conn, broker, name, LedgerKind.LIVE, lane="live",
            cancel=lambda: _scoped_cancel(conn, broker, name),
            ingest=lambda: ingest_activities(
                conn, _broker_account_activities(broker, fill_cursor(conn, LedgerKind.LIVE)),
                LedgerKind.LIVE),
            held=lambda: _broker_net_positions(broker),
        )
        payload = breach_payload(exc.detail, strategy=name, kind=exc.kind,
                                  liquidation_submitted=res.flatten_error is None)
        if res.flatten_error is not None:
            payload["flatten_error"] = res.flatten_error
        return payload
    except LiveSizingError as exc:
        # Mark-data problems now raise RiskBreach (HALTED above); only a residual non-wall
        # sizing error reaches here and skips this strategy without trading.
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
            clock_source=clock_source, snapshot_id=snapshot_id,
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
    """#312: backfill broker_order_id onto any crash-stranded NULL live_orders row (asks the venue
    by client_order_id; never submits). Audit-only; a broker error propagates via json_errors."""
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


@live_app.command("run-all")
@json_errors
def run_all(
    snapshot: str | None = typer.Option(
        None, "--snapshot", help="tick against this ingested bars snapshot id (explicit/replay)"),
    refresh: bool = typer.Option(
        False, "--refresh",
        help="resolve-or-ingest the lane's bars for the cycle window — the union of every "
             "tickable strategy's gate-bound universe, ledger-held and broker-held symbols — "
             "requiring each symbol's newest bar on the session the tick decides on and each "
             "universe symbol's history floor; the always-on operator path. Exactly one of "
             "--snapshot/--refresh is required; --start/--end are derived (not accepted) under "
             "--refresh (#556)"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
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
    reconcile against the broker, then tick each (scoped cancel) only when reconciled clean — a
    persistent drift engages the global halt. The book-level loss breaker (#390) halts + flattens
    the WHOLE account on aggregate drawdown/daily loss before any strategy can order."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    if bool(snapshot) == refresh:
        raise ValueError("pass exactly one of --snapshot <id> or --refresh")
    if refresh and (start is not None or end is not None):
        raise ValueError("--refresh derives the cycle window (end = today, start from the "
                         "strategies' history need); --start/--end are not accepted")
    max_drawdown = resolve_drawdown_breaker(max_drawdown, disable_drawdown_breaker)
    start, end = resolve_wall_clock_window(start, end)
    configure_logging()
    counters = CycleCounters()
    # One correlation id per cycle; golden_signals flushes in `finally` so the rollup survives
    # even when the cycle fails before/around the strategy loop (#346).
    with correlation_context():
        log.info("cycle_start",
                 extra={"fields": {"lane": "live", "snapshot": snapshot or "refresh"}})
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
                # Inverted capital flow (#497): a strategy enters `live` UNALLOCATED and the human
                # allocates it afterward. SKIP an unallocated one (nothing to size against) rather
                # than crash; partition BEFORE the loop so it never reaches the allocation raise.
                skipped_unallocated = [
                    name for name, _ in verified
                    if active_allocation(conn, repo.get(name).id) is None
                ]
                _unallocated = set(skipped_unallocated)
                verified = [(name, auth) for name, auth in verified
                            if name not in _unallocated]
                if not verified:
                    # Every authorized strategy is unallocated: skip cleanly WITHOUT building the
                    # broker or touching live credentials (mirrors the no-authorized early return
                    # above) — a no-op cycle must not require the real-money broker.
                    emit(ok({
                        "strategies": [],
                        "skipped": skipped,
                        "skipped_unallocated": skipped_unallocated,
                        "note": "no allocated live strategies",
                    }))
                    return
                # At least one strategy remains allocated: build the account-level broker (from a
                # still-allocated strategy's authorization) so the whole-account reconcile + book
                # breakers run over orphan/residual holdings before any strategy orders.
                account_authorization = verified[0][1]
                broker = _alpaca_live_broker(account_authorization)
                # ingest fills, then reconcile the account before trading
                cursor = fill_cursor(conn, LedgerKind.LIVE)
                ingest_activities(conn, _broker_account_activities(broker, cursor), LedgerKind.LIVE)
                # #312: resolve any crash-stranded NULL-broker_order_id live row (accepted-but-not-
                # backfilled) BEFORE reconcile, so its now-attributed fill no longer reads as drift.
                _recover_live_stranded(conn, broker)

                results: list[dict] = []
                # Lane bars refresh (#556): AFTER fill ingest/stranded-recovery, BEFORE the
                # reconcile — mirrors paper (see paper_cmd.run_all for the discovery/reconcile
                # discipline: broker positions are read TWICE on purpose, this first read only
                # feeds symbol DISCOVERY, the reconcile below takes its own fresh read AFTER the
                # network round-trip). A refresh failure fails the WHOLE cycle closed.
                snapshot_info: dict = {"id": snapshot, "refreshed": False,
                                       "start": start, "end": end}
                if refresh:
                    plan = build_cycle_plan(
                        conn, names=[n for n, _a in verified], kind=LedgerKind.LIVE,
                        data_dir=get_settings().data_dir)
                    results.extend(plan.skipped)
                    if verified and not plan.universes:
                        audit_append(conn, actor="system", action="cycle_plan_failed",
                                     reason=f"{len(plan.skipped)} tenant(s) failed planning",
                                     strategy=None)
                        emit({"ok": False, "code": "cycle_plan_failed",
                              "error": "every tickable strategy failed the cycle plan",
                              "strategies": results, "skipped": skipped,
                              "skipped_unallocated": skipped_unallocated})
                        raise typer.Exit(1)
                    verified = [(n, a) for n, a in verified if n in plan.universes]
                    if verified:
                        try:
                            snapshot_info = refresh_lane_snapshot(
                                lane_symbols(plan, _broker_net_positions(broker)), end=end,
                                min_rows=plan.min_rows, kind=LedgerKind.LIVE)
                        except Exception as exc:  # noqa: BLE001 — any refresh fault fails closed
                            audit_append(conn, actor="system", action="bars_refresh_failed",
                                         reason=str(exc), strategy=None)
                            log.error("bars_refresh_failed",
                                      extra={"fields": {"lane": "live"}}, exc_info=True)
                            emit({"ok": False, "code": "refresh_failed", "error": str(exc),
                                  "strategies": results, "skipped": skipped,
                                  "skipped_unallocated": skipped_unallocated})
                            raise typer.Exit(1) from exc
                        snapshot, start = snapshot_info["id"], snapshot_info["start"]
                        log.info("bars_refreshed",
                                 extra={"fields": {"lane": "live", **snapshot_info}})
                # No tickable tenant -> no provider needed; the reconcile below still runs.
                provider = _select_provider(False, snapshot) if verified else None

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
                    emit({"ok": False, "reconcile": recon_payload, "skipped": skipped,
                          "skipped_unallocated": skipped_unallocated, "strategies": results,
                          "snapshot": snapshot_info})
                    raise typer.Exit(1)
                if not recon.clean:
                    counters.reconcile_deferred += 1
                    log.info("reconcile_deferred", extra={"fields": {"lane": "live"}})
                    emit(ok({
                        "reconcile": recon_payload,
                        "skipped": skipped,
                        "skipped_unallocated": skipped_unallocated,
                        "note": "reconcile pending; deferring trades this cycle",
                        "strategies": results,
                        "snapshot": snapshot_info,
                    }))
                    return
                # BOOK-LEVEL LOSS CIRCUIT BREAKER (#390): whole-account equity vs high-water mark
                # (drawdown) and the prior session close (daily loss); a breach halts + flattens
                # the ENTIRE account (per-strategy breakers can't see a correlated book crash).
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
                               "skipped": skipped,
                               "skipped_unallocated": skipped_unallocated,
                               "strategies": results, "snapshot": snapshot_info}
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
                # BOOK-LEVEL aggregate risk (#389): one account-scoped exposure accumulator caps
                # aggregate gross/net/concentration ACROSS strategies. BENIGNLY DEFER on a
                # policy/economic precondition (short position, already-breaching seed book); a
                # DATA-INTEGRITY failure instead raises RiskBreach (#452 HIGH#2) — a dark feed is
                # SYSTEMIC, so HALT and PRESERVE positions (halt-only; the book-loss breaker above
                # already flattens a REAL drawdown independently).
                try:
                    book, book_reason = _build_book_exposure(
                        broker, provider, net_positions, start, end
                    )
                except RiskBreach as book_exc:
                    counters.breaches += 1
                    global_halt.engage(conn, reason=book_exc.detail, actor="system")
                    audit_append(conn, actor="system", action="book_stale_marks_halt",
                                 reason=f"{book_exc.kind}: {book_exc.detail}", strategy=None)
                    log.error("book_stale_marks_halt",
                              extra={"fields": {"lane": "live", "kind": book_exc.kind}})
                    emit({"ok": False,
                          "book_breach": {"kind": book_exc.kind, "detail": book_exc.detail},
                          "global_halt": "set", "liquidation_submitted": False,
                          "reconcile": recon_payload, "skipped": skipped,
                          "skipped_unallocated": skipped_unallocated, "strategies": results,
                          "snapshot": snapshot_info})
                    raise typer.Exit(1) from book_exc
                if book is None:
                    log.info("book_risk_deferred",
                             extra={"fields": {"lane": "live", "reason": book_reason}})
                    emit(ok({
                        "reconcile": recon_payload,
                        "skipped": skipped,
                        "skipped_unallocated": skipped_unallocated,
                        "note": f"book-level risk precondition failed: {book_reason}; "
                                "deferring trades this cycle",
                        "strategies": results,
                        "snapshot": snapshot_info,
                    }))
                    return
                pool = {"available": _broker_buying_power(broker)}

                def _reserve_for(strategy_name):
                    def _reserve(symbol: str, notional: float) -> float:
                        # Pool trims first; book.permit_buy trims to account-level headroom and
                        # MUTATES its running book by the FINAL permitted (siblings see it). Audit
                        # any shortfall vs intended notional.
                        pool_permitted = min(notional, max(0.0, pool["available"]))
                        # min_notional: a sub-minimum book trim is skipped downstream, so the book
                        # does not burn budget for a phantom fill (accounting stays in step).
                        permitted = book.permit_buy(
                            symbol, pool_permitted, min_notional=MIN_NOTIONAL
                        )
                        pool["available"] -= permitted
                        if permitted < notional:  # trimmed/skipped -> audit the shortfall
                            record_reservation(
                                conn, cycle, strategy_name, symbol, notional, permitted
                            )
                        return permitted
                    return _reserve

                breached = False
                for name, authorization in verified:
                    # Per-strategy fault isolation (#374/GATE-2): ONLY a pre-side-effect setup
                    # fault (StrategySetupError) is contained here so the loop continues for
                    # siblings; TickHalted/RiskBreach are already {ok:False} markers. Any OTHER
                    # exception is book-integrity-critical and propagates RAW to abort the cycle.
                    # The raw message is NEVER put in the envelope/audit (only the stable code +
                    # exc_info=True log), to avoid leaking credentials/paths.
                    try:
                        result = _run_strategy_tick(
                            conn, name, authorization, broker, provider, max_drawdown,
                            start=start, end=end,
                            reserve_buy=_reserve_for(name),
                            cancel=lambda n=name: _scoped_cancel(conn, broker, n),
                            snapshot_id=snapshot,
                        )
                    except StrategySetupError as exc:
                        log.error("strategy_setup_error",
                                  extra={"fields": {"lane": "live", "strategy": name,
                                                    "code": exc.code}},
                                  exc_info=True)
                        audit_append(conn, actor="system", action="strategy_setup_error",
                                     reason=exc.code, strategy=name)
                        results.append({"ok": False, "strategy": name, "kind": "setup_error",
                                        "error": exc.code})
                        counters.setup_errors += 1
                        continue
                    results.append(result)
                    counters.ticks += 1
                    if result.get("ok") is False:  # breach/halt marker: stop, keep prior results
                        counters.breaches += 1
                        if result.get("flatten_error") is not None:
                            counters.flatten_failures += 1
                        breached = True
                        break
            envelope = {"reconcile": recon_payload, "skipped": skipped,
                        "skipped_unallocated": skipped_unallocated, "strategies": results,
                        "setup_errors": [r for r in results if r.get("kind") == "setup_error"],
                        "snapshot": snapshot_info}
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


@live_app.command("flatten")
@json_errors
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: flatten THIS strategy's believed positions only and trip its kill-switch.
    Kill-switch trips BEFORE any authorization check (fail-safe). The offset loop reads broker
    net positions ONCE and caps each offset to the actually-held signed qty (Fork B, #449);
    resting orders for this strategy alone are cancelled — siblings are never touched."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching a switch
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)  # LookupError if unknown
        # Require Stage.LIVE before touching the kill-switch (Fork E) — a non-LIVE strategy cannot
        # be emergency-flattened through the live surface.
        if rec.stage is not Stage.LIVE:
            emit({
                "ok": False,
                "strategy": name,
                "error": f"live flatten requires a LIVE strategy; {name} is {rec.stage.value}",
                "kill_switch": "not_tripped",
            })
            raise typer.Exit(1)
        # Trip the kill-switch FIRST, fail-safe, BEFORE any authorization check (Fork A) — future
        # ticks halt even for a revoked/drifted strategy whose broker cannot be built.
        kill_switch.trip(conn, name, reason="flatten", actor=actor_enum.value)
        audit_append(conn, actor=actor_enum.value, action="flatten",
                     reason="manual flatten", strategy=name)
        # Now verify authorization to build the broker. If it fails, the kill-switch is still
        # tripped and the payload reports the emergency escalation path.
        try:
            authorization = verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)
        except LiveAuthorizationError as exc:
            emit({
                "ok": False,
                "strategy": name,
                "kill_switch": "tripped",
                "liquidation_submitted": False,
                "error": str(exc),
                "note": ("future ticks are halted (kill-switch tripped); the open position "
                         "was NOT closed — the live authorization is revoked/absent or the "
                         "artifact drifted. Break-glass: close via the raw broker "
                         "(DELETE /v2/positions?cancel_orders=true with the live keys). "
                         "A signed account-level break-glass path is tracked in #478."),
            })
            raise typer.Exit(1) from exc
        broker = _alpaca_live_broker(authorization)
        # Delegate the offset-liquidation loop to the single-sourced helper, injecting held so
        # every offset is capped to the actually-held signed quantity (Fork B, #449).
        res = flatten_strategy(
            conn, broker, name, LedgerKind.LIVE, lane="live",
            cancel=lambda: _scoped_cancel(conn, broker, name),
            ingest=lambda: ingest_activities(
                conn, _broker_account_activities(broker, fill_cursor(conn, LedgerKind.LIVE)),
                LedgerKind.LIVE),
            held=lambda: _broker_net_positions(broker),
        )
        if res.flatten_error is not None:
            emit(breach_payload(res.flatten_error, strategy=name, liquidation_submitted=False,
                                offsets_submitted=res.n_offsets))
            raise typer.Exit(1)
    # liquidation_submitted reflects whether any offset order ACTUALLY went out (Fork B / GATE-2):
    # a strategy already flat submits none → False. Offset fills land async (may be next open).
    emit(ok({
        "strategy": name,
        "kill_switch": "tripped",
        "liquidation_submitted": res.n_offsets > 0,
        "offsets_submitted": res.n_offsets,
    }))


@live_app.command("halt-all")
@json_errors
def halt_all(
    reason: str = typer.Option(..., "--reason", help="why the whole live account is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: engage the global halt — stop ALL future ticks (paper AND live).

    Does NOT close open positions.
    """
    actor_enum = Actor(actor)
    with registry_conn() as conn:
        global_halt.engage(conn, reason=reason, actor=actor_enum.value)
        audit_append(conn, actor=actor_enum.value, action="halt_all", reason=reason,
                     strategy=None)
    emit(ok({
        "global_halt": "set",
        "liquidation_submitted": False,
        "note": ("global halt engaged — all future ticks (paper AND live) are stopped. "
                 "This command does NOT close open positions. To exit LIVE positions: "
                 "run 'live flatten <name>' per live strategy (each is provably "
                 "risk-reducing). A single-call account-wide close requires the signed "
                 "account-level emergency-liquidation authority tracked in #478; the "
                 "interim account-wide break-glass is the raw broker "
                 "(DELETE /v2/positions?cancel_orders=true with the live keys)."),
    }))


def _scoped_cancel(conn, broker: ScopedCancelBroker, strategy: str) -> None:
    """Cancel only THIS strategy's open orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, strategy):
        broker.cancel_order(oid)
