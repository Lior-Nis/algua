from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.cli.paper_cmd import _breach_payload, _tick_clock, _trip
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.contracts.types import LiveAuthorization
from algua.execution import live_reconcile
from algua.execution.alpaca_broker import AlpacaLiveBroker, BrokerError
from algua.execution.live_ledger import (
    backfill_broker_order_id,
    believed_positions,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
    record_live_order,
)
from algua.execution.live_reservations import record_reservation
from algua.execution.live_sizing import LiveSizingError, build_live_sizing_snapshot
from algua.execution.order_state import (
    client_order_id,
    get_nav_peak,
    record_tick_snapshot,
    update_nav_peak,
)
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.registry import allocations
from algua.registry.allocations import AllocationError, active_allocation
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.live_gate import (
    ALLOWED_SIGNERS_PATH,
    LiveAuthorizationError,
    authorization_active,
    verify_live_authorization,
)
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_strategy

live_app = typer.Typer(help="LIVE (real-money) trading — human-authorized strategies only",
                       no_args_is_help=True)
app.add_typer(live_app, name="live")


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
@json_errors(ValueError, LookupError, AllocationError)
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


def _run_strategy_tick(  # noqa: PLR0913
    conn, name: str, authorization, broker, provider, max_drawdown,
    start: str = "2023-01-01", end: str = "2023-12-31", reserve_buy=None, cancel=None,
) -> dict:
    """Drive ONE strategy's live tick: hooks (incl. the scoped `cancel`), run_tick, breach handling
    (trip + scoped flatten), snapshot persistence. Returns a result dict; raises typer.Exit(1) with
    an emitted breach/halt payload on TickHalted/RiskBreach (same behaviour as the single-strategy
    command)."""
    strategy = load_strategy(name)
    from algua.strategies.base import (
        assert_tradable_without_fundamentals,
        assert_tradable_without_news,
    )
    assert_tradable_without_fundamentals(strategy)
    assert_tradable_without_news(strategy)

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
        live_snapshot=_live_snap, live_positions=lambda: believed_positions(conn, name),
        should_halt=lambda: (kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn)
                             or not authorization_active(conn, authorization)),
        peak_equity=get_nav_peak(conn, name),
        reserve_buy=reserve_buy,
    )
    try:
        result = run_tick(strategy, broker, provider, utc(start), utc(end),
                          hooks=hooks, max_drawdown=max_drawdown)
    except TickHalted as exc:
        audit_append(conn, actor="system", action="live_trade_tick_halted",
                     reason=str(exc), strategy=name)
        emit(_breach_payload(str(exc), strategy=name, halted=True))
        raise typer.Exit(1) from exc
    except RiskBreach as exc:
        _trip(conn, name, exc)
        liquidation_submitted = True
        flatten_error = None
        try:
            _scoped_cancel(conn, broker, name)                       # cancel only our orders
            ingest_activities(conn, _broker_account_activities(broker, fill_cursor(conn)))
            for sym, qty in believed_positions(conn, name).items():  # fresh believed qty
                # RECORD the offset in the books (+ backfill) so its fill attributes back to this
                # strategy and believed_positions drops to flat — else the resume gate would block
                # resume forever (codex CRITICAL). The kill-switch (just tripped) prevents a re-run
                # from re-offsetting, so the per-attempt coid is safe.
                coid = client_order_id(name, datetime.now(UTC), sym)
                record_live_order(conn, name, sym, "sell" if qty > 0 else "buy", None, coid)
                oid = broker.submit_offset(sym, qty, coid)
                backfill_broker_order_id(conn, coid, oid)
        except BrokerError as fexc:
            liquidation_submitted = False
            flatten_error = str(fexc)
            audit_append(conn, actor="system", action="flatten_failed",
                         reason=str(fexc), strategy=name)
        payload = _breach_payload(exc.detail, kind=exc.kind,
                                  liquidation_submitted=liquidation_submitted)
        if flatten_error is not None:
            payload["flatten_error"] = flatten_error
        emit(payload)
        raise typer.Exit(1) from exc
    except LiveSizingError as exc:        # fail-closed mark -> skip this strategy, don't trade
        audit_append(conn, actor="system", action="live_sizing_skipped",
                     reason=str(exc), strategy=name)
        return {"strategy": name, "skipped": str(exc)}
    if result.peak_equity is not None:
        update_nav_peak(conn, name, result.peak_equity)
        tick_ts, clock_source = _tick_clock(broker.clock)
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


def _broker_net_positions(broker) -> dict:
    pos = broker.get_positions()  # pandas Series symbol->qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


def _broker_buying_power(broker) -> float:
    return float(broker.account().buying_power)


@live_app.command("run-all")
@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)
def run_all(
    snapshot: str = typer.Option(..., "--snapshot"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown"),
    grace_cycles: int = typer.Option(
        3, "--grace-cycles",
        help="cycles a reconcile mismatch may persist before halting",
    ),
    tolerance: float = typer.Option(1e-6, "--tolerance", help="reconcile share tolerance"),
) -> None:
    """One sequenced portfolio cycle over ALL live strategies: re-verify each, ingest fills,
    reconcile the account against the broker, then tick each (scoped cancel). Trades only when the
    account reconciles clean; a persistent unexplained drift engages the global halt."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        live = repo.list_strategies(Stage.LIVE)
        if not live:
            emit(ok({"strategies": [], "note": "no live strategies"}))
            return
        if global_halt.is_engaged(conn):
            emit(_breach_payload("global halt engaged", halted=True))
            raise typer.Exit(1)
        # re-verify each; skip + flag failures, keep one authorization for the account broker
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
        ingest_activities(conn, _broker_account_activities(broker, fill_cursor(conn)))
        cycle = live_reconcile.next_cycle(conn)
        recon = live_reconcile.reconcile(
            conn, _broker_net_positions(broker), cycle,
            tolerance=tolerance, grace_cycles=grace_cycles,
        )
        recon_payload = {
            "cycle": cycle,
            "clean": recon.clean,
            "halt": recon.halt,
            "mismatches": recon.mismatches,
        }
        if recon.halt:
            global_halt.engage(
                conn, reason=f"reconcile drift {recon.mismatches}", actor="system"
            )
            emit({"ok": False, "reconcile": recon_payload, "skipped": skipped})
            raise typer.Exit(1)
        if not recon.clean:
            emit(ok({
                "reconcile": recon_payload,
                "skipped": skipped,
                "note": "reconcile pending; deferring trades this cycle",
                "strategies": [],
            }))
            return
        pool = {"available": _broker_buying_power(broker)}

        def _reserve_for(strategy_name):
            def _reserve(symbol: str, notional: float) -> float:
                permitted = min(notional, max(0.0, pool["available"]))
                pool["available"] -= permitted
                if permitted < notional:  # trimmed or fully skipped -> audit the shortfall
                    record_reservation(
                        conn, cycle, strategy_name, symbol, notional, permitted
                    )
                return permitted
            return _reserve

        results = []
        for name, authorization in verified:
            results.append(_run_strategy_tick(
                conn, name, authorization, broker, provider, max_drawdown,
                start=start, end=end,
                reserve_buy=_reserve_for(name),
                cancel=lambda n=name: _scoped_cancel(conn, broker, n),
            ))
    emit(ok({"reconcile": recon_payload, "skipped": skipped, "strategies": results}))


def _scoped_cancel(conn, broker, strategy: str) -> None:
    """Cancel only THIS strategy's open orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, strategy):
        broker.cancel_order(oid)
