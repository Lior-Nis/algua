from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.cli.paper_cmd import _breach_payload, _trip
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.contracts.types import LiveAuthorization
from algua.execution import live_reconcile
from algua.execution.alpaca_broker import AlpacaLiveBroker, BrokerError
from algua.execution.live_ledger import (
    backfill_broker_order_id,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
    record_live_order,
)
from algua.execution.order_state import (
    client_order_id,
    get_peak_equity,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.registry import allocations
from algua.registry.allocations import AllocationError
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
        sid = SqliteStrategyRepository(conn).get(name).id
        allocations.allocate(conn, sid, capital=capital, actor="human",
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
    start: str = "2023-01-01", end: str = "2023-12-31", cancel=None,
) -> dict:
    """Drive ONE strategy's live tick: hooks (incl. the scoped `cancel`), run_tick, breach handling
    (trip + scoped flatten), snapshot persistence. Returns a result dict; raises typer.Exit(1) with
    an emitted breach/halt payload on TickHalted/RiskBreach (same behaviour as the single-strategy
    command)."""
    strategy = load_strategy(name)

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
        client_order_id_for=client_order_id,
        on_submitted=_persist,
        cancel=cancel,
        should_halt=lambda: (kill_switch.is_tripped(conn, name)
                             or global_halt.is_engaged(conn)
                             or not authorization_active(conn, authorization)),
        peak_equity=get_peak_equity(conn, name),
        derived_positions=None,  # Alpaca is the sole source of truth; no local-ledger reconcile
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
            broker.cancel_open_orders()
            broker.close_positions(strategy.universe)
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
    if result.peak_equity is not None:
        update_peak_equity(conn, name, result.peak_equity)
        record_tick_snapshot(
            conn, name, tick_ts=datetime.now(UTC).isoformat(),
            decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
            equity=result.equity, peak_equity=result.peak_equity,
            positions=result.positions_before, n_submitted=len(result.submitted),
            reconcile_ok=result.reconcile_ok,
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


@live_app.command("trade-tick")
@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)
def trade_tick(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="halt + flatten if equity falls this fraction "
                                            "below the persisted peak"),
) -> None:
    """Run ONE wall-clock tick against the Alpaca LIVE venue (REAL MONEY). Re-verifies the human
    go-live signature against the trust anchor before trading; Alpaca is the source of truth (no
    local ledger). A drawdown/exposure breach trips the kill-switch and scoped-flattens."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        # THE WALL: re-verify the human signature for the current artifact (requires Stage.LIVE).
        authorization = verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)
        if kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn):
            raise ValueError(f"{name} is halted; resume before live trading")
        broker = _alpaca_live_broker(authorization)
        provider = _select_provider(False, snapshot)
        result = _run_strategy_tick(
            conn, name, authorization, broker, provider, max_drawdown,
            start=start, end=end,
        )
    emit(ok(result))


def _broker_account_activities(broker, after):
    return broker.account_activities(after=after)


def _broker_net_positions(broker) -> dict:
    pos = broker.get_positions()  # pandas Series symbol->qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


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
        results = []
        for name, authorization in verified:
            results.append(_run_strategy_tick(
                conn, name, authorization, broker, provider, max_drawdown,
                start=start, end=end,
                cancel=lambda n=name: _scoped_cancel(conn, broker, n),
            ))
    emit(ok({"reconcile": recon_payload, "skipped": skipped, "strategies": results}))


def _scoped_cancel(conn, broker, strategy: str) -> None:
    """Cancel only THIS strategy's open orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, strategy):
        broker.cancel_order(oid)
