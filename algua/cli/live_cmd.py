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
from algua.contracts.types import LiveAuthorization
from algua.execution.alpaca_broker import AlpacaLiveBroker, BrokerError
from algua.execution.order_state import (
    client_order_id,
    get_peak_equity,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
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


def _alpaca_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker:
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError(
            "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
            "and ALGUA_ALPACA_LIVE_API_SECRET"
        )
    return AlpacaLiveBroker(authorization, s.alpaca_live_api_key, s.alpaca_live_api_secret,
                            base_url=s.alpaca_live_url)


@live_app.command("trade-tick")
@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)
def trade_tick(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="halt + flatten if equity falls this fraction below the persisted peak"),  # noqa: E501
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
        strategy = load_strategy(name)
        broker = _alpaca_live_broker(authorization)
        provider = _select_provider(False, snapshot)

        def _persist(record: SubmittedOrder) -> None:
            # Audit each accepted LIVE order IMMEDIATELY (before the next submit) so a mid-loop
            # crash still records what hit the real-money venue (#18) — never batch after the loop.
            audit_append(conn, actor="agent", action="live_order",
                         reason=f"{record.side} {record.symbol} {record.order_id}", strategy=name)

        hooks = TickHooks(
            client_order_id_for=client_order_id,
            on_submitted=_persist,
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

    emit(ok({
        "strategy": name,
        "venue": "live",
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "submitted": result.submitted,
        "reconcile_ok": result.reconcile_ok,
    }))
