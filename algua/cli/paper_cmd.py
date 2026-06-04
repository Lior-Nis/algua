from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime

import typer

from algua.audit.log import append as audit_append
from algua.backtest.engine import BacktestError
from algua.cli.app import app, emit
from algua.cli.backtest_cmd import _select_provider, _utc
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.alpaca_broker import AlpacaPaperBroker, BrokerError
from algua.execution.order_state import derive_positions, persist_run
from algua.execution.sim_broker import SimBroker
from algua.live.live_loop import run_tick
from algua.live.paper_loop import run_paper
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_strategy

paper_app = typer.Typer(help="Paper trading: run a paper-stage strategy", no_args_is_help=True)
app.add_typer(paper_app, name="paper")


def _alpaca_broker_from_settings() -> AlpacaPaperBroker:
    s = get_settings()
    if not s.alpaca_api_key or not s.alpaca_api_secret:
        raise ValueError(
            "Alpaca paper credentials not configured; set ALGUA_ALPACA_API_KEY "
            "and ALGUA_ALPACA_API_SECRET"
        )
    return AlpacaPaperBroker(api_key=s.alpaca_api_key, api_secret=s.alpaca_api_secret,
                             base_url=s.alpaca_paper_url)


@paper_app.command("run")
@json_errors(ValueError, LookupError, BacktestError)
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="paper-run an ingested bars snapshot"),  # noqa: B008
    cash: float = typer.Option(100_000.0, "--cash", help="starting paper cash"),
    max_drawdown: float = typer.Option(1.0, "--max-drawdown",
                                       help="trip the kill-switch if equity falls this fraction below peak (1.0 = off)"),  # noqa: E501
) -> None:
    """Replay a paper-stage strategy through the sim broker and persist orders/fills."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; paper run requires 'paper'")
        if kill_switch.is_tripped(conn, name):
            raise ValueError(
                f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'"
            )
        provider = _select_provider(demo, snapshot)
        try:
            result = run_paper(strategy, SimBroker(cash=cash), provider,
                               _utc(start), _utc(end), max_drawdown=max_drawdown)
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped", "error": exc.detail})
            raise typer.Exit(1) from exc
        persist_run(conn, result)
        audit_append(
            conn, actor="agent", action="paper_run",
            reason=f"{len(result.orders)} orders, {len(result.fills)} fills",
            strategy=name,
        )

    emit({
        "strategy": result.strategy,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "final_positions": result.final_positions,
        "final_cash": result.final_cash,
        "final_equity": result.final_equity,
        "reconcile_ok": result.reconcile_ok,
    })


@paper_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show persisted paper state (orders count + derived positions) for a strategy."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        n_orders = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE strategy = ?", (name,)
        ).fetchone()[0]
        positions = derive_positions(conn, name)
        ks = kill_switch.get(conn, name)
    emit({
        "strategy": name, "n_orders": n_orders, "positions": positions,
        "kill_switch": {"tripped": ks is not None, "reason": ks["reason"] if ks else None},
    })


@paper_app.command("kill")
@json_errors(ValueError, LookupError)
def kill(
    name: str,
    reason: str = typer.Option(..., "--reason", help="why the strategy is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Manually trip the kill-switch for a strategy (halts paper runs until reset)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # reject unknown/mistyped names before tripping a switch
        SqliteStrategyRepository(conn).get(name)
        kill_switch.trip(conn, name, reason=reason, actor=actor)
        audit_append(conn, actor=actor, action="kill_switch_trip", reason=reason, strategy=name)
    emit({"strategy": name, "kill_switch": "tripped", "reason": reason})


@paper_app.command("resume")
@json_errors(ValueError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. Human action."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        was_tripped = kill_switch.is_tripped(conn, name)
        if was_tripped:
            # Audit BEFORE clearing: if the reset write fails, the switch stays tripped
            # (fail-safe — still halted) rather than cleared with no audit trail.
            audit_append(conn, actor="human", action="kill_switch_reset",
                         reason="manual resume", strategy=name)
            kill_switch.reset(conn, name)
    emit({"strategy": name, "kill_switch": "reset" if was_tripped else "not_tripped"})


@paper_app.command("account")
@json_errors(ValueError, BrokerError)
def account() -> None:
    """Show the Alpaca paper account (equity/cash/buying-power) — a connectivity smoke."""
    broker = _alpaca_broker_from_settings()
    acct = broker.account()
    emit({"equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power})


@paper_app.command("trade-live")
@json_errors(ValueError, LookupError, BrokerError)
def trade_live(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
) -> None:
    """Run ONE wall-clock tick: submit Alpaca market-order deltas toward the strategy's target."""
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; trade-live requires 'paper'")
        if kill_switch.is_tripped(conn, name):
            raise ValueError(
                f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'"
            )
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        try:
            result = run_tick(strategy, broker, provider, _utc(start), _utc(end))
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
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
            payload = {"ok": False, "kind": exc.kind, "kill_switch": "tripped",
                       "liquidation_submitted": liquidation_submitted, "error": exc.detail}
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
        now = datetime.now(UTC).isoformat()
        decision_ts_str = result.decision_ts.isoformat() if result.decision_ts else None
        for o in result.submitted:
            conn.execute(
                "INSERT INTO paper_orders"
                "(strategy, symbol, side, target_weight, decision_ts, submitted_ts,"
                " status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
                (name, o["symbol"], o["side"], o["target_weight"],
                 decision_ts_str, now, "submitted", o["order_id"]),
            )
        conn.commit()
        audit_append(conn, actor="agent", action="trade_live",
                     reason=f"{len(result.submitted)} orders submitted", strategy=name)

    emit({
        "strategy": name,
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "target_weights": result.target_weights,
        "positions_before": result.positions_before,
        "submitted": result.submitted,
    })


@paper_app.command("flatten")
@json_errors(ValueError, LookupError, BrokerError)
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: close this strategy's live positions (its universe) and trip its kill-switch."""
    strategy = load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; flatten requires 'paper'")
        broker = _alpaca_broker_from_settings()
        # Halt first (fail-safe): the strategy is stopped even if the close call then fails.
        kill_switch.trip(conn, name, reason="flatten", actor=actor)
        audit_append(conn, actor=actor, action="flatten", reason="manual flatten", strategy=name)
        try:
            broker.cancel_open_orders()
            broker.close_positions(strategy.universe)
        except BrokerError as exc:
            emit({"ok": False, "strategy": name, "kill_switch": "tripped",
                  "liquidation_submitted": False, "error": str(exc)})
            raise typer.Exit(1) from exc
    # liquidation_submitted: Alpaca accepted the close orders; fills land async (may be next open).
    emit({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True})
