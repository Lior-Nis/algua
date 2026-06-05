from __future__ import annotations

import sqlite3

import typer

from algua.audit.log import append as audit_append
from algua.backtest.engine import BacktestError
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.alpaca_broker import AlpacaPaperBroker, BrokerError
from algua.execution.order_state import (
    clear_peak_equity,
    client_order_id,
    count_orders,
    derive_positions,
    get_peak_equity,
    persist_run,
    record_submitted_order,
    update_peak_equity,
)
from algua.execution.sim_broker import SimBroker
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.live.paper_loop import run_paper
from algua.registry.store import SqliteStrategyRepository
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy
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


def _load_gated_strategy(conn: sqlite3.Connection, name: str, command: str) -> LoadedStrategy:
    """Load a strategy and clear the two gates every trading command shares: it must be at the
    PAPER stage and its kill-switch must not be tripped. ``command`` only colours the error text.

    Centralises the stage + kill-switch preamble that ``run`` and ``trade-tick`` both need (an SRP
    fix: the command bodies no longer hand-roll these checks). Commands that intentionally TRIP the
    switch (kill/flatten) do their own, narrower gating instead.
    """
    strategy = load_strategy(name)
    rec = SqliteStrategyRepository(conn).get(name)
    if rec.stage is not Stage.PAPER:
        raise ValueError(f"{name} is at stage '{rec.stage.value}'; {command} requires 'paper'")
    if kill_switch.is_tripped(conn, name):
        raise ValueError(f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
    return strategy


def _trip(conn: sqlite3.Connection, name: str, exc: RiskBreach) -> None:
    """Trip the kill-switch for a risk breach and write the matching audit row (the shared half of
    both commands' breach handling; the divergent emit/flatten stays in each caller)."""
    kill_switch.trip(conn, name, reason=exc.detail, actor="system")
    audit_append(conn, actor="system", action="kill_switch_trip",
                 reason=f"{exc.kind}: {exc.detail}", strategy=name)


@paper_app.command("run")
@json_errors(ValueError, LookupError, BacktestError)
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="paper-run an ingested bars snapshot"),
    cash: float = typer.Option(100_000.0, "--cash", help="starting paper cash"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="trip the kill-switch if equity falls this fraction below peak (omit to disable)"),  # noqa: E501
) -> None:
    """Replay a paper-stage strategy through the sim broker and persist orders/fills."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        strategy = _load_gated_strategy(conn, name, "paper run")
        provider = _select_provider(demo, snapshot)
        try:
            result = run_paper(strategy, SimBroker(cash=cash), provider,
                               utc(start), utc(end), max_drawdown=max_drawdown)
        except RiskBreach as exc:
            _trip(conn, name, exc)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped", "error": exc.detail})
            raise typer.Exit(1) from exc
        persist_run(conn, result)
        audit_append(
            conn, actor="agent", action="paper_run",
            reason=f"{len(result.orders)} orders, {len(result.fills)} fills",
            strategy=name,
        )

    emit(ok({
        "strategy": result.strategy,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "final_positions": result.final_positions,
        "final_cash": result.final_cash,
        "final_equity": result.final_equity,
        "reconcile_ok": result.reconcile_ok,
    }))


@paper_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show persisted paper state (orders count + derived positions) for a strategy."""
    with registry_conn() as conn:
        n_orders = count_orders(conn, name)
        positions = derive_positions(conn, name)
        ks = kill_switch.get(conn, name)
    emit(ok({
        "strategy": name, "n_orders": n_orders, "positions": positions,
        "kill_switch": {"tripped": ks is not None, "reason": ks["reason"] if ks else None},
    }))


@paper_app.command("kill")
@json_errors(ValueError, LookupError)
def kill(
    name: str,
    reason: str = typer.Option(..., "--reason", help="why the strategy is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Manually trip the kill-switch for a strategy (halts paper runs until reset)."""
    with registry_conn() as conn:
        # reject unknown/mistyped names before tripping a switch
        SqliteStrategyRepository(conn).get(name)
        kill_switch.trip(conn, name, reason=reason, actor=actor)
        audit_append(conn, actor=actor, action="kill_switch_trip", reason=reason, strategy=name)
    emit(ok({"strategy": name, "kill_switch": "tripped", "reason": reason}))


@paper_app.command("resume")
@json_errors(ValueError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. Human action."""
    with registry_conn() as conn:
        was_tripped = kill_switch.is_tripped(conn, name)
        if was_tripped:
            # Audit BEFORE clearing: if the reset write fails, the switch stays tripped
            # (fail-safe — still halted) rather than cleared with no audit trail.
            audit_append(conn, actor="human", action="kill_switch_reset",
                         reason="manual resume", strategy=name)
            kill_switch.reset(conn, name)
            # Re-base the drawdown high-water mark to the (post-flatten) current equity, so the
            # next tick doesn't instantly re-trip the breaker against the stale pre-loss peak (#27).
            clear_peak_equity(conn, name)
    emit(ok({"strategy": name, "kill_switch": "reset" if was_tripped else "not_tripped"}))


@paper_app.command("account")
@json_errors(ValueError, BrokerError)
def account() -> None:
    """Show the Alpaca paper account (equity/cash/buying-power) — a connectivity smoke."""
    broker = _alpaca_broker_from_settings()
    acct = broker.account()
    emit(ok({"equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power}))


@paper_app.command("trade-tick")
@json_errors(ValueError, LookupError, BrokerError)
def trade_tick(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="halt + flatten if equity falls this fraction below the persisted peak (omit to disable)"),  # noqa: E501
) -> None:
    """Run ONE wall-clock tick against the PAPER venue: submit Alpaca market-order deltas toward
    the strategy's target. Each accepted order persists immediately; a drawdown/exposure/reconcile
    breach trips the kill-switch and flattens. Never trades live (the broker refuses a live URL)."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        strategy = _load_gated_strategy(conn, name, "trade-tick")
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)

        def _persist(record: SubmittedOrder) -> None:
            # Persist each accepted order immediately so a mid-loop death can't lose it (#18).
            record_submitted_order(conn, name, record.symbol, record.side, record.target_weight,
                                   record.decision_ts.isoformat(), record.order_id)

        hooks = TickHooks(
            client_order_id_for=client_order_id,
            on_submitted=_persist,
            # Re-read the switch from the DB right before submit so an externally-tripped switch
            # aborts before any order goes out (#21).
            should_halt=lambda: kill_switch.is_tripped(conn, name),
            peak_equity=get_peak_equity(conn, name),
            derived_positions=derive_positions(conn, name),
        )
        try:
            result = run_tick(strategy, broker, provider, utc(start), utc(end),
                              hooks=hooks, max_drawdown=max_drawdown)
        except TickHalted as exc:
            # Switch tripped between cancel and submit: nothing was sent this tick. Already halted.
            audit_append(conn, actor="system", action="trade_tick_halted",
                         reason=str(exc), strategy=name)
            emit({"ok": False, "strategy": name, "kill_switch": "tripped", "halted": True,
                  "error": str(exc)})
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
            payload = {"ok": False, "kind": exc.kind, "kill_switch": "tripped",
                       "liquidation_submitted": liquidation_submitted, "error": exc.detail}
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
        if result.peak_equity is not None:
            update_peak_equity(conn, name, result.peak_equity)
        audit_append(conn, actor="agent", action="trade_tick",
                     reason=f"{len(result.submitted)} orders submitted", strategy=name)

    emit(ok({
        "strategy": name,
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "target_weights": result.target_weights,
        "positions_before": result.positions_before,
        "submitted": result.submitted,
        "reconcile_ok": result.reconcile_ok,
        "realized_gross": result.realized_gross,
    }))


@paper_app.command("flatten")
@json_errors(ValueError, LookupError, BrokerError)
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: close this strategy's live positions (its universe) and trip its kill-switch."""
    strategy = load_strategy(name)
    with registry_conn() as conn:
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
    emit(ok({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True}))
