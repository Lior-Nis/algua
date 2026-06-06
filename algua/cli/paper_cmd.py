from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

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
    clear_all_nav_peaks,
    clear_all_peaks,
    clear_nav_peak,
    clear_peak_equity,
    client_order_id,
    count_orders,
    derive_positions,
    get_peak_equity,
    latest_tick_snapshot,
    persist_run,
    recent_orders,
    record_submitted_order,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.execution.sim_broker import SimBroker
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.live.paper_loop import run_paper
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
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
    if global_halt.is_engaged(conn):
        raise ValueError("global halt active; clear with 'algua paper resume-all'")
    if kill_switch.is_tripped(conn, name):
        raise ValueError(f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
    return strategy


def _breach_payload(error: str, **extra: object) -> dict:
    """A failure envelope for a tripped kill-switch: ``{"ok": false, "kill_switch": "tripped"...}``.

    The shared skeleton of every paper-command halt/breach emit; callers pass the human-readable
    ``error`` plus whatever variant keys (``kind``, ``strategy``, ``halted``, ...) that path adds.
    """
    return {"ok": False, "kill_switch": "tripped", "error": error, **extra}


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
            emit(_breach_payload(exc.detail, kind=exc.kind))
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
    """Consolidated per-strategy operability view — stage, kill-switch, drawdown, last tick,
    recent orders, and a health rollup. A pure read of persisted state (no broker call)."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # unknown name -> LookupError -> {ok:false}
        n_orders = count_orders(conn, name)
        if rec.stage is Stage.LIVE:
            from algua.execution.live_ledger import believed_positions
            from algua.execution.order_state import get_nav_peak
            positions = believed_positions(conn, name)
            peak = get_nav_peak(conn, name)
        else:
            positions = derive_positions(conn, name)
            peak = get_peak_equity(conn, name)
        ks = kill_switch.get(conn, name)
        halted_globally = global_halt.is_engaged(conn)
        last = latest_tick_snapshot(conn, name)
        orders = recent_orders(conn, name, 10)
    tripped = ks is not None
    last_equity = last["equity"] if last else None
    drawdown = (
        1.0 - last_equity / peak
        if last_equity is not None and peak is not None and peak > 0 else None
    )
    if tripped or halted_globally:
        health = "halted"
    elif last is not None and not last["reconcile_ok"]:
        health = "drift"
    elif last is None:
        health = "idle"
    else:
        health = "ok"
    emit(ok({
        "strategy": name,
        "stage": rec.stage.value,
        "kill_switch": {"tripped": tripped, "reason": ks["reason"] if ks else None,
                        "global_halt": halted_globally},
        "drawdown": {"peak_equity": peak, "last_equity": last_equity, "drawdown": drawdown},
        "last_tick": last,
        "positions": positions,
        "n_orders": n_orders,
        "recent_orders": orders,
        "health": health,
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
    from algua.execution.live_ledger import believed_positions
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.LIVE and believed_positions(conn, name):
            raise ValueError(
                f"{name} is not flat (believed positions: {believed_positions(conn, name)}); "
                "re-flatten before resuming a live strategy"
            )
        was_tripped = kill_switch.is_tripped(conn, name)
        if was_tripped:
            # Audit BEFORE mutating: if a write fails the switch stays tripped (fail-safe — still
            # halted) rather than cleared with no audit trail.
            audit_append(conn, actor="human", action="kill_switch_reset",
                         reason="manual resume (re-bases drawdown peak)", strategy=name)
            # Re-base the drawdown high-water mark to current equity FIRST, then clear the
            # kill-switch LAST so the actual un-halt is the final write: any earlier failure leaves
            # the strategy safely halted and resume is retryable. Without the rebase, a drawdown
            # trip -> flatten-to-cash re-trips every tick against the stale pre-loss peak (#27).
            # A live strategy's drawdown breaker uses the NAV peak (live_nav_peaks), not the paper
            # peak — clear the right one per stage, else a resumed live strategy re-trips on a stale
            # pre-breach NAV peak (codex C1 review).
            if rec.stage is Stage.LIVE:
                clear_nav_peak(conn, name)
            else:
                clear_peak_equity(conn, name)
            kill_switch.reset(conn, name)
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
            should_halt=lambda: kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn),
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
            emit(_breach_payload(str(exc), strategy=name, liquidation_submitted=False))
            raise typer.Exit(1) from exc
    # liquidation_submitted: Alpaca accepted the close orders; fills land async (may be next open).
    emit(ok({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True}))


@paper_app.command("halt-all")
@json_errors(ValueError, LookupError, BrokerError)
def halt_all(
    reason: str = typer.Option(..., "--reason", help="why the whole account is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """ACCOUNT-WIDE emergency: engage the global halt and flatten the ENTIRE Alpaca account."""
    with registry_conn() as conn:
        broker = _alpaca_broker_from_settings()
        # Engage first (fail-safe): all trading is stopped even if the close call then fails.
        global_halt.engage(conn, reason=reason, actor=actor)
        audit_append(conn, actor=actor, action="halt_all", reason=reason, strategy=None)
        try:
            broker.close_all_positions()
        except BrokerError as exc:
            audit_append(conn, actor="system", action="flatten_failed", reason=str(exc),
                         strategy=None)
            emit({"ok": False, "global_halt": "set", "liquidation_submitted": False,
                  "error": str(exc)})
            raise typer.Exit(1) from exc
    emit(ok({"global_halt": "set", "liquidation_submitted": True}))


@paper_app.command("resume-all")
@json_errors(ValueError)
def resume_all(
    actor: str = typer.Option("human", "--actor", help="human | agent"),
) -> None:
    """Clear the global halt and re-base every strategy's drawdown peak (the account was flattened
    to cash). Per-strategy kill-switches are left untouched."""
    from algua.execution.live_ledger import believed_positions
    with registry_conn() as conn:
        was_set = global_halt.is_engaged(conn)
        # Flag any LIVE strategies that still carry a ledger position (partial-fill residual): they
        # are not flat and must be re-flattened before their individual kill-switches can be reset.
        # We skip+flag rather than aborting so the global halt clears and other strategies recover.
        live_rows = conn.execute(
            "SELECT name FROM strategies WHERE stage = 'live'"
        ).fetchall()
        not_flat = [
            r["name"] for r in live_rows if believed_positions(conn, r["name"])
        ]
        if was_set:
            audit_append(conn, actor=actor, action="resume_all",
                         reason="clear global halt; re-base all drawdown peaks", strategy=None)
            # Re-base peaks first, clear the halt LAST so the un-halt is the final write (#109).
            # Clear BOTH the paper (account-equity) and live (NAV) peak tables so resumed strategies
            # re-base on their next tick rather than re-tripping a stale peak (codex C1 review).
            clear_all_peaks(conn)
            clear_all_nav_peaks(conn)
            global_halt.clear(conn)
    result: dict = {"global_halt": "reset" if was_set else "not_set"}
    if not_flat:
        result["live_not_flat"] = not_flat
        result["warning"] = (
            "the above live strategies are not flat; re-flatten each before resuming individually"
        )
    emit(ok(result))
