from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import typer

from algua.audit.log import append as audit_append
from algua.backtest.engine import BacktestError
from algua.calendar.market_calendar import MarketCalendar
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.execution.alpaca_broker import AlpacaLiveReadOnlyBroker, AlpacaPaperBroker, BrokerError
from algua.execution.live_ledger import (
    believed_positions,
    fill_cursor,
    ingest_activities,
    strategy_live_symbols,
)
from algua.execution.live_reconcile import attributed_live_net
from algua.execution.order_state import (
    clear_all_nav_peaks,
    clear_all_peaks,
    clear_derived_positions,
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
from algua.execution.tick_clock import tick_clock
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.live.paper_loop import run_paper
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.forward_promotion import forward_promotion_preflight, run_forward_gate
from algua.registry.gating import load_gated_strategy
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import (
    DEGRADATION_FACTOR,
    MAX_FORWARD_DRAWDOWN,
    MAX_STALENESS_SESSIONS,
    MIN_FORWARD_OBSERVATIONS,
    MIN_FORWARD_VOL,
    MIN_SESSION_COVERAGE,
    SHARPE_FLOOR,
    ForwardGateCriteria,
)
from algua.risk import global_halt, kill_switch
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


_RECONCILE_TOL = 1e-6


def _alpaca_live_readonly_from_settings() -> AlpacaLiveReadOnlyBroker:
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError(
            "Alpaca LIVE credentials not configured; cannot confirm the strategy is flat at the "
            "broker — set ALGUA_ALPACA_LIVE_API_KEY and ALGUA_ALPACA_LIVE_API_SECRET"
        )
    return AlpacaLiveReadOnlyBroker(s.alpaca_live_api_key, s.alpaca_live_api_secret,
                                    base_url=s.alpaca_live_url)


def _maybe_live_readonly() -> AlpacaLiveReadOnlyBroker | None:
    """A read-only live client if live creds are configured, else None (resume-all stays lenient:
    with no creds it just computes not_flat from the current belief)."""
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        return None
    return AlpacaLiveReadOnlyBroker(s.alpaca_live_api_key, s.alpaca_live_api_secret,
                                    base_url=s.alpaca_live_url)


def _live_strategy_flat(
    conn: sqlite3.Connection, name: str, universe: list[str], broker: object,
) -> tuple[bool, dict]:
    """Ingest pending broker activities, then ACCOUNT-WIDE reconcile: the strategy is flat iff its
    own believed_positions is empty AND the broker holds no UNEXPLAINED qty (broker net minus the
    books' LIVE-attributed net) in any symbol it is responsible for. A sibling LIVE strategy that
    legitimately holds the same symbol explains the broker qty and does not block resume; an orphan
    (unattributed/manual) or non-live holding does NOT explain it, so it fails closed (refuse)."""
    ingest_activities(conn, broker.account_activities(after=fill_cursor(conn)))  # type: ignore[attr-defined]
    own = believed_positions(conn, name)
    broker_net = {s: float(q) for s, q in broker.get_positions().items()  # type: ignore[attr-defined]
                  if float(q) != 0.0}
    expected = attributed_live_net(conn)
    syms = set(universe) | strategy_live_symbols(conn, name)
    unexplained = {
        s: broker_net.get(s, 0.0) - expected.get(s, 0.0)
        for s in syms
        if abs(broker_net.get(s, 0.0) - expected.get(s, 0.0)) > _RECONCILE_TOL
    }
    is_flat = (not own) and (not unexplained)
    return is_flat, {"believed": own, "broker_unexplained": unexplained}


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


def _strategy_held_symbols(
    conn: sqlite3.Connection, strategy: str, universe: list[str]
) -> list[str]:
    """Universe ∪ every symbol THIS strategy has submitted a paper order for — so a held-but-dropped
    symbol (traded before its universe changed) is still exited, WITHOUT closing a sibling's
    positions on a shared paper account. Closing a flat name is a 404 no-op, so over-including
    never-filled names is harmless."""
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM paper_orders WHERE strategy = ?", (strategy,)
    ).fetchall()
    return sorted(set(universe) | {r["symbol"] for r in rows})


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
        strategy, _rec = load_gated_strategy(conn, name, "paper run")
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
@json_errors(ValueError, BrokerError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. For a LIVE strategy,
    confirms the strategy is flat via broker-truth reconcile before allowing resume. Human
    action."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.LIVE:
            strategy = load_strategy(name)
            broker = _alpaca_live_readonly_from_settings()
            is_flat, residual = _live_strategy_flat(conn, name, strategy.universe, broker)
            if not is_flat:
                raise ValueError(
                    f"{name} is not flat after reconcile: {residual}; offset fills pending or "
                    "liquidation incomplete — re-flatten or retry after fills land"
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
        strategy, rec = load_gated_strategy(conn, name, "trade-tick")
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        identity = compute_artifact_hashes(name)
        acct = broker.account()

        def _persist(record: SubmittedOrder) -> None:
            # Persist each accepted order immediately so a mid-loop death can't lose it (#18).
            record_submitted_order(conn, name, record.symbol, record.side, record.target_weight,
                                   record.decision_ts.isoformat(), record.order_id,
                                   strategy_id=rec.id)

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
            symbols = _strategy_held_symbols(conn, name, strategy.universe)
            try:
                broker.cancel_open_orders()
                broker.close_positions(symbols)
            except BrokerError as fexc:
                liquidation_submitted = False
                flatten_error = str(fexc)
                audit_append(conn, actor="system", action="flatten_failed",
                             reason=str(fexc), strategy=name)
            else:
                # Close succeeded: reset the derived belief so the next reconcile starts flat.
                clear_derived_positions(conn, name)
            payload = _breach_payload(exc.detail, kind=exc.kind,
                                      liquidation_submitted=liquidation_submitted,
                                      closed_symbols=symbols)
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
        if result.peak_equity is not None:
            update_peak_equity(conn, name, result.peak_equity)
            tick_ts, clock_source = tick_clock(broker.clock)
            record_tick_snapshot(
                conn, name,
                tick_ts=tick_ts,
                decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
                equity=result.equity, peak_equity=result.peak_equity,
                positions=result.positions_before, n_submitted=len(result.submitted),
                reconcile_ok=result.reconcile_ok,
                lane="paper", strategy_id=rec.id,
                code_hash=identity.code_hash, config_hash=identity.config_hash,
                dependency_hash=identity.dependency_hash,
                account_id=acct.account_id, cash=acct.cash,
                clock_source=clock_source,
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


@paper_app.command("promote")
@json_errors(ValueError, LookupError, BrokerError)
def promote(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
    degradation_factor: float = typer.Option(
        DEGRADATION_FACTOR, "--degradation-factor",
        help="realized Sharpe must beat this fraction of the qualified holdout Sharpe "
             "(raising it is stricter; lowering is human-only)"),
    sharpe_floor: float = typer.Option(
        SHARPE_FLOOR, "--sharpe-floor",
        help="absolute realized-Sharpe floor (raising is stricter; lowering is human-only)"),
    min_observations: int = typer.Option(
        MIN_FORWARD_OBSERVATIONS, "--min-observations",
        help="minimum daily return observations in the forward window "
             "(raising is stricter; lowering is human-only)"),
    min_coverage: float = typer.Option(
        MIN_SESSION_COVERAGE, "--min-coverage",
        help="minimum decided-sessions / trading-sessions coverage "
             "(raising is stricter; lowering is human-only)"),
    min_vol: float = typer.Option(
        MIN_FORWARD_VOL, "--min-vol",
        help="annualized volatility floor — a do-nothing strategy must not pass "
             "(raising is stricter; lowering is human-only)"),
    max_drawdown: float = typer.Option(
        MAX_FORWARD_DRAWDOWN, "--max-drawdown",
        help="max drawdown over the evidence series "
             "(lowering is stricter; raising is human-only)"),
    max_staleness: int = typer.Option(
        MAX_STALENESS_SESSIONS, "--max-staleness",
        help="newest admissible tick may be at most this many sessions old "
             "(lowering is stricter; raising is human-only)"),
) -> None:
    """Forward-test evidence gate (#124): evaluate this strategy's wall-clock paper evidence
    and promote paper -> forward_tested on pass. At forward_tested: re-evaluate, refreshing
    the live-wall certificate, no stage change. The paper-side analog of `research promote`;
    relaxing any threshold below its protected default is human-only."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching the DB
    criteria = ForwardGateCriteria(
        min_forward_observations=min_observations,
        min_session_coverage=min_coverage,
        degradation_factor=degradation_factor,
        sharpe_floor=sharpe_floor,
        min_forward_vol=min_vol,
        max_forward_drawdown=max_drawdown,
        max_staleness_sessions=max_staleness,
    )
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        # PREFLIGHT: actor legality + relaxations-need-human + stage legality. Refuses here,
        # before the broker is even constructed (TransitionError is a ValueError -> JSON error).
        forward_promotion_preflight(repo, name, actor=actor_enum, criteria=criteria)
        broker = _alpaca_broker_from_settings()
        outcome = run_forward_gate(
            repo, conn, name=name, actor=actor_enum, criteria=criteria,
            calendar=MarketCalendar(), now=datetime.now(UTC),
            activities_fetch=broker.account_activities_window)
        audit_append(conn, actor=actor, action="paper_promote",
                     reason="pass" if outcome.decision.passed else "fail", strategy=name)
    payload = {
        "strategy": name,
        "passed": outcome.decision.passed,
        "promoted": outcome.promoted,
        "decision": outcome.decision.to_dict(),
        "excluded_ticks": outcome.assembled.excluded,
        "n_concurrent_forward": outcome.assembled.n_concurrent_forward,
    }
    # Pass mirrors research_cmd.promote's success envelope; a fail still emits the full
    # decision payload (the evaluation row was recorded) but carries the repo-wide exit-1
    # discriminator ("ok": false, see cli._common.ok) and exits non-zero.
    emit(ok(payload) if outcome.decision.passed else {"ok": False, **payload})
    if not outcome.decision.passed:
        raise typer.Exit(1)


@paper_app.command("flatten")
@json_errors(ValueError, LookupError, BrokerError)
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: close this strategy's paper positions (its own held + universe symbols) and trip
    its kill-switch."""
    strategy = load_strategy(name)
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        # A forward_tested strategy still holds paper positions while awaiting the go-live
        # signature, so the emergency exit must reach it too (#124 GATE-2).
        if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
            raise ValueError(
                f"{name} is at stage '{rec.stage.value}'; "
                "flatten requires 'paper' or 'forward_tested'")
        broker = _alpaca_broker_from_settings()
        # Halt first (fail-safe): the strategy is stopped even if the close call then fails.
        kill_switch.trip(conn, name, reason="flatten", actor=actor)
        audit_append(conn, actor=actor, action="flatten", reason="manual flatten", strategy=name)
        symbols = _strategy_held_symbols(conn, name, strategy.universe)
        try:
            broker.cancel_open_orders()
            broker.close_positions(symbols)
        except BrokerError as exc:
            emit(_breach_payload(str(exc), strategy=name, liquidation_submitted=False))
            raise typer.Exit(1) from exc
        # Close succeeded: reset the derived belief so a later trade-tick reconcile starts flat.
        clear_derived_positions(conn, name)
    # liquidation_submitted: Alpaca accepted the close orders; fills land async (may be next open).
    emit(ok({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True,
             "closed_symbols": symbols}))


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
@json_errors(ValueError, BrokerError)
def resume_all(
    actor: str = typer.Option("human", "--actor", help="human | agent"),
) -> None:
    """Clear the global halt and re-base every strategy's drawdown peak (the account was flattened
    to cash). Per-strategy kill-switches are left untouched."""
    with registry_conn() as conn:
        was_set = global_halt.is_engaged(conn)
        # Flag any LIVE strategies that still carry a ledger position (partial-fill residual): they
        # are not flat and must be re-flattened before their individual kill-switches can be reset.
        # We skip+flag rather than aborting so the global halt clears and other strategies recover.
        live_rows = conn.execute(
            "SELECT name FROM strategies WHERE stage = 'live'"
        ).fetchall()
        if live_rows:
            broker = _maybe_live_readonly()
            if broker is not None:
                # account-wide ingest so not_flat reflects post-ingest belief (landed offset fills)
                ingest_activities(conn, broker.account_activities(after=fill_cursor(conn)))
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
