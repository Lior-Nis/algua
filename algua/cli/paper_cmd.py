from __future__ import annotations

import importlib
import json
import sqlite3
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import typer

from algua.audit.log import append as audit_append
from algua.calendar.factory import get_calendar
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
from algua.contracts.types import (
    ActivityWindowBroker,
    LiveReconcileBroker,
    OrderIntent,
    OrderLookupBroker,
    PositionsBroker,
    ScopedCancelBroker,
)
from algua.evaluation.backtest_run import run_backtest_task
from algua.evaluation.inputs import select_provider as _select_provider
from algua.evaluation.sweep_run import sweep_task
from algua.execution import paper_reconcile
from algua.execution.alpaca_broker import (
    AlpacaLiveReadOnlyBroker,
    AlpacaPaperBroker,
    posted_notional,
)
from algua.execution.broker_factory import BrokerKind, build_broker, maybe_broker
from algua.execution.errors import BrokerError
from algua.execution.flatten import flatten_strategy
from algua.execution.fleet_health import strategy_health
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_paper_venue_broker_order_id,
    believed_positions,
    delete_paper_venue_order,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
    paper_believed_positions,
    record_paper_venue_order,
    recover_stranded_broker_order_ids,
    strategy_live_symbols,
)
from algua.execution.live_reconcile import attributed_live_net
from algua.execution.live_sizing import LiveSizingError, build_paper_sizing_snapshot
from algua.execution.order_state import (
    client_order_id,
    get_peak_equity,
    persist_run,
    recent_orders,
    recent_venue_orders,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.execution.peaks import rebase_all_peaks, rebase_strategy_peak
from algua.execution.sim_broker import SimBroker
from algua.execution.tick_clock import tick_clock
from algua.live.live_loop import (
    _RECONCILE_TOL,
    SubmittedOrder,
    TickHalted,
    TickHooks,
    run_tick,
)
from algua.live.paper_loop import run_paper
from algua.observability import (
    CycleCounters,
    configure_logging,
    correlation_context,
    get_logger,
)
from algua.operator.journal import JsonlJournal
from algua.operator.mergeback import RealGitOps, merge_back_lock, run_merge_back
from algua.primitives.timeparse import utc
from algua.registry import allocations
from algua.registry.allocations import active_allocation
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.db import registry_conn
from algua.registry.forward_promotion import forward_promotion_preflight, run_forward_gate
from algua.registry.gating import load_gated_strategy
from algua.registry.human_actor import authenticate_actor, canonical_run_context
from algua.registry.intake import run_intake
from algua.registry.kb_sync import sync_kb_doc
from algua.registry.promote_run import promote_task
from algua.registry.repository import StrategyNotFound
from algua.registry.store import SqliteStrategyRepository
from algua.registry.universe_binding import SOURCE_CONFIG_LEGACY, resolve_operational_universe
from algua.research.forward_gates import (
    DEGRADATION_FACTOR,
    FORWARD_SHARPE_CONFIDENCE,
    MAX_FORWARD_DRAWDOWN,
    MAX_STALENESS_SESSIONS,
    MIN_FORWARD_OBSERVATIONS,
    MIN_FORWARD_VOL,
    MIN_SESSION_COVERAGE,
    SHARPE_FLOOR,
    ForwardGateCriteria,
)
from algua.risk import global_halt, kill_switch
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_strategy

paper_app = typer.Typer(help="Paper trading: run a paper-stage strategy", no_args_is_help=True)
app.add_typer(paper_app, name="paper")

log = get_logger(__name__)


def _alpaca_broker_from_settings() -> AlpacaPaperBroker:
    return build_broker(BrokerKind.ALPACA_PAPER)


def _paper_broker_net(broker: PositionsBroker) -> dict[str, float]:
    """Paper broker's net positions per symbol (nonzero only) for account reconcile.

    Local to paper_cmd because the live analog (_broker_net_positions) can't be imported (cli->cli).
    """
    pos = broker.get_positions()  # pandas Series symbol -> qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


_PAPER_CURSOR_FAR_PAST = "1970-01-01T00:00:00Z"


def _recover_stranded(
    conn: sqlite3.Connection, broker: OrderLookupBroker, kind: LedgerKind
) -> None:
    """#312: backfill broker_order_id onto any crash-stranded NULL order row (asks the venue for the
    order carrying its client_order_id; never submits). ACCOUNT-WIDE, so the audit is too
    (strategy=None) — a per-strategy label would misattribute a sibling's order."""
    outcome = recover_stranded_broker_order_ids(conn, broker, kind=kind)
    if outcome.recovered:
        audit_append(conn, actor="system", action="stranded_order_recovered",
                     reason=f"{len(outcome.recovered)} backfilled: {outcome.recovered}",
                     strategy=None)
    if outcome.mismatched:
        audit_append(conn, actor="system", action="stranded_recovery_mismatch",
                     reason=f"{len(outcome.mismatched)} broker mismatch: {outcome.mismatched}",
                     strategy=None)


def _paper_scoped_cancel(conn, broker: ScopedCancelBroker, name: str) -> None:
    """Cancel only THIS strategy's open PAPER orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, name, kind=LedgerKind.PAPER):
        broker.cancel_order(oid)


def _ingest_paper_venue(
    conn: sqlite3.Connection, broker: ActivityWindowBroker, until: str
) -> None:
    """Exhaustively ingest the paper venue's activities into paper_venue_fills, fail-closed.
    Cursor is a broker-time high-water: fetch (cursor, until] (raises on a partial page), dedup by
    activity_id, persist `until` as the new cursor in the SAME transaction. The caller resolves
    `until` itself (never calls broker.clock() here), so a clock failure stays in its hands."""
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    acts = broker.account_activities_window(after, until)
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=until)


def _alpaca_live_readonly_from_settings() -> AlpacaLiveReadOnlyBroker:
    return build_broker(BrokerKind.ALPACA_LIVE_READONLY)


def _maybe_live_readonly() -> AlpacaLiveReadOnlyBroker | None:
    """A read-only live client if live creds are configured, else None (resume-all stays lenient:
    with no creds it just computes not_flat from the current belief)."""
    return maybe_broker(BrokerKind.ALPACA_LIVE_READONLY)


def _live_strategy_flat(
    conn: sqlite3.Connection, name: str, universe: list[str], broker: LiveReconcileBroker,
) -> tuple[bool, dict]:
    """Ingest pending broker activities, then ACCOUNT-WIDE reconcile: the strategy is flat iff its
    own believed_positions is empty AND the broker holds no UNEXPLAINED qty (broker net minus the
    books' LIVE-attributed net) in any symbol it is responsible for. A sibling LIVE strategy that
    legitimately holds the same symbol explains the broker qty and does not block resume; an orphan
    (unattributed/manual) or non-live holding does NOT explain it, so it fails closed (refuse)."""
    cursor = fill_cursor(conn, LedgerKind.LIVE)
    ingest_activities(conn, broker.account_activities(after=cursor), LedgerKind.LIVE)
    # #312: recover any crash-stranded NULL-broker_order_id live row before the flatness check, so a
    # stranded (accepted-but-not-backfilled) fill does not block resume as an unexplained residual.
    _recover_stranded(conn, broker, LedgerKind.LIVE)
    own = believed_positions(conn, name, LedgerKind.LIVE)
    broker_net = {s: float(q) for s, q in broker.get_positions().items()
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


@paper_app.command("run")
@json_errors
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="paper-run an ingested bars snapshot"),
    cash: float = typer.Option(100_000.0, "--cash", help="starting paper cash"),
    max_drawdown: float | None = typer.Option(None, "--max-drawdown",
                                       help="per-strategy drawdown breaker fraction; omit for the default-ON bound"),  # noqa: E501
    disable_drawdown_breaker: bool = typer.Option(
        False, "--disable-drawdown-breaker",
        help="HUMAN-ONLY emergency: turn the drawdown breaker fully OFF (audited)"),
) -> None:
    """Replay a paper-stage strategy through the sim broker and persist orders/fills."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    max_drawdown = resolve_drawdown_breaker(max_drawdown, disable_drawdown_breaker)
    with registry_conn() as conn:
        if disable_drawdown_breaker:
            audit_append(conn, actor="human", action="drawdown_breaker_disabled",
                         reason="paper run invoked with --disable-drawdown-breaker", strategy=name)
        strategy, _rec = load_gated_strategy(conn, name, "paper run")
        provider = _select_provider(demo, snapshot)
        try:
            result = run_paper(strategy, SimBroker(cash=cash), provider,
                               utc(start), utc(end), max_drawdown=max_drawdown)
        except RiskBreach as exc:
            trip_for_breach(conn, name, exc)
            emit(breach_payload(exc.detail, kind=exc.kind))
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
@json_errors
def show(name: str) -> None:
    """Consolidated per-strategy operability view — stage, kill-switch, drawdown, last tick,
    tick staleness, recent orders, and a health rollup. A pure read of persisted state (no broker
    call). The core rollup (incl. the #399 liveness/staleness verdict) is the SAME
    ``strategy_health`` engine that backs ``fleet status`` (DRY); ``show`` layers ``recent_orders``
    on top for the single-strategy drill-down."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # unknown name -> LookupError -> {ok:false}
        halted_globally = global_halt.is_engaged(conn)
        rollup = strategy_health(conn, rec, get_calendar(),
                                 halted_globally=halted_globally, now=datetime.now(UTC))
        if rec.stage is Stage.LIVE:
            orders = recent_orders(conn, name, 10)
        elif conn.execute(
            "SELECT 1 FROM paper_venue_orders WHERE strategy = ? LIMIT 1", (name,)
        ).fetchone() is not None:
            orders = recent_venue_orders(conn, name, 10)
        else:
            orders = recent_orders(conn, name, 10)
    emit(ok({**rollup, "recent_orders": orders}))


@paper_app.command("kill")
@json_errors
def kill(
    name: str,
    reason: str = typer.Option(..., "--reason", help="why the strategy is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Manually trip the kill-switch for a strategy (halts paper runs until reset)."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching a switch
    with registry_conn() as conn:
        # reject unknown/mistyped names before tripping a switch
        SqliteStrategyRepository(conn).get(name)
        kill_switch.trip(conn, name, reason=reason, actor=actor_enum.value)
        audit_append(conn, actor=actor_enum.value, action="kill_switch_trip",
                     reason=reason, strategy=name)
    emit(ok({"strategy": name, "kill_switch": "tripped", "reason": reason}))


@paper_app.command("resume")
@json_errors
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
            # Which table depends on the stage (a LIVE breaker reads the NAV peak, not the paper
            # peak) — that choice is the `rebase_strategy_peak` policy, named once in
            # algua.execution.peaks.
            rebase_strategy_peak(conn, name, rec.stage)
            kill_switch.reset(conn, name)
    emit(ok({"strategy": name, "kill_switch": "reset" if was_tripped else "not_tripped"}))


@paper_app.command("account")
@json_errors
def account() -> None:
    """Show the Alpaca paper account (equity/cash/buying-power) — a connectivity smoke."""
    broker = _alpaca_broker_from_settings()
    acct = broker.account()
    emit(ok({"equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power}))


def _resolve_max_concurrent(explicit: int | None) -> int:
    """Resolve ``--max-concurrent`` for ``intake``/``merge-back``/``allocate`` (factory slice 3).

    An omitted value (``None``) resolves to ``settings.paper_book_capacity`` (the shared wide-book
    default, env ``ALGUA_PAPER_BOOK_CAPACITY``); an explicit CLI value is honored as-is. Mirrors
    ``resolve_drawdown_breaker``'s None-sentinel shape rather than embedding ``get_settings()``
    directly in the ``typer.Option(...)`` default expression: a Python default value is evaluated
    ONCE, at function-definition (module-import) time, so baking ``get_settings()`` into the
    decorator would freeze the capacity at whatever the environment was when ``paper_cmd`` first
    imported — never re-reading `ALGUA_PAPER_BOOK_CAPACITY` afterward (including in tests that
    monkeypatch the env post-import). Resolving fresh, per invocation, inside the command body is
    what actually makes the setting env-overridable at runtime."""
    return explicit if explicit is not None else get_settings().paper_book_capacity


@paper_app.command('intake')
@json_errors
def intake(
    max_concurrent: int | None = typer.Option(None, '--max-concurrent',
        help='max concurrent paper-lane strategies admitted into the shared book '
             '(default: settings.paper_book_capacity / ALGUA_PAPER_BOOK_CAPACITY)'),
    actor: str = typer.Option('agent', '--actor', help='human | agent'),
) -> None:
    """Deterministic paper-book intake: fill the shared paper book up to capacity.

    RE-ADMISSION runs first (reported as ``readmitted``): a strategy already at a book stage
    (paper/forward_tested) that holds NO active allocation gets its slice restored. That state is a
    broken invariant, not a queue — `paper -> dormant` and `live -> paper` revoke the allocation
    while the return edges restore only the stage, leaving a strategy `paper run-all` skips as
    unallocated and `fleet health` alerts on forever. Re-entrants precede queued candidates (they
    were admitted first) and are funded under the same Σ ≤ equity and count-cap bounds; an
    already-allocated tenant is never touched (use `paper allocate` to resize).

    Then ADMISSION: each candidate is offered, in FIFO order (by candidate-entry
    stage_transitions.id, tie-break strategy id), to the ATOMIC
    ``intake_candidate_to_paper`` primitive, which under ONE
    write lock re-checks the --max-concurrent count cap, cap-checks + allocates an equal slice =
    floor(equity / max_concurrent to cents) (Σ allocations + slice ≤ paper account equity), and
    CASes candidate→paper — commit-or-rollback together, so there is no reachable
    transitioned-but-unallocated state. On either hard bound (book full / no capital headroom) the
    remaining candidates are left queued; a candidate raced out of ``candidate`` between selection
    and the txn is reported in ``skipped_stale`` and passed over. Reads paper account equity
    READ-ONLY before opening any transaction (no trading).

    ``--actor`` is a plain audit label here, NOT an authorization: ``candidate→paper`` is a
    non-token-gated transition legal for ``agent`` already (the multiple-testing gates were paid at
    the ``candidate`` boundary), so ``--actor human`` grants no privilege the agent lacks and needs
    no authenticated-actor discipline (unlike the ``forward_tested→live`` go-live edge)."""
    max_concurrent = _resolve_max_concurrent(max_concurrent)
    if max_concurrent <= 0:
        raise ValueError('--max-concurrent must be positive')
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching the DB
    equity = float(_alpaca_broker_from_settings().account().equity)  # broker read BEFORE any txn
    with registry_conn() as conn:
        payload = run_intake(conn, equity=equity, max_concurrent=max_concurrent, actor=actor_enum)
    emit(ok(payload))


@paper_app.command('allocate')
@json_errors
def allocate(
    name: str,
    capital: float = typer.Option(..., '--capital', help='paper capital base $'),
    max_concurrent: int | None = typer.Option(None, '--max-concurrent',
        help='max concurrent active paper-lane tenants '
             '(default: settings.paper_book_capacity / ALGUA_PAPER_BOOK_CAPACITY)'),
) -> None:
    """Set a paper/forward_tested strategy's capital base (the fixed sizing denominator).

    Lane-scoped to {paper, forward_tested}: the re-admission path for recovery/demotion re-entrants
    (dormant→paper, live→paper — which land UNALLOCATED by design) and for manual paper-book
    resizes. Enforces Σ(active paper allocations) ≤ paper-account equity AND the --max-concurrent
    count cap on a count-INCREASING allocation (a strategy with no active allocation yet). The
    authoritative stage + Σ + count checks all run UNDER one write lock inside ``allocate_in_lane``;
    the stage check here is only a friendly early error before the network account read (a strategy
    that leaves the lane between this check and the write can never be allocated). Re-allocating an
    existing tenant RESIZES it (exempt from the count cap; emits prior→new capital)."""
    max_concurrent = _resolve_max_concurrent(max_concurrent)
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # friendly early error only
        if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
            raise ValueError(
                f"cannot paper-allocate {name!r} at stage {rec.stage.value}; requires stage "
                "'paper' or 'forward_tested' (candidates enter only via `paper intake`; use "
                "`live allocate` for live)")
        prior = active_allocation(conn, rec.id)
        prior_capital = float(prior['capital']) if prior is not None else 0.0
        # Paper equity source, read BEFORE the write txn (mirrors `paper intake`).
        equity = float(_alpaca_broker_from_settings().account().equity)
        allocations.allocate_in_lane(
            conn, rec.id, capital=capital, actor='agent', account_equity=equity,
            allowed_stages=frozenset({Stage.PAPER.value, Stage.FORWARD_TESTED.value}),
            max_concurrent=max_concurrent)
    emit(ok({'strategy': name, 'capital': capital, 'prior_capital': prior_capital}))


def _run_quality_gate(repo_root: Path) -> str | None:
    """Run the FULL quality gate against ``repo_root``'s STAGED tree, returning the gated TREE SHA
    iff ALL of ``pytest -q``, ``ruff check .``, ``mypy algua``, and ``lint-imports`` exit 0 (else
    None); the saga binds ``commit_merge`` to that sha.

    This is the ``run_gate`` seam ``run_merge_back`` invokes against the ``--no-ff --no-commit``
    merge preview before it will commit + promote: a red gate aborts the merge with ``main``
    untouched. The gate runs HERMETICALLY in a throwaway worktree of the staged tree (body in the
    unprotected ``algua.operator.gate_runner``): the live checkout's held ``merge_back.lock``,
    real ``.env`` credentials, and dirty staged tree all break the suite's hermetic-environment
    assumptions when the gate runs in-place (proven by the first end-to-end factory drains)."""
    gate_runner = importlib.import_module("algua.operator.gate_runner")
    result: str | None = gate_runner.run_hermetic_quality_gate(repo_root)
    return result


@paper_app.command('merge-back')
@json_errors
def merge_back(
    branch: str = typer.Option(..., '--branch',
        help='the research candidate branch to merge back onto main'),
    strategy: str = typer.Option(..., '--strategy', help='the strategy authored on --branch'),
    universe: str = typer.Option(..., '--universe',
        help='PIT universe for the strict-agent promote gate (non-PIT fails closed)'),
    start: str = typer.Option(..., '--start', help='promote-window start (YYYY-MM-DD)'),
    end: str = typer.Option(..., '--end', help='promote-window end (YYYY-MM-DD)'),
    max_concurrent: int | None = typer.Option(None, '--max-concurrent',
        help='max concurrent paper-lane strategies admitted into the shared book '
             '(default: settings.paper_book_capacity / ALGUA_PAPER_BOOK_CAPACITY)'),
    actor: str = typer.Option('agent', '--actor', help='human | agent (audit label)'),
    demo: bool = typer.Option(False, '--demo',
        help='eval context: synthetic data provider (mutually exclusive with --snapshot)'),
    snapshot: str = typer.Option(None, '--snapshot',
        help='eval context: ingested bars snapshot id for evidence + promote'),
    fundamentals_snapshot: str = typer.Option(None, '--fundamentals-snapshot',
        help='eval context: fundamentals snapshot id (needs_fundamentals strategies)'),
    news_snapshot: str = typer.Option(None, '--news-snapshot',
        help='eval context: news snapshot id (needs_news strategies)'),
    delistings: str = typer.Option(None, '--delistings',
        help='eval context: delistings snapshot handle (survivorship-free realization)'),
    rank_by: str = typer.Option('mean_sharpe', '--rank-by',
        help='evidence sweep ranking: mean_sharpe | min_sharpe'),
    sweep_param: list[str] = typer.Option(None, '--sweep-param',
        help='evidence sweep grid KEY=v1,v2,... (repeatable) — the exact grid the scratch '
             'preview swept; re-run authoritatively post-merge to record measured breadth'),
) -> None:
    """Autonomous research-cycle merge-back (#485): turn a research candidate branch into an
    on-``main``, allocated paper strategy with no human merging the branch. One repo-global-locked
    cycle: preview-merge ``--branch`` (``--no-ff --no-commit``), run the FULL quality gate on the
    staged tree, commit only on green, run the metered strict-agent ``research promote`` (hard-wired
    to ``actor=agent`` — no relaxation flags), and on a PASS run the FIFO paper intake to allocate a
    book slice; a proven promote FAILURE reverts the merge, leaving ``main`` as it was.

    Terminal ``status`` is one of ``already_done`` (strategy already at ``paper``), ``gate_failed``
    (quality gate red — merge aborted, ``main`` untouched), ``promote_failed`` (gate green but
    promote rejected — merge reverted), or ``promoted_allocated`` (promoted + intake ran). A
    completed-but-not-promoted cycle (``gate_failed``/``promote_failed``) is NOT a command error —
    it emits ``ok`` with the honest status.

    MUTUAL EXCLUSION: like ``paper trade-tick``/``run-all`` (#316), ``merge-back`` mutates shared
    state (the working tree + the registry) and is mutually exclusive with the paper-account cycles
    BY OPERATOR DISCIPLINE when invoked directly — do not run a paper trade-tick / run-all
    concurrently with a direct merge-back invocation. Concurrent merge-back invocations are
    hard-serialized by a repo-global ``merge_back.lock`` flock; a second live invocation fails
    closed. The automated merge-back drainer (factory slice 3) invokes this command wrapped as
    ``algua operator lock-run -- algua paper merge-back ...``, which shares the SAME
    ``operator.lock`` the daily paper tick takes — turning that discipline into real kernel-enforced
    mutual exclusion for the automated path (see ``algua operator lock-run``)."""
    max_concurrent = _resolve_max_concurrent(max_concurrent)
    if max_concurrent <= 0:
        raise ValueError('--max-concurrent must be positive')
    actor_enum = Actor(actor)  # fail fast on a bad actor before any git mutation
    # Fail-fast eval-context preflight (GATE-2 #5): a typo'd --demo/--snapshot combo, rank_by, or
    # --sweep-param must die HERE — before the repo lock, the preview merge, and the ~9-minute
    # quality gate — not deep inside promote after a merge that then has to revert. Validator body
    # lives in the unprotected intake module (dynamic import, same style as the seams below).
    importlib.import_module("algua.registry.mergeback_intake").validate_transport_inputs(
        demo=demo, snapshot=snapshot, rank_by=rank_by,
        sweep_params=list(sweep_param) if sweep_param else None)
    settings = get_settings()
    # The lock must be scoped to the CHECKOUT it protects, NOT to ``db_path.parent`` (HIGH-4): two
    # invocations on the SAME working tree with different ALGUA_DB_PATH would otherwise take
    # DIFFERENT db-rooted locks and mutate the one shared tree concurrently. Anchor it at the repo's
    # own git dir (``git rev-parse --absolute-git-dir`` — the per-worktree ``.git``), resolved
    # BEFORE the lock. The git dir is outside the working tree, so the lock file never dirties
    # ``git status`` (which would fail the clean-checkout precondition).
    _here = Path(__file__).resolve().parent
    repo_root = Path(subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
        ['git', 'rev-parse', '--show-toplevel'],
        cwd=_here, capture_output=True, text=True, check=True).stdout.strip())
    git_dir = Path(subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
        ['git', 'rev-parse', '--absolute-git-dir'],
        cwd=_here, capture_output=True, text=True, check=True).stdout.strip())
    # Repo-global exclusive flock for the whole saga: a live concurrent cycle fails closed rather
    # than mutating the shared checkout under a second driver (kernel-released on death).
    with merge_back_lock(git_dir / 'merge_back.lock'):
        git = RealGitOps(repo_root)
        # The driver's OWN durable recovery journal (per-strategy JSONL beside the registry db); NOT
        # registry-domain state, so no schema bump. The CODEOWNERS text feeds the diff-policy
        # denylist (fail closed if it cannot be read).
        journal = JsonlJournal(settings.db_path.parent)
        codeowners_text = (repo_root / 'CODEOWNERS').read_text(encoding='utf-8')

        def stage_of(name: str) -> str | None:
            # Its OWN short-lived registry_conn (own tx) — the saga never holds one conn across the
            # whole cycle. Returns the raw lifecycle stage string run_merge_back dispatches on;
            # None = unregistered (the canonical factory-fresh state the intake chokepoint
            # registers post-merge).
            with registry_conn() as conn:
                try:
                    return SqliteStrategyRepository(conn).get(name).stage.value
                except StrategyNotFound:
                    return None

        def ensure_backtested(branch_tip: str, merge_sha: str, base_sha: str) -> str:
            # Authoritative-intake registration chokepoint (body in the unprotected
            # algua.registry.mergeback_intake; dynamic import mirrors the promote seam's style).
            intake_mod = importlib.import_module("algua.registry.mergeback_intake")
            with registry_conn() as conn:
                return intake_mod.ensure_backtested(
                    conn, strategy=strategy, branch=branch, branch_tip=branch_tip,
                    merge_sha=merge_sha, base_sha=base_sha)

        def produce_evidence(ensure_status: str, branch_tip: str) -> str:
            # Authoritative evidence reproduction: the REAL sweep/backtest task bodies are injected.
            # sweep_task lives in algua.evaluation.sweep_run and run_backtest_task lives in
            # algua.evaluation.backtest_run (neither is a cli sibling), so both arrive via a legal
            # static import.
            # Strict-agent pinning: windows/holdout_frac stay the task defaults;
            # assume_terminal_last_close stays False.
            intake_mod = importlib.import_module("algua.registry.mergeback_intake")
            params = list(sweep_param) if sweep_param else None
            return intake_mod.produce_evidence(
                strategy=strategy, branch_tip=branch_tip, ensure_status=ensure_status,
                sweep_params=params, conn_factory=registry_conn,
                # The FULL transported data context — bound into the marker's recipe hash so a
                # resumed attempt with a drifted context fails closed (GATE-2 #4).
                eval_context={
                    "demo": demo, "snapshot": snapshot,
                    "fundamentals_snapshot": fundamentals_snapshot,
                    "news_snapshot": news_snapshot, "delistings": delistings,
                    "rank_by": rank_by, "universe": universe, "start": start, "end": end},
                sweep_fn=lambda: sweep_task(
                    strategy, start=start, end=end, demo=demo, snapshot=snapshot,
                    universe=universe, param=params, rank_by=rank_by,
                    fundamentals_snapshot=fundamentals_snapshot, news_snapshot=news_snapshot,
                    delistings=delistings),
                backtest_fn=lambda: run_backtest_task(
                    strategy, start=start, end=end, demo=demo, snapshot=snapshot,
                    universe=universe, fundamentals_snapshot=fundamentals_snapshot,
                    news_snapshot=news_snapshot, delistings=delistings))

        def promote(attempt_token: str) -> object:
            # promote_task lives in algua.registry.promote_run (not a cli sibling), so it arrives
            # via a legal static `from algua.registry.promote_run import promote_task` at module
            # scope — the cli-independence import-linter contract (#165) forbids a
            # paper_cmd<->research_cmd sibling edge, but neither command module imports the other
            # here; both import the same shared registry module. Strict-agent inputs ONLY — no
            # relaxation flags reach the seam, so a human-only relaxation is impossible by
            # construction. The per-attempt ``attempt_token`` is stamped on the gate row so the
            # driver reads the outcome authoritatively (finding #5). promote_task opens+closes its
            # OWN registry_conn (per its contract).
            return promote_task(
                name=strategy, universe=universe, start=start, end=end,
                demo=demo, snapshot=snapshot, fundamentals_snapshot=fundamentals_snapshot,
                news_snapshot=news_snapshot, delistings=delistings,
                actor='agent', attempt_token=attempt_token)

        def passing_gate_by_token(attempt_token: str) -> int | None:
            with registry_conn() as conn:
                return SqliteStrategyRepository(conn).passing_gate_by_token(strategy, attempt_token)

        def gate_exists_by_token(attempt_token: str) -> bool:
            # Crash-idempotency read (#485 HIGH-2): does ANY gate row (pass OR fail) already bear
            # this attempt's token? If so, a prior crashed attempt already ran the metered promote —
            # the driver must NOT re-invoke it (double holdout burn / unique-index late-fail).
            with registry_conn() as conn:
                return SqliteStrategyRepository(conn).gate_exists_by_token(strategy, attempt_token)

        def intake() -> dict:
            # Read paper equity READ-ONLY BEFORE opening the intake txn (no trading), then run the
            # shared FIFO admit over its OWN short-lived registry_conn.
            equity = float(_alpaca_broker_from_settings().account().equity)
            with registry_conn() as conn:
                return run_intake(conn, equity=equity, max_concurrent=max_concurrent,
                                   actor=actor_enum)

        def target_allocated(name: str) -> bool:
            # Did intake allocate THIS strategy (now paper WITH an active allocation)? FIFO intake
            # may admit an older queued candidate ahead of ours, so outcome is target-verified,
            # never inferred from "intake admitted something".
            with registry_conn() as conn:
                repo = SqliteStrategyRepository(conn)
                if repo.get(name).stage is not Stage.PAPER:
                    return False
                return active_allocation(conn, repo.get(name).id) is not None

        def audit_log(event: dict) -> None:
            # Every autonomous push/revert is attributable after the fact (the accountability record
            # the bypassed local push hook never produced).
            with registry_conn() as conn:
                audit_append(conn, actor='merge_back_driver',
                             action=str(event.get('event', 'merge_back')),
                             reason=json.dumps(event, sort_keys=True), strategy=strategy)

        result = run_merge_back(
            git=git, journal=journal, strategy=strategy, branch=branch,
            codeowners_text=codeowners_text,
            stage_of=stage_of, run_gate=lambda: _run_quality_gate(repo_root),
            ensure_backtested=ensure_backtested, produce_evidence=produce_evidence,
            promote=promote, passing_gate_by_token=passing_gate_by_token,
            gate_exists_by_token=gate_exists_by_token,
            intake=intake, target_allocated=target_allocated, audit_log=audit_log)
    # A completed cycle is a successful invocation regardless of the promote outcome: emit ok() with
    # the terminal status (gate_failed/promote_failed/diff_policy_rejected carry honest statuses).
    emit(ok({
        'strategy': strategy, 'branch': branch, 'status': result.status,
        'merged': result.merged, 'reverted': result.reverted,
        'promoted': result.promoted, 'intake': result.intake,
        'attempt_token': result.attempt_token, 'gate_id': result.gate_id,
    }))


def _still_paper_allocated(conn, name: str) -> bool:
    """True iff `name` is still a paper-lane book tenant (Stage.PAPER/FORWARD_TESTED AND an active
    allocation). Re-read at submit time so a mid-cycle lane-crossing transition (which atomically
    revokes the slice, #497) halts further submits instead of trading a stale capital base — the
    paper mirror of live `_still_live_allocated` (#281)."""
    rec = SqliteStrategyRepository(conn).get(name)
    return (rec.stage in (Stage.PAPER, Stage.FORWARD_TESTED)
            and active_allocation(conn, rec.id) is not None)


def _run_paper_strategy_tick(  # noqa: PLR0913
    conn, name: str, strategy, rec, broker, provider, max_drawdown,
    tick_ts, clock_source, acct, *, cancel=None, reserve_buy=None,
    start: str, end: str, snapshot_id: str | None = None,
) -> dict:
    """ONE strategy's multi-tenant paper tick: NAV-snapshot sizing (#314), crash-safe ledger
    recording, breach trip + scoped flatten, tick-snapshot persistence (equity = per-strategy NAV).
    Returns ok({...}), or an {"ok": False, ...} marker on TickHalted/RiskBreach so run-all can
    surface siblings on a breach (#316b, live #270). Caller reconciles BEFORE calling this.

    Fault-isolation boundary (#374 GATE-2): SETUP (allocation, identity, hook wiring) runs before
    any broker/ledger side effect, so a setup failure is wrapped ``StrategySetupError`` for run-all
    to isolate — except a :data:`SYSTEMIC_SETUP_EXCEPTIONS` member, which propagates raw (a
    shared-infra fault is book-wide, not this tenant's). Everything from ``run_tick`` onward is
    unwrapped: any exception escaping there is book-integrity-critical and aborts the cycle."""
    try:
        alloc = active_allocation(conn, rec.id)
        if alloc is None:
            raise ValueError(f"{name} has no paper allocation")
        allocation = float(alloc["capital"])
        identity = compute_artifact_hashes(name)

        # #559: bind this tick to the GATED universe (never the CONFIG template); a missing gate
        # row raises -> StrategySetupError; a legacy row (universe_name NULL) falls back to CONFIG.
        resolved_universe, universe_source = resolve_operational_universe(
            conn, get_settings().data_dir, name, strategy.universe)
        if universe_source == SOURCE_CONFIG_LEGACY:
            log.warning("universe_binding_config_legacy", extra={"fields": {
                "strategy": name, "lane": "paper",
                "note": "newest passing gate row has no universe_name (pre-#559); ticking on "
                        "CONFIG.universe — re-run research promote to bind the gate universe"}})
        if resolved_universe != strategy.universe:
            strategy = replace(
                strategy,
                config=strategy.config.model_copy(update={"universe": resolved_universe}))

        # coids THIS tick's before_submit freshly inserted are safe to retract on a noop; a
        # pre-existing NULL row may be a crash-orphaned real order and MUST be preserved (#311).
        freshly_recorded: set[str] = set()

        def _before_submit(intent: OrderIntent, coid: str | None) -> None:
            if coid is not None and record_paper_venue_order(
                conn, name, intent.symbol, intent.side.value, None, coid, strategy_id=rec.id
            ):
                freshly_recorded.add(coid)

        def _on_submitted(rec_: SubmittedOrder) -> None:
            # A real order landed: backfill its broker id and drop it from the fresh set so on_noop
            # can never retract a resolved order (defense beyond submit_sized's noop/POST split).
            backfill_paper_venue_broker_order_id(conn, rec_.client_order_id, rec_.order_id)
            freshly_recorded.discard(rec_.client_order_id)

        def _on_noop(intent: OrderIntent, coid: str | None) -> None:
            # submit_sized reported noop/skipped -> no POST. Retract the phantom intent ONLY if THIS
            # tick freshly recorded it (else it may be a prior run's real, crash-orphaned order).
            if coid is not None and coid in freshly_recorded:
                delete_paper_venue_order(conn, coid)
                freshly_recorded.discard(coid)

        hooks = TickHooks(
            client_order_id_for=client_order_id,
            before_submit=_before_submit,
            on_submitted=_on_submitted,
            on_noop=_on_noop,
            should_halt=lambda: (kill_switch.is_tripped(conn, name)
                                 or global_halt.is_engaged(conn)
                                 or not _still_paper_allocated(conn, name)),
            cancel=cancel,
            reserve_buy=reserve_buy,
            peak_equity=get_peak_equity(conn, name),
            live_snapshot=lambda bars: build_paper_sizing_snapshot(
                conn, name, allocation, bars, strategy.universe),
            live_positions=lambda: paper_believed_positions(conn, name),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except StrategySetupError:
        # A nested setup helper that already classified itself: propagate as-is so its original
        # ``code`` survives instead of being double-wrapped (defense-in-depth; unreachable today).
        raise
    except SYSTEMIC_SETUP_EXCEPTIONS:
        # A shared-infra fault (e.g. sqlite3.Error) is book-wide, not this tenant's: propagate RAW
        # so the db_unavailable/retryable signal survives (#374 GATE-2) instead of misclassifying.
        raise
    except Exception as exc:  # noqa: BLE001 - pre-side-effect setup fault: isolate ONE tenant
        raise StrategySetupError(name, exc) from exc
    try:
        result = run_tick(strategy, broker, provider, utc(start), utc(end),
                          hooks=hooks, max_drawdown=max_drawdown)
    except TickHalted as exc:
        audit_append(conn, actor="system", action="trade_tick_halted", reason=str(exc),
                     strategy=name)
        log.info("tick_halted", extra={"fields": {"strategy": name, "lane": "paper"}})
        return {"ok": False, "strategy": name, **breach_payload(str(exc), strategy=name,
                                                                halted=True)}
    except RiskBreach as exc:
        trip_for_breach(conn, name, exc)
        log.error("breach", extra={"fields": {"strategy": name, "lane": "paper",
                                              "tick_ts": str(tick_ts), "kind": exc.kind}},
                  exc_info=True)
        if exc.is_dark_feed:
            # DARK BAR FEED, broker still alive (#452 HIGH#3): a stale mark means risk state is
            # untrustworthy, not that the position is losing money, so HALT (systemic — one shared
            # provider) and PRESERVE positions instead of flattening blind at unknown prices.
            global_halt.engage(conn, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="paper_mark_freshness_halt",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            return {"ok": False, "strategy": name,
                    **breach_payload(exc.detail, strategy=name, kind=exc.kind, halted=True,
                                     global_halt="set", liquidation_submitted=False)}
        # ECONOMIC / integrity breach: trip + scoped flatten (run-all: per-strategy, never a
        # sibling's orders; account-wide fallback for trade-tick). Offsets every belief (#336).
        res = flatten_strategy(
            conn, broker, name, LedgerKind.PAPER, lane="paper", strategy_id=rec.id,
            cancel=cancel if cancel is not None else broker.cancel_open_orders,
            ingest=lambda: _ingest_paper_venue(conn, broker, tick_clock(broker.clock)[0]),
        )
        payload = breach_payload(exc.detail, kind=exc.kind,
                                 liquidation_submitted=res.n_offsets > 0,
                                 offsets_submitted=res.n_offsets)
        if res.flatten_error is not None:
            payload["flatten_error"] = res.flatten_error
        return {"ok": False, "strategy": name, **payload}
    except LiveSizingError as exc:
        # Mark-data problems now raise RiskBreach and are HALTED (no flatten) above; only a RESIDUAL
        # non-wall sizing error reaches here and skips this strategy for the cycle without trading.
        audit_append(conn, actor="system", action="paper_sizing_skipped",
                     reason=str(exc), strategy=name)
        return ok({"strategy": name, "traded": False, "skipped": str(exc)})

    if result.peak_equity is not None:
        update_peak_equity(conn, name, result.peak_equity)
        record_tick_snapshot(
            conn, name, tick_ts=tick_ts,
            decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
            equity=result.equity, peak_equity=result.peak_equity,
            positions=result.positions_before, n_submitted=len(result.submitted),
            reconcile_ok=result.reconcile_ok, lane="paper", strategy_id=rec.id,
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=identity.dependency_hash, account_id=acct.account_id,
            cash=acct.cash, clock_source=clock_source, snapshot_id=snapshot_id)
    audit_append(conn, actor="agent", action="trade_tick",
                 reason=f"{len(result.submitted)} orders submitted", strategy=name)
    return ok({
        "strategy": name,
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "target_weights": result.target_weights, "positions_before": result.positions_before,
        "submitted": result.submitted, "reconcile_ok": result.reconcile_ok,
        "realized_gross": result.realized_gross})


@paper_app.command("trade-tick")
@json_errors
def trade_tick(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    max_drawdown: float | None = typer.Option(None, "--max-drawdown",
                                       help="per-strategy drawdown breaker fraction; omit for the default-ON bound"),  # noqa: E501
    disable_drawdown_breaker: bool = typer.Option(
        False, "--disable-drawdown-breaker",
        help="HUMAN-ONLY emergency: turn the drawdown breaker fully OFF (audited)"),
) -> None:
    """Run ONE wall-clock tick against the PAPER venue: submit Alpaca market-order deltas toward
    the strategy's target. Each accepted order persists immediately; a drawdown/exposure/reconcile
    breach trips the kill-switch and flattens. Never trades live (the broker refuses a live URL)."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    max_drawdown = resolve_drawdown_breaker(max_drawdown, disable_drawdown_breaker)
    start, end = resolve_wall_clock_window(start, end)
    configure_logging()
    counters = CycleCounters()
    # One correlation id per wall-clock tick; golden_signals flushes in `finally` so the rollup
    # survives even when the cycle fails before the tick (ingest/reconcile) (#346).
    with correlation_context():
        log.info("cycle_start",
                 extra={"fields": {"lane": "paper", "strategy": name, "snapshot": snapshot}})
        try:
            with registry_conn() as conn:
                if disable_drawdown_breaker:
                    audit_append(conn, actor="human", action="drawdown_breaker_disabled",
                                 reason="paper trade-tick invoked with --disable-drawdown-breaker",
                                 strategy=name)
                strategy, rec = load_gated_strategy(conn, name, "trade-tick")
                broker = _alpaca_broker_from_settings()
                provider = _select_provider(False, snapshot)
                acct = broker.account()
                tick_ts, clock_source = tick_clock(broker.clock)
                try:
                    _ingest_paper_venue(conn, broker, tick_ts)
                    # #312: resolve any crash-stranded NULL-broker_order_id row BEFORE reconcile,
                    # so its now-attributed fill no longer reads as drift.
                    _recover_stranded(conn, broker, LedgerKind.PAPER)
                except Exception as exc:   # fail closed on ANY ingest/transport error
                    audit_append(conn, actor="system", action="venue_ingest_failed",
                                 reason=str(exc), strategy=name)
                    log.error("venue_ingest_failed",
                              extra={"fields": {"strategy": name, "lane": "paper"}}, exc_info=True)
                    emit(breach_payload(str(exc), strategy=name, kind="venue_ingest_failed"))
                    raise typer.Exit(1) from exc

                # Account-wide reconcile (multi-tenant): attributed_paper_net vs the broker book,
                # grace window. halt -> global halt; not clean -> defer (no trade); clean -> tick.
                cycle = paper_reconcile.next_cycle(conn)
                recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)
                if recon.halt:
                    counters.reconcile_halted += 1
                    log.error("reconcile_halt",
                              extra={"fields": {"strategy": name, "lane": "paper",
                                                "mismatches": recon.mismatches}})
                    global_halt.engage(conn, reason=f"paper reconcile drift {recon.mismatches}",
                                       actor="system")
                    emit({"ok": False, "strategy": name, "deferred": True, "halted": True,
                          "reconcile": recon.mismatches})
                    raise typer.Exit(1)
                if not recon.clean:
                    counters.reconcile_deferred += 1
                    log.info("reconcile_deferred",
                             extra={"fields": {"strategy": name, "lane": "paper"}})
                    audit_append(conn, actor="system", action="trade_tick_deferred",
                                 reason="reconcile pending", strategy=name)
                    emit(ok({"strategy": name, "traded": False, "deferred": True,
                             "reconcile": recon.mismatches}))
                    return

                try:
                    out = _run_paper_strategy_tick(
                        conn, name, strategy, rec, broker, provider, max_drawdown,
                        tick_ts, clock_source, acct, start=start, end=end,
                        snapshot_id=snapshot)
                except StrategySetupError as exc:
                    # SINGLE-strategy path has no siblings to isolate: unwrap the per-tenant
                    # StrategySetupError so json_errors renders the real cause's message/code (#374)
                    raise exc.cause from exc
                counters.ticks += 1
                if out.get("ok") is False:
                    counters.breaches += 1
                    if out.get("flatten_error") is not None:
                        counters.flatten_failures += 1
            emit(out)
            if not out.get("ok", False):
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception:
            log.error("cycle_failed", extra={"fields": {"strategy": name, "lane": "paper"}},
                      exc_info=True)
            raise
        finally:
            log.info("golden_signals", extra={"fields": counters.as_fields()})


@paper_app.command("run-all")
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
             "--snapshot/--refresh is required; --end is derived (not accepted) under --refresh "
             "(#556)"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    max_drawdown: float | None = typer.Option(
        None, "--max-drawdown",
        help="halt + flatten a strategy if equity falls this fraction below its peak; "
             "omit for the default-ON bound"),
    disable_drawdown_breaker: bool = typer.Option(
        False, "--disable-drawdown-breaker",
        help="HUMAN-ONLY emergency: turn the drawdown breaker fully OFF (audited)"),
) -> None:
    """One sequenced multi-tenant cycle over ALL paper-lane strategies: ingest venue fills,
    reconcile the account against the paper broker, then tick each strategy (scoped cancel on a
    breach so one strategy never cancels a sibling's resting orders). Trades only when the account
    reconciles clean; a persistent unexplained drift engages the global halt. A simple whole-account
    buying-power pool caps the aggregate of this cycle's buys (NO book-level #389 risk here).

    Strategy selection is ``stage IN ('paper','forward_tested')`` — same admission set as
    ``load_gated_strategy``/``trade-tick``: forward_tested keeps ticking to keep its live-wall
    certificate fresh (#124). "paper" in the name is the LANE, not a stage filter.

    CONCURRENCY: shares the mutable paper account with ``trade-tick``/liquidation — not safe run
    concurrently (double-ingest, reconcile races, BP over-commit); serialized by operator
    discipline only (an advisory lock is a filed follow-up)."""
    if bool(snapshot) == refresh:
        raise ValueError("pass exactly one of --snapshot <id> or --refresh")
    if refresh and (start is not None or end is not None):
        raise ValueError("--refresh derives the cycle window (end = today, start from the "
                         "strategies' history need); --start/--end are not accepted")
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    max_drawdown = resolve_drawdown_breaker(max_drawdown, disable_drawdown_breaker)
    start, end = resolve_wall_clock_window(start, end)
    configure_logging()
    counters = CycleCounters()
    # One correlation id per cycle; golden_signals flushes in `finally` so the rollup survives even
    # when the cycle fails before/around the strategy loop (#346).
    with correlation_context():
        log.info("cycle_start",
                 extra={"fields": {"lane": "paper", "snapshot": snapshot or "refresh"}})
        try:
            with registry_conn() as conn:
                if disable_drawdown_breaker:
                    audit_append(conn, actor="human", action="drawdown_breaker_disabled",
                                 reason="paper run-all invoked with --disable-drawdown-breaker",
                                 strategy=None)
                repo = SqliteStrategyRepository(conn)
                # Both paper-lane stages tick (parity with load_gated_strategy/trade-tick),
                # preserving each list's insertion order (#124).
                paper = repo.list_strategies(Stage.PAPER) + repo.list_strategies(
                    Stage.FORWARD_TESTED)
                # A recovery/demotion re-entrant (dormant->paper, live->paper) has no allocation and
                # is always FLAT: SKIP it, don't tick it (would abort the whole cycle, #317 #2).
                skipped_unallocated = [prec.name for prec in paper
                                       if active_allocation(conn, prec.id) is None]
                tickable = [prec for prec in paper if prec.name not in set(skipped_unallocated)]
                if not paper:
                    emit(ok({"strategies": [], "skipped_unallocated": skipped_unallocated,
                             "note": "no paper-lane strategies"}))
                    return
                if global_halt.is_engaged(conn):
                    emit({**breach_payload("global halt engaged", halted=True),
                          "skipped_unallocated": skipped_unallocated})
                    raise typer.Exit(1)
                broker = _alpaca_broker_from_settings()
                acct = broker.account()
                tick_ts, clock_source = tick_clock(broker.clock)
                # ingest fills + recover crash-stranded rows BEFORE reconcile; fail closed on any
                # transport/venue error so a partial ingest can never read as reconcile drift.
                try:
                    _ingest_paper_venue(conn, broker, tick_ts)
                    _recover_stranded(conn, broker, LedgerKind.PAPER)
                except Exception as exc:  # fail closed on ANY ingest/transport error
                    audit_append(conn, actor="system", action="venue_ingest_failed",
                                 reason=str(exc), strategy=None)
                    log.error("venue_ingest_failed",
                              extra={"fields": {"lane": "paper"}}, exc_info=True)
                    emit({**breach_payload(str(exc), kind="venue_ingest_failed"),
                          "skipped_unallocated": skipped_unallocated})
                    raise typer.Exit(1) from exc

                results: list[dict] = []
                # Lane bars refresh (#556): AFTER fill ingest (so ledger-held symbols are current),
                # BEFORE the reconcile. Broker positions are read TWICE on purpose: this first
                # read only feeds symbol DISCOVERY (orphan/residual marks get fetched too); the
                # reconcile below takes its own fresh read AFTER the network round-trip, so a fill
                # that lands during the refresh is judged by the reconcile (defer), never traded
                # against a stale sample. A per-tenant plan fault is isolated like any setup
                # error; EVERY tenant failing to plan is a failed cycle, not a benign no-op; a
                # refresh failure fails the WHOLE cycle closed — the operator alerts, leaves the
                # session marker unwritten, and the next fire retries. Never fall back to an older
                # snapshot. The derived window (start from the deepest history floor) is the one
                # the ticks read too.
                snapshot_info: dict = {"id": snapshot, "refreshed": False,
                                       "start": start, "end": end}
                if refresh:
                    plan = build_cycle_plan(
                        conn, names=[prec.name for prec in tickable], kind=LedgerKind.PAPER,
                        data_dir=get_settings().data_dir)
                    results.extend(plan.skipped)
                    if tickable and not plan.universes:
                        audit_append(conn, actor="system", action="cycle_plan_failed",
                                     reason=f"{len(plan.skipped)} tenant(s) failed planning",
                                     strategy=None)
                        emit({"ok": False, "code": "cycle_plan_failed",
                              "error": "every tickable strategy failed the cycle plan",
                              "strategies": results,
                              "skipped_unallocated": skipped_unallocated})
                        raise typer.Exit(1)
                    tickable = [prec for prec in tickable if prec.name in plan.universes]
                    if tickable:
                        try:
                            snapshot_info = refresh_lane_snapshot(
                                lane_symbols(plan, _paper_broker_net(broker)), end=end,
                                min_rows=plan.min_rows, kind=LedgerKind.PAPER)
                        except Exception as exc:  # noqa: BLE001 — any refresh fault fails closed
                            audit_append(conn, actor="system", action="bars_refresh_failed",
                                         reason=str(exc), strategy=None)
                            log.error("bars_refresh_failed",
                                      extra={"fields": {"lane": "paper"}}, exc_info=True)
                            emit({"ok": False, "code": "refresh_failed", "error": str(exc),
                                  "strategies": results,
                                  "skipped_unallocated": skipped_unallocated})
                            raise typer.Exit(1) from exc
                        snapshot, start = snapshot_info["id"], snapshot_info["start"]
                        log.info("bars_refreshed",
                                 extra={"fields": {"lane": "paper", **snapshot_info}})
                provider = _select_provider(False, snapshot)

                # Account-wide reconcile (multi-tenant): attributed_paper_net vs the broker book.
                # halt -> global halt; not clean -> defer the whole cycle (no trade); clean -> tick.
                cycle = paper_reconcile.next_cycle(conn)
                recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)
                recon_payload = {
                    "cycle": cycle,
                    "clean": recon.clean,
                    "halt": recon.halt,
                    "mismatches": recon.mismatches,
                }
                if recon.halt:
                    counters.reconcile_halted += 1
                    log.error("reconcile_halt",
                              extra={"fields": {"lane": "paper",
                                                "mismatches": recon.mismatches}})
                    global_halt.engage(conn, reason=f"paper reconcile drift {recon.mismatches}",
                                       actor="system")
                    emit({"ok": False, "deferred": True, "halted": True,
                          "reconcile": recon_payload,
                          "skipped_unallocated": skipped_unallocated})
                    raise typer.Exit(1)
                if not recon.clean:
                    counters.reconcile_deferred += 1
                    log.info("reconcile_deferred", extra={"fields": {"lane": "paper"}})
                    emit(ok({"strategies": [], "deferred": True, "reconcile": recon_payload,
                             "skipped_unallocated": skipped_unallocated,
                             "note": "reconcile pending; deferring trades this cycle"}))
                    return

                # Whole-account buying-power pool: this cycle's buys can never exceed it; each
                # strategy's reserve trims against the running pool, and a trim is audited.
                pool = {"available": float(acct.buying_power)}

                def _paper_reserve_for(strategy_name):
                    def _reserve(symbol: str, notional: float) -> float:
                        grant = min(notional, max(0.0, pool["available"]))
                        # Debit what submit_sized ACTUALLY posts, not the raw grant: a
                        # below-MIN_NOTIONAL grant posts nothing and would phantom-starve a sibling.
                        pool["available"] -= posted_notional(grant)
                        if grant < notional:  # the POOL bound this buy -> audit the shortfall
                            audit_append(conn, actor="system", action="paper_reserve_trim",
                                         reason=f"{symbol} {notional}->{grant}",
                                         strategy=strategy_name)
                        return grant
                    return _reserve

                breached = False
                for prec in tickable:
                    name = prec.name
                    # Per-strategy fault isolation (#374/GATE-2): ONLY a pre-side-effect setup fault
                    # is contained here (siblings still tick); any other exception is
                    # book-integrity-critical and propagates RAW to abort the cycle.
                    try:
                        # load_gated_strategy is also pre-side-effect setup: a load/gate-token
                        # failure here isolates this tenant, so wrap it as StrategySetupError too.
                        try:
                            strategy, rec = load_gated_strategy(conn, name, "paper run-all")
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except StrategySetupError:
                            raise
                        except global_halt.GlobalHaltActive:
                            # Book-wide, not this tenant's fault: propagate RAW to abort the cycle —
                            # never demote to a StrategySetupError that ticks siblings under a halt.
                            raise
                        except SYSTEMIC_SETUP_EXCEPTIONS:
                            # A shared-infra fault is book-wide too: propagate RAW so the top-level
                            # db_unavailable/retryable signal survives (#374 GATE-2).
                            raise
                        except Exception as load_exc:  # noqa: BLE001 - pre-side-effect setup fault
                            raise StrategySetupError(name, load_exc) from load_exc
                        out = _run_paper_strategy_tick(
                            conn, name, strategy, rec, broker, provider, max_drawdown,
                            tick_ts, clock_source, acct,
                            reserve_buy=_paper_reserve_for(name),
                            cancel=lambda n=name: _paper_scoped_cancel(conn, broker, n),
                            start=start, end=end, snapshot_id=snapshot)
                    except StrategySetupError as exc:
                        log.error("strategy_setup_error",
                                  extra={"fields": {"lane": "paper", "strategy": name,
                                                    "code": exc.code}},
                                  exc_info=True)
                        audit_append(conn, actor="system", action="strategy_setup_error",
                                     reason=exc.code, strategy=name)
                        results.append({"ok": False, "strategy": name, "kind": "setup_error",
                                        "error": exc.code})
                        counters.setup_errors += 1
                        continue
                    results.append(out)
                    counters.ticks += 1
                    if out.get("ok") is False:  # breach/halt marker: stop, keep prior results
                        counters.breaches += 1
                        if out.get("flatten_error") is not None:
                            counters.flatten_failures += 1
                        breached = True
                        break
            envelope = {"reconcile": recon_payload, "strategies": results,
                        "skipped_unallocated": skipped_unallocated,
                        "setup_errors": [r for r in results if r.get("kind") == "setup_error"],
                        "snapshot": snapshot_info}
            if breached:
                # A breach (already tripped + scoped-flattened): surface it AND every sibling
                # ticked before it in one envelope, then exit non-zero (#270).
                emit({"ok": False, **envelope})
                raise typer.Exit(1)
            emit(ok(envelope))
        except typer.Exit:
            raise
        except Exception:
            log.error("cycle_failed", extra={"fields": {"lane": "paper"}}, exc_info=True)
            raise
        finally:
            log.info("golden_signals", extra={"fields": counters.as_fields()})


@paper_app.command("promote")
@json_errors
def promote(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
    actor_signature: str = typer.Option(
        None, "--actor-signature",
        help="path to the SSH signature over the printed human-actor challenge (#329). Required to "
             "authenticate --actor human: a bare --actor human unlocks NO threshold relaxation — "
             "run once without this to print a challenge, sign it with your enrolled "
             "algua-human-actor key (ssh-keygen -Y sign -n algua-human-actor), then re-run with "
             "--actor-signature."),
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
    forward_sharpe_confidence: float = typer.Option(
        FORWARD_SHARPE_CONFIDENCE, "--forward-sharpe-confidence",
        help="one-sided confidence at which the realized-Sharpe lower bound must clear the "
             "performance bar (raising is stricter; lowering is human-only)"),
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
        forward_sharpe_confidence=forward_sharpe_confidence,
    )
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)  # StrategyNotFound -> JSON error before any work
        # AUTHENTICATE the human actor (#329) BEFORE the relaxation guard is even consulted. A bare
        # `--actor human` is forgeable, so asserting a human actor here requires an SSH signature
        # (namespace algua-human-actor) over a fresh single-use challenge binding this command +
        # strategy + RECOMPUTED artifact identity + the FULL ForwardGateCriteria (all 8 thresholds).
        # No signature => a challenge is issued+printed and NOTHING runs. A declared agent is
        # returned unchanged (the relaxation guard refuses its relaxations exactly as before).
        actor_enum = authenticate_actor(
            conn, command="paper promote", name=name, rec=rec,
            stage_to=Stage.FORWARD_TESTED.value, declared_actor=actor_enum,
            actor_signature=actor_signature,
            run_context=canonical_run_context({
                "min_observations": min_observations, "min_coverage": min_coverage,
                "degradation_factor": degradation_factor, "sharpe_floor": sharpe_floor,
                "min_vol": min_vol, "max_drawdown": max_drawdown, "max_staleness": max_staleness,
                "forward_sharpe_confidence": forward_sharpe_confidence,
            }),
        )
        # PREFLIGHT: actor legality + relaxations-need-human + stage legality. Refuses here,
        # before the broker is even constructed (TransitionError is a ValueError -> JSON error).
        forward_promotion_preflight(repo, name, actor=actor_enum, criteria=criteria)
        broker = _alpaca_broker_from_settings()
        outcome = run_forward_gate(
            repo, conn, name=name, actor=actor_enum, criteria=criteria,
            calendar=get_calendar(), now=datetime.now(UTC),
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
        "n_prior_forward_looks": outcome.assembled.n_prior_forward_looks,
    }
    # Re-sync the kb doc to the (possibly) new stage (#331): best-effort, out-of-transaction —
    # the `with registry_conn()` block above has already committed and closed.
    sync_kb_doc(name)
    # Pass mirrors research_cmd.promote's success envelope; a fail still emits the full
    # decision payload (the evaluation row was recorded) but carries the repo-wide exit-1
    # discriminator ("ok": false, see cli._common.ok) and exits non-zero.
    emit(ok(payload) if outcome.decision.passed else {"ok": False, **payload})
    if not outcome.decision.passed:
        raise typer.Exit(1)


@paper_app.command("flatten")
@json_errors
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: close this strategy's believed paper positions and trip its kill-switch.
    The offset loop iterates paper_believed_positions (strategy-attributed paper_venue_fills),
    so sibling positions on the shared account are never touched."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching a switch (#259)
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
        kill_switch.trip(conn, name, reason="flatten", actor=actor_enum.value)
        audit_append(conn, actor=actor_enum.value, action="flatten",
                     reason="manual flatten", strategy=name)
        # Account-wide cancel; ingest fills up to the broker clock, then offset every believed
        # position — single-sourced in the execution layer (#336). Fails SAFE: any liquidation
        # error is captured to res.flatten_error (never an unstructured traceback).
        res = flatten_strategy(
            conn, broker, name, LedgerKind.PAPER, lane="paper", strategy_id=rec.id,
            cancel=broker.cancel_open_orders,
            ingest=lambda: _ingest_paper_venue(conn, broker, tick_clock(broker.clock)[0]),
        )
        if res.flatten_error is not None:
            emit(breach_payload(res.flatten_error, strategy=name, liquidation_submitted=False,
                                offsets_submitted=res.n_offsets))
            raise typer.Exit(1)
    # liquidation_submitted reflects whether any offset order ACTUALLY went out (GATE-2 HIGH): a
    # strategy already flat (no believed positions) submits none, so report False rather than imply
    # a liquidation that never happened. Accepted offset fills land async (may be next open).
    emit(ok({"strategy": name, "kill_switch": "tripped",
             "liquidation_submitted": res.n_offsets > 0, "offsets_submitted": res.n_offsets}))


@paper_app.command("halt-all")
@json_errors
def halt_all(
    reason: str = typer.Option(..., "--reason", help="why the whole account is being halted"),
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """ACCOUNT-WIDE emergency: engage the global halt and flatten the ENTIRE Alpaca account."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before engaging the halt
    with registry_conn() as conn:
        broker = _alpaca_broker_from_settings()
        # Engage first (fail-safe): all trading is stopped even if the close call then fails.
        global_halt.engage(conn, reason=reason, actor=actor_enum.value)
        audit_append(conn, actor=actor_enum.value, action="halt_all", reason=reason, strategy=None)
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
@json_errors
def resume_all(
    # Default 'agent' to match the sibling halt commands (kill/flatten/halt-all): resume-all is
    # not enforced human-only and an agent can legitimately invoke it, so a 'human' default would
    # mislabel the (non-load-bearing) audit row when an agent uses the default (#272).
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Clear the global halt and re-base every strategy's drawdown peak (the account was flattened
    to cash). Per-strategy kill-switches are left untouched."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before touching the halt
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
                cursor = fill_cursor(conn, LedgerKind.LIVE)
                ingest_activities(conn, broker.account_activities(after=cursor), LedgerKind.LIVE)
        not_flat = [
            r["name"] for r in live_rows if believed_positions(conn, r["name"], LedgerKind.LIVE)
        ]
        if was_set:
            audit_append(conn, actor=actor_enum.value, action="resume_all",
                         reason="clear global halt; re-base all drawdown peaks", strategy=None)
            # Re-base peaks first, clear the halt LAST so the un-halt is the final write (#109).
            # Every table a resumed account can re-trip through — paper, live NAV, and the
            # account-wide book peak — is the `rebase_all_peaks` policy in algua.execution.peaks.
            rebase_all_peaks(conn)
            global_halt.clear(conn)
    result: dict = {"global_halt": "reset" if was_set else "not_set"}
    if not_flat:
        result["live_not_flat"] = not_flat
        result["warning"] = (
            "the above live strategies are not flat; re-flatten each before resuming individually"
        )
    emit(ok(result))
