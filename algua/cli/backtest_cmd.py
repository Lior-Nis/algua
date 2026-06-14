from __future__ import annotations

import json
from collections.abc import Callable

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.engine import run as run_backtest
from algua.backtest.sweep import SweepResult, parse_grid, sweep
from algua.backtest.walkforward import walk_forward
from algua.cli._common import (
    ok,
    registry_conn,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.tracking.mlflow_tracker import log_backtest, log_sweep, log_walk_forward

backtest_app = typer.Typer(help="Run backtests", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


def _track(call: Callable[[], str]) -> str:
    """Run a tracker call; convert any failure into a BacktestError so MLflow problems stay in
    the JSON command contract instead of leaking a traceback."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - tracking is a best-effort side effect
        raise BacktestError(f"mlflow tracking failed: {exc}") from exc


@backtest_app.command("run")
@json_errors(ValueError, LookupError, BacktestError)
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    register: bool = typer.Option(False, "--register", help="advance registry idea->backtested"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
) -> None:
    """Backtest a strategy and emit metrics JSON."""
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError(
            "--fundamentals-snapshot was given but the strategy does not declare needs_fundamentals"
        )
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError(
            "--news-snapshot was given but the strategy does not declare needs_news"
        )
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot
        else None
    )
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot
        else None
    )
    result = run_backtest(
        strategy, provider, start_dt, end_dt,
        universe_by_date=universe_by_date,
        universe_name=universe, universe_snapshots=universe_prov,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
    )

    if register:
        with registry_conn() as conn:
            repo = SqliteStrategyRepository(conn)
            existing = {s.name for s in repo.list_strategies()}
            if name not in existing:
                repo.add(name)
            reason = (
                f"backtest sharpe={result.metrics['sharpe']:.2f} "
                f"ret={result.metrics['total_return']:.2%}"
            )
            transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT, reason)

    payload = result.to_dict()
    if track:
        payload["mlflow_run_id"] = _track(
            lambda: log_backtest(
                result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
            )
        )
    emit(ok(payload))


@backtest_app.command("walk-forward")
@json_errors(ValueError, LookupError, BacktestError)
def walk_forward_cmd(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    windows: int = typer.Option(4, "--windows", help="number of equal out-of-sample windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
) -> None:
    """Walk-forward (out-of-sample) evaluation: per-window metrics + stability.

    The final OOS holdout segment is COMPUTED by walk_forward (research promote depends on it) but
    is WITHHELD from this command's output. The holdout is revealed — and burned — in exactly one
    place: `research promote`. Emitting it here would defeat that single-use guarantee, letting a
    caller peek at (and select on) the holdout without consuming it.
    """
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    result = walk_forward(strategy, provider, start_dt, end_dt,
                          windows=windows, holdout_frac=holdout_frac,
                          universe_by_date=universe_by_date,
                          universe_name=universe, universe_snapshots=universe_prov)
    payload = result.to_dict()
    payload.pop("holdout_metrics")  # withhold the holdout (reserved for `research promote`)
    if track:
        payload["mlflow_run_id"] = _track(
            lambda: log_walk_forward(
                result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
            )
        )
    emit(ok(payload))


@backtest_app.command("sweep")
@json_errors(ValueError, LookupError, BacktestError)
def sweep_cmd(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows per combo"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    param: list[str] = typer.Option(None, "--param", help="KEY=v1,v2,... (repeatable)"),
    rank_by: str = typer.Option("mean_sharpe", "--rank-by", help="mean_sharpe | min_sharpe"),
    top: int = typer.Option(20, "--top", help="max ranked rows to print"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
) -> None:
    """Sweep a strategy across a parameter grid; walk-forward score each combo and rank."""
    if top < 1:
        raise ValueError("--top must be >= 1")
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    grid = parse_grid(param or [])
    result = sweep(strategy, provider, start_dt, end_dt,
                   grid=grid, windows=windows, holdout_frac=holdout_frac, rank_by=rank_by,
                   universe_by_date=universe_by_date,
                   universe_name=universe, universe_snapshots=universe_prov)
    run_id = None
    if track:
        run_id = _track(lambda: log_sweep(result, tracking_uri=get_settings().mlflow_tracking_uri))
    recorded = _record_search_breadth(name, result)
    payload = result.to_dict()
    payload["ranked"] = payload["ranked"][:top]
    # Surface the MEASURED breadth this sweep contributed (this sweep's n_combos) and the
    # cumulative family total now on record, so the operator sees what promotion will read back.
    # Recorded by strategy NAME, so even a sweep of an UNREGISTERED strategy counts.
    payload["recorded_breadth"] = recorded
    if run_id is not None:
        payload["mlflow_run_id"] = run_id
    emit(ok(payload))


def _record_search_breadth(name: str, result: SweepResult) -> dict[str, int]:
    """Record this sweep's measured breadth into the registry, keyed by strategy NAME.

    Recorded UNCONDITIONALLY — even for a not-yet-registered strategy. Exploration precedes
    registration: keying by name (not the registry id) means a pre-registration sweep still
    counts toward the promotion breadth, so an agent can't sweep broadly first and then promote a
    freshly-registered strategy under a smaller declared --n-combos. Returns the recorded count
    plus the new cumulative family total for the emitted JSON.
    """
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.record_search_trial(name, result.n_combos, json.dumps(result.grid, sort_keys=True))
        return {"n_combos": result.n_combos, "cumulative": repo.total_search_combos(name)}
