from __future__ import annotations

import typer

from algua.backtest.walkforward import walk_forward
from algua.cli._common import ok, project
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.evaluation.backtest_run import run_backtest_task
from algua.evaluation.inputs import (
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.evaluation.sweep_run import sweep_task
from algua.tracking.factory import get_tracker
from algua.tracking.record import record_tracking

backtest_app = typer.Typer(help="Run backtests", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")

# --summary keep-lists (#349): the decision-relevant scalars per command. Keep-lists (not
# drop-lists) so a future field is excluded-by-default. Walk-forward keeps everything but the
# bulky per-window `window_metrics` (its scalar summary is `stability`); sweep keeps everything
# but the per-combo `ranked` list and the `grid` (the headline combo is `best`).
_WF_SUMMARY_KEYS = (
    "strategy", "data_source", "snapshot_id", "timeframe", "seed", "period", "windows",
    "holdout_frac", "stability", "code_hash", "dependency_hash", "config_hash",
    "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
    "mlflow_run_id", "mlflow_tracking_error", "mlflow_tracking_skipped",
)
_SWEEP_SUMMARY_KEYS = (
    "strategy", "n_combos", "rank_by", "best", "trial_sharpe_count", "trial_sharpe_mean",
    "trial_sharpe_var_ann", "recorded_breadth", "code_hash", "dependency_hash", "data_source",
    "snapshot_id", "timeframe", "seed", "period", "windows", "holdout_frac", "universe_name",
    "universe_snapshots", "fundamentals_snapshot", "news_snapshot", "mlflow_run_id",
    "mlflow_tracking_error", "mlflow_tracking_skipped",
)


@backtest_app.command("run")
@json_errors
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
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="realize a held-into-gap name at its last close when no delisting record exists"),
    register: bool = typer.Option(False, "--register", help="advance registry idea->backtested"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
    emit_series: str = typer.Option(
        None, "--emit-series",
        help="write the daily return series to a parquet at PATH (for series plots)"),
) -> None:
    """Backtest a strategy and emit metrics JSON."""
    emit(ok(run_backtest_task(
        name, start=start, end=end, demo=demo, snapshot=snapshot, universe=universe,
        fundamentals_snapshot=fundamentals_snapshot, news_snapshot=news_snapshot,
        delistings=delistings, assume_terminal_last_close=assume_terminal_last_close,
        register=register, emit_series=emit_series, track=track,
    )))


@backtest_app.command("walk-forward")
@json_errors
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
    embargo: int = typer.Option(
        None, "--embargo", min=0,
        help="override the in-sample/holdout purge gap in bars (#345); default = "
             "max(feature_lookback, decision_lag_bars) from the strategy"),
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="realize a held-into-gap name at its last close when no delisting record exists"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
    summary: bool = typer.Option(
        False, "--summary",
        help="emit only decision-relevant scalars (drops per-window metrics; context-rot defense)"),
) -> None:
    """Walk-forward (out-of-sample) evaluation: per-window metrics + stability.

    The final OOS holdout segment is COMPUTED by walk_forward (research promote depends on it) but
    is WITHHELD from this command's output. The holdout is revealed — and burned — in exactly one
    place: `research promote`. Emitting it here would defeat that single-use guarantee, letting a
    caller peek at (and select on) the holdout without consuming it.
    """
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
    delisting_records, _delisting_prov = resolve_delisting_inputs(delistings, end_dt)
    result = walk_forward(strategy, provider, start_dt, end_dt,
                          windows=windows, holdout_frac=holdout_frac, embargo=embargo,
                          universe_by_date=universe_by_date,
                          universe_name=universe, universe_snapshots=universe_prov,
                          fundamentals_provider=fundamentals_provider,
                          news_provider=news_provider,
                          delisting_records=delisting_records,
                          assume_terminal_last_close=assume_terminal_last_close)
    payload = result.to_dict()
    payload.pop("holdout_metrics")  # withhold the holdout (reserved for `research promote`)
    if track:
        record_tracking(payload, lambda: get_tracker().log_walk_forward(
            result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
        ))
    out = ok(payload)
    emit(project(out, _WF_SUMMARY_KEYS) if summary else out)


@backtest_app.command("sweep")
@json_errors
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
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="realize a held-into-gap name at its last close when no delisting record exists"),
    track: bool = typer.Option(False, "--track", help="log this run to MLflow"),
    summary: bool = typer.Option(
        False, "--summary",
        help="emit only decision-relevant scalars (drops the ranked combo list; context-rot "
             "defense)"),
) -> None:
    """Sweep a strategy across a parameter grid; walk-forward score each combo and rank."""
    payload = sweep_task(
        name, start=start, end=end, demo=demo, snapshot=snapshot, universe=universe,
        windows=windows, holdout_frac=holdout_frac, param=param, rank_by=rank_by, top=top,
        fundamentals_snapshot=fundamentals_snapshot, news_snapshot=news_snapshot,
        delistings=delistings, assume_terminal_last_close=assume_terminal_last_close, track=track,
    )
    out = ok(payload)
    emit(project(out, _SWEEP_SUMMARY_KEYS) if summary else out)
