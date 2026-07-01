from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import typer

from algua.backtest.engine import BacktestError
from algua.backtest.engine import run as run_backtest
from algua.backtest.result import BacktestResult, series_frame
from algua.backtest.sweep import parse_grid, sweep
from algua.backtest.walkforward import walk_forward
from algua.cli._common import (
    ok,
    project,
    registry_conn,
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
    sync_kb_doc,
)
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.data.files import frame_to_parquet_bytes, write_bytes_atomic
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.registry.search_breadth import record_search_breadth
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.tracking.mlflow_tracker import log_backtest, log_sweep, log_walk_forward

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
    "mlflow_run_id",
)
_SWEEP_SUMMARY_KEYS = (
    "strategy", "n_combos", "rank_by", "best", "trial_sharpe_count", "trial_sharpe_mean",
    "trial_sharpe_var_ann", "recorded_breadth", "code_hash", "dependency_hash", "data_source",
    "snapshot_id", "timeframe", "seed", "period", "windows", "holdout_frac", "universe_name",
    "universe_snapshots", "fundamentals_snapshot", "news_snapshot", "mlflow_run_id",
)


def _track(call: Callable[[], str]) -> str:
    """Run a tracker call; convert any failure into a BacktestError so MLflow problems stay in
    the JSON command contract instead of leaking a traceback."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - tracking is a best-effort side effect
        raise BacktestError(f"mlflow tracking failed: {exc}") from exc


def emit_series_file(result: BacktestResult, path: Path) -> dict:
    """Write the backtest's daily return series to a deterministic, provenance-stamped parquet at
    `path` and return the stdout `series` descriptor. Fail closed (#181): a `None`, empty, or
    non-finite series raises BacktestError — never a partial/empty file."""
    if (
        result.returns is None
        or len(result.returns) == 0
        or not np.isfinite(result.returns.to_numpy(dtype=float)).all()
    ):
        raise BacktestError("backtest produced no finite return series; nothing to emit")
    frame, metadata = series_frame(result)
    try:
        write_bytes_atomic(frame_to_parquet_bytes(frame, metadata), path)
    except (OSError, pa.ArrowInvalid, Exception) as exc:
        if isinstance(exc, BacktestError):
            raise
        raise BacktestError(f"failed to write series to {path}: {exc}") from exc
    return {
        "path": str(path), "n": int(len(frame)),
        "code_hash": result.code_hash, "dependency_hash": result.dependency_hash,
        "config_hash": result.config_hash, "snapshot_id": result.snapshot_id,
        "seed": result.seed, "data_source": result.data_source,
        "start": result.period["start"], "end": result.period["end"],
        "timeframe": result.timeframe,
        "universe_name": result.universe_name,
        "fundamentals_snapshot": result.fundamentals_snapshot,
        "news_snapshot": result.news_snapshot,
        "delisting_snapshot": result.delisting_snapshot,
    }


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
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    delisting_records, delisting_snapshot_id = resolve_delisting_inputs(delistings, end_dt)
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
        delisting_records=delisting_records,
        delisting_snapshot=delisting_snapshot_id,
        assume_terminal_last_close=assume_terminal_last_close,
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
        # Re-sync the kb doc to the new `backtested` stage (#331): best-effort, out-of-transaction.
        sync_kb_doc(name)

    # Persist return series for the return-correlation clustering axis (#222, Task 7).
    # Only persists for registered strategies; silently skips otherwise.
    if result.returns is not None:
        with registry_conn() as conn:
            repo = SqliteStrategyRepository(conn)
            try:
                repo.get(name)
            except Exception:  # noqa: BLE001 — strategy not yet registered, skip
                pass
            else:
                repo.persist_backtest_returns(
                    name,
                    start_dt.date().isoformat(),
                    end_dt.date().isoformat(),
                    result.returns,
                )

    payload = result.to_dict()
    if emit_series:
        payload["series"] = emit_series_file(result, Path(emit_series))
    if track:
        payload["mlflow_run_id"] = _track(
            lambda: log_backtest(
                result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
            )
        )
    emit(ok(payload))


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
        payload["mlflow_run_id"] = _track(
            lambda: log_walk_forward(
                result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
            )
        )
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
    if top < 1:
        raise ValueError("--top must be >= 1")
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
    grid = parse_grid(param or [])
    result = sweep(strategy, provider, start_dt, end_dt,
                   grid=grid, windows=windows, holdout_frac=holdout_frac, rank_by=rank_by,
                   universe_by_date=universe_by_date,
                   universe_name=universe, universe_snapshots=universe_prov,
                   fundamentals_provider=fundamentals_provider,
                   news_provider=news_provider,
                   delisting_records=delisting_records,
                   assume_terminal_last_close=assume_terminal_last_close)
    run_id = None
    if track:
        run_id = _track(lambda: log_sweep(result, tracking_uri=get_settings().mlflow_tracking_uri))
    with registry_conn() as conn:
        recorded = record_search_breadth(SqliteStrategyRepository(conn), name, result)
    payload = result.to_dict()
    payload["ranked"] = payload["ranked"][:top]
    # Surface the MEASURED breadth this sweep contributed (this sweep's n_combos) and the
    # cumulative family total now on record, so the operator sees what promotion will read back.
    # Recorded by strategy NAME, so even a sweep of an UNREGISTERED strategy counts.
    payload["recorded_breadth"] = recorded
    if run_id is not None:
        payload["mlflow_run_id"] = run_id
    out = ok(payload)
    emit(project(out, _SWEEP_SUMMARY_KEYS) if summary else out)
