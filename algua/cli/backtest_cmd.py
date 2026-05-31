from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime

import typer

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.engine import run as run_backtest
from algua.backtest.walkforward import walk_forward
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.contracts.types import DataProvider
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.strategies.loader import load_strategy

backtest_app = typer.Typer(help="Run backtests", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


def _utc(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(tzinfo=UTC)


def _select_provider(demo: bool, snapshot: str | None) -> DataProvider:
    if demo and snapshot:
        raise ValueError("pass only one of --demo or --snapshot")
    if demo:
        return SyntheticProvider(seed=0)
    if snapshot:
        return StoreBackedProvider(DataStore(get_settings().data_dir), snapshot)
    raise ValueError("pass one of --demo (synthetic) or --snapshot <id> (real data)")


@backtest_app.command("run")
@json_errors(ValueError, LookupError, BacktestError)
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    register: bool = typer.Option(False, "--register", help="advance registry idea->backtested"),
) -> None:
    """Backtest a strategy and emit metrics JSON."""
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    result = run_backtest(strategy, provider, _utc(start), _utc(end))

    if register:
        with closing(connect(get_settings().db_path)) as conn:
            migrate(conn)
            existing = {s.name for s in store.list_strategies(conn)}
            if name not in existing:
                store.add_strategy(conn, name)
            reason = (
                f"backtest sharpe={result.metrics['sharpe']:.2f} "
                f"ret={result.metrics['total_return']:.2%}"
            )
            # NOTE: code_hash == config_hash for now; real source-code hashing is wired in a
            # later sub-project before the paper->live gate is ever in scope.
            store.transition(conn, name, Stage.BACKTESTED, Actor.AGENT, reason,
                             code_hash=result.config_hash, config_hash=result.config_hash)

    emit(result.to_dict())


@backtest_app.command("walk-forward")
@json_errors(ValueError, LookupError, BacktestError)
def walk_forward_cmd(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    windows: int = typer.Option(4, "--windows", help="number of equal out-of-sample windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
) -> None:
    """Walk-forward (out-of-sample) evaluation: per-window metrics, holdout, and stability."""
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    result = walk_forward(strategy, provider, _utc(start), _utc(end),
                          windows=windows, holdout_frac=holdout_frac)
    emit(result.to_dict())
