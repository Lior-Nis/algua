from __future__ import annotations

from contextlib import closing
from typing import Any

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.cli.app import app, emit
from algua.cli.backtest_cmd import _select_provider, _utc
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate
from algua.strategies.loader import load_strategy

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")


def _gate_reason(decision: GateDecision) -> str:
    parts = [f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}" for c in decision.checks]
    extra = f"; n_combos={decision.n_combos}" if decision.n_combos is not None else ""
    return "gate pass: " + ", ".join(parts) + extra


@research_app.command("promote")
@json_errors(ValueError, LookupError, BacktestError)
def promote(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),  # noqa: B008
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    min_holdout_sharpe: float = typer.Option(0.5, "--min-holdout-sharpe"),
    min_holdout_return: float = typer.Option(0.0, "--min-holdout-return"),
    min_pct_positive: float = typer.Option(0.6, "--min-pct-positive"),
    min_window_sharpe: float = typer.Option(0.0, "--min-window-sharpe"),
    n_combos: int = typer.Option(None, "--n-combos", help="combos searched (recorded as evidence)"),  # noqa: B008
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Gate backtested->shortlisted on walk-forward holdout + stability; promote only on pass."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before running the walk-forward
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    wf = walk_forward(strategy, provider, _utc(start), _utc(end),
                      windows=windows, holdout_frac=holdout_frac)
    criteria = GateCriteria(
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive_windows=min_pct_positive, min_window_sharpe=min_window_sharpe,
    )
    decision = evaluate_gate(wf, criteria, n_combos=n_combos)

    promoted = False
    if decision.passed:
        with closing(connect(get_settings().db_path)) as conn:
            migrate(conn)
            store.transition(conn, name, Stage.SHORTLISTED, actor_enum, _gate_reason(decision),
                             code_hash=wf.config_hash, config_hash=wf.config_hash)
        promoted = True

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "strategy": name,
        "promoted": promoted,
        "config_hash": wf.config_hash,
        "snapshot_id": wf.snapshot_id,
        "holdout": wf.holdout_metrics,
        "stability": wf.stability,
    }
    emit(payload)
