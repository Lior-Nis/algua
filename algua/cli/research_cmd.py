from __future__ import annotations

from typing import Any

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.cli._common import ok, registry_conn, resolve_eval_inputs
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate

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
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    min_holdout_sharpe: float = typer.Option(0.5, "--min-holdout-sharpe"),
    min_holdout_return: float = typer.Option(0.0, "--min-holdout-return"),
    min_pct_positive: float = typer.Option(0.6, "--min-pct-positive"),
    min_window_sharpe: float = typer.Option(0.0, "--min-window-sharpe"),
    n_combos: int = typer.Option(None, "--n-combos", help="combos searched (recorded as evidence)"),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Gate backtested->shortlisted on walk-forward holdout + stability; promote only on pass."""
    actor_enum = Actor(actor)  # fail fast on a bad actor before running the walk-forward
    if n_combos is not None and n_combos < 1:
        raise ValueError("--n-combos must be >= 1 when provided")
    if not 0.0 <= min_pct_positive <= 1.0:
        raise ValueError("--min-pct-positive must be in [0, 1]")
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    wf = walk_forward(strategy, provider, start_dt, end_dt,
                      windows=windows, holdout_frac=holdout_frac)
    criteria = GateCriteria(
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive_windows=min_pct_positive, min_window_sharpe=min_window_sharpe,
    )
    decision = evaluate_gate(wf, criteria, n_combos=n_combos)

    promoted = False
    if decision.passed:
        with registry_conn() as conn:
            transition_strategy(SqliteStrategyRepository(conn), name, Stage.SHORTLISTED,
                                actor_enum, _gate_reason(decision))
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
    emit(ok(payload))
