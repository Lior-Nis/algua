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
    # The `n_combos is not None` guard is defensive: `promote` refuses unless _resolve_breadth
    # succeeds, so n_combos is always set when this function is reached on a passing decision.
    breadth = (
        f"; breadth={decision.n_combos}({decision.breadth_provenance})"
        f"; min_holdout_sharpe={decision.base_min_holdout_sharpe:.4g}"
        f"->{decision.effective_min_holdout_sharpe:.4g}"
        if decision.n_combos is not None
        else ""
    )
    return "gate pass: " + ", ".join(parts) + breadth


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
    n_combos: int = typer.Option(
        None, "--n-combos",
        help="OPERATOR DECLARATION of search breadth, used ONLY when no measured sweep trials "
             "exist; the measured sum from `backtest sweep` is preferred and always wins",
    ),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Gate backtested->shortlisted on walk-forward holdout + stability; promote only on pass.

    The holdout-Sharpe bar is DEFLATED by the strategy's search breadth (the multiple-testing
    defense). Breadth is MEASURED as the sum of recorded `search_trials` (from `backtest sweep`)
    when any exist; otherwise it must be DECLARED via --n-combos, else promotion is refused. A
    declared number is recorded with provenance="declared" so it is auditably less trustworthy.
    """
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

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        breadth, provenance = _resolve_breadth(repo, name, n_combos)
        decision = evaluate_gate(wf, criteria, n_combos=breadth, breadth_provenance=provenance)
        promoted = False
        if decision.passed:
            transition_strategy(repo, name, Stage.SHORTLISTED, actor_enum, _gate_reason(decision))
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


def _resolve_breadth(
    repo: SqliteStrategyRepository, name: str, declared: int | None
) -> tuple[int, str]:
    """Resolve the search breadth N and its provenance for the gate.

    The MEASURED sum of recorded `search_trials` is the default, trusted path and always wins
    over a declaration. If nothing is recorded, fall back to the operator DECLARATION
    (--n-combos). If neither exists, REFUSE — never silently default breadth to 1, which would
    waive the multiple-testing penalty for an unmeasured search.
    """
    rec = repo.get(name)  # raises StrategyNotFound -> JSON error (promotion needs registration)
    measured = repo.total_search_combos(rec.id)
    if measured > 0:
        return measured, "measured"
    if declared is not None:
        return declared, "declared"
    raise ValueError(
        f"no recorded search breadth for {name!r}; run `algua backtest sweep {name} ...` "
        f"(records breadth) or pass --n-combos N to declare it explicitly"
    )
