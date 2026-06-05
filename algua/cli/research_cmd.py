from __future__ import annotations

from typing import Any

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.cli._common import (
    ok,
    registry_conn,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")

_HOLDOUT_REUSE_OVERRIDE = "override"


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
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
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
    allow_holdout_reuse: bool = typer.Option(
        False, "--allow-holdout-reuse",
        help="OVERRIDE the single-use holdout guard: re-evaluate a holdout window already burned "
             "by a prior promote. Records the reuse (reused=1) and marks it in the audit trail. "
             "Statistically costly — only with fresh justification.",
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
    # 1. Resolve inputs. The PIT universe is resolved up front alongside the other inputs (a bad
    # --universe refuses here, before any holdout is peeked at). The universe is intentionally NOT
    # part of the holdout-burn identity below (conservative: the same OOS data window is burned
    # regardless of universe).
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    data_source = type(provider).__name__
    snapshot_id = getattr(provider, "snapshot_id", None)
    period_start = start_dt.date().isoformat()
    period_end = end_dt.date().isoformat()
    criteria = GateCriteria(
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive_windows=min_pct_positive, min_window_sharpe=min_window_sharpe,
    )

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)  # StrategyNotFound -> JSON error before any evaluation
        # 2. Resolve search breadth — if it would refuse, refuse here, before evaluating anything.
        breadth, provenance = _resolve_breadth(repo, name, n_combos)
        # 3. Holdout-reuse pre-check — BEFORE walk_forward, so a burned holdout is never peeked at.
        # The match identity is the data window (strategy, data_source, snapshot_id, period,
        # holdout_frac) and deliberately EXCLUDES the universe: the same OOS data window is burned
        # regardless of which universe it was evaluated under (conservative).
        overlap = repo.overlapping_holdout_evaluations(
            rec.id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
        )
        if overlap and not allow_holdout_reuse:
            raise ValueError(
                f"holdout already consumed for {name!r}: an overlapping out-of-sample window was "
                f"already evaluated ({data_source}"
                f"{f'/{snapshot_id}' if snapshot_id else ''}, "
                f"{period_start}..{period_end}, holdout_frac={holdout_frac}). Re-gating the same "
                f"holdout leaks it. Use fresh out-of-sample data (a different period or data "
                f"snapshot), or pass --allow-holdout-reuse to override and accept the statistical "
                f"cost (the result will be flagged as a holdout reuse)."
            )
        reused = overlap and allow_holdout_reuse
        # 4. Run walk_forward — this evaluates (consumes) the holdout. The PIT universe threads
        # into the engine here so the holdout/stability that drives the gate is computed against
        # point-in-time membership, not the static (survivorship-biased) universe.
        wf = walk_forward(strategy, provider, start_dt, end_dt,
                          windows=windows, holdout_frac=holdout_frac,
                          universe_by_date=universe_by_date,
                          universe_name=universe, universe_snapshots=universe_prov)
        # 5. Record the holdout evaluation NOW — looking at it consumes it regardless of pass/fail.
        repo.record_holdout_evaluation(
            rec.id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            config_hash=wf.config_hash, reused=reused,
        )
        # 6. Evaluate the gate and promote on pass.
        decision = evaluate_gate(wf, criteria, n_combos=breadth, breadth_provenance=provenance)
        promoted = False
        if decision.passed:
            reason = _gate_reason(decision)
            if reused:
                reason += "; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE
            transition_strategy(repo, name, Stage.SHORTLISTED, actor_enum, reason)
            promoted = True

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "strategy": name,
        "promoted": promoted,
        "config_hash": wf.config_hash,
        "snapshot_id": wf.snapshot_id,
        "holdout": wf.holdout_metrics,
        "stability": wf.stability,
        "universe_name": wf.universe_name,
        "universe_snapshots": wf.universe_snapshots,
    }
    if reused:
        payload["holdout_reuse"] = _HOLDOUT_REUSE_OVERRIDE
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
