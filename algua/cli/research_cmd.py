from __future__ import annotations

import sqlite3
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
from algua.contracts.lifecycle import Actor
from algua.registry.promotion import promotion_preflight, run_gate
from algua.registry.store import SqliteStrategyRepository
from algua.research.gates import GateCriteria

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")

_HOLDOUT_REUSE_OVERRIDE = "override"


@research_app.command("promote")
# sqlite3.OperationalError keeps lock-contention ("database is locked") from reserve_holdout's
# BEGIN IMMEDIATE inside the JSON envelope, not a leaked traceback (CLI JSON-output contract).
@json_errors(ValueError, LookupError, BacktestError, sqlite3.OperationalError)
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
    allow_non_pit: bool = typer.Option(
        False, "--allow-non-pit",
        help="HUMAN-ONLY override: promote a non-PIT (survivorship-biased) backtest. Audited. "
             "Agents may not pass this.",
    ),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Gate backtested->candidate on walk-forward holdout + stability; promote only on pass.

    The holdout-Sharpe bar is DEFLATED by FUNNEL-WIDE search breadth (the multiple-testing defense):
    the max of this strategy's lifetime recorded breadth and the funnel-wide breadth in the rolling
    window. Breadth is MEASURED as the sum of recorded `search_trials` (from `backtest sweep`); an
    agent must have measured breadth (no measured trials => refused). Declaring breadth via
    --n-combos is HUMAN-ONLY and recorded with provenance="declared" (auditably less trustworthy).
    For an agent the universe must be PIT (`--universe`); non-PIT fails closed unless a human passes
    --allow-non-pit. A minimum holdout-observations floor (63) also fails closed (underpowered
    holdouts). On pass for an agent this mints the single-use gate token the BACKTESTED->CANDIDATE
    transition consumes.
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
        repo.get(name)  # StrategyNotFound -> JSON error before any work
        # PREFLIGHT (pre-peek): relaxations-need-human + stage legality + breadth. Refuses here,
        # before walk_forward touches the holdout.
        breadth = promotion_preflight(
            repo, name, actor=actor_enum, declared_combos=n_combos,
            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
            provider=provider, start=start_dt, end=end_dt)
        # Atomic holdout reservation (#161): claim the window under the write lock (fast SELECT +
        # INSERT a pending row), run walk_forward with NO lock held, then finalize on success /
        # release on a clean failure. The match identity is the data window and deliberately
        # EXCLUDES the universe (the same OOS window is burned regardless of universe). A pending
        # reservation blocks a concurrent run exactly like a committed burn (fail closed).
        reservation_id, reused = repo.reserve_holdout(
            repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            allow_reuse=allow_holdout_reuse)  # raises here = fail closed (overlap, no reuse)
        try:
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov,
                # Burn-on-peek: commit the reservation into a burn the instant BEFORE walk_forward
                # evaluates the holdout metric. Because release_holdout_reservation no-ops on a
                # committed row, the except-release below is then correct for EVERY post-peek
                # failure (incl. KeyboardInterrupt) — a computed holdout can never be released.
                on_peek=lambda cfg: repo.finalize_holdout_reservation(
                    reservation_id, config_hash=cfg),
            )
        except BaseException:
            # Pre-peek failure: the row is still pending, so release frees the window. Post-peek
            # failure: on_peek already committed, so this DELETE matches 0 rows (harmless no-op) and
            # the burn survives. Swallow a release error so it never masks the original failure.
            try:
                repo.release_holdout_reservation(reservation_id)
            except Exception:
                pass
            raise
        outcome = run_gate(
            repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
            universe_name=universe, universe_snapshots=universe_prov,
            period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
            data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
            reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
        decision, promoted = outcome.decision, outcome.promoted

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "n_funnel": decision.n_combos,
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
