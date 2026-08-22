"""Recorder seam for the economic-layer run ledger.

Mirrors ``algua/registry/search_breadth.py``: pure mapping helpers plus thin recorders whose
TRANSACTION IS CALLER-OWNED — the CLI wraps them in ``with registry_conn() as conn:`` and passes
``SqliteStrategyRepository(conn)``.

The mapping helpers are PURE (no I/O, no DB) so they are testable without a connection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from algua.backtest.result import BacktestResult
    from algua.backtest.sweep import SweepResult
    from algua.backtest.walkforward import WalkForwardResult
    from algua.registry.store import SqliteStrategyRepository

# Provenance attributes shared by BacktestResult / WalkForwardResult / SweepResult. Read with
# getattr because the three dataclasses carry overlapping but not identical field sets.
_PROVENANCE_ATTRS = (
    "code_hash", "config_hash", "dependency_hash", "data_source", "snapshot_id",
    "universe_name", "fundamentals_snapshot", "news_snapshot", "delisting_snapshot",
    "seed", "timeframe",
)


def provenance_of(result: Any) -> dict[str, Any]:
    """Provenance columns present on a backtest-family result. Absent attributes are omitted, not
    written as NULL, so a later widening of a result type needs no change here."""
    out: dict[str, Any] = {
        k: getattr(result, k) for k in _PROVENANCE_ATTRS if getattr(result, k, None) is not None
    }
    period = getattr(result, "period", None)
    if isinstance(period, dict):
        if period.get("start") is not None:
            out["period_start"] = period["start"]
        if period.get("end") is not None:
            out["period_end"] = period["end"]
    return out


def _sample(metrics: dict[str, float], suffix: str) -> dict[str, float | int | None]:
    """Map one `metrics_from_returns` dict onto sample-suffixed vocabulary keys.

    SENTINEL RULE: `metrics_from_returns` returns a literal 0.0 for an UNDEFINED Sharpe
    (ann_volatility == 0), Sortino (zero downside deviation) and Calmar (max_drawdown == 0).
    Recording those as 0.0 would rank a degenerate run above a genuinely negative one, so they
    become NULL. We key off the DEGENERACY CONDITION, not the value — a genuine 0.0 Sharpe on a
    volatile series is a measurement and survives.

    Accepted imprecision: a constant NEGATIVE return series has ann_volatility == 0 but a
    computable Sortino; it is nulled here. NULL is honest; a sentinel zero is not.
    """
    ann_vol = metrics.get("ann_volatility")
    mdd = metrics.get("max_drawdown")
    degenerate_ratio = ann_vol is None or ann_vol == 0.0
    out: dict[str, float | int | None] = {
        f"sharpe{suffix}": None if degenerate_ratio else metrics.get("sharpe"),
        f"sortino{suffix}": None if degenerate_ratio else metrics.get("sortino"),
        f"total_return{suffix}": metrics.get("total_return"),
        f"max_drawdown{suffix}": mdd,
        f"ann_vol{suffix}": ann_vol,
    }
    if suffix == "_is":
        out["cagr_is"] = metrics.get("cagr")
        out["calmar_is"] = None if (mdd is None or mdd == 0.0) else metrics.get("calmar")
    return out


def backtest_metrics(result: BacktestResult) -> dict[str, float | int | None]:
    """Fixed-vocabulary metrics for a `backtest` run. In-sample by construction."""
    out = _sample(dict(result.metrics), "_is")
    returns = getattr(result, "returns", None)
    if returns is not None:
        out["n_obs_is"] = int(len(returns))
    return out


def walk_forward_metrics(result: WalkForwardResult) -> dict[str, float | int | None]:
    """Fixed-vocabulary metrics for a `walk_forward` run.

    The holdout segment is the OUT-OF-SAMPLE measurement; the per-window stability figures keep
    their own names (they are neither IS nor OOS — they are a dispersion across folds). No
    in-sample full-period figure is emitted: a walk-forward does not measure one, and inventing
    it would put a fabricated number on the scatter's x-axis.
    """
    holdout = dict(result.holdout_metrics)
    out = _sample(holdout, "_oos")
    if holdout.get("n_bars") is not None:
        out["n_obs_oos"] = int(holdout["n_bars"])
    stability = dict(result.stability)
    out["mean_window_sharpe"] = stability.get("mean_sharpe")
    out["std_window_sharpe"] = stability.get("std_sharpe")
    out["min_window_sharpe"] = stability.get("min_sharpe")
    out["pct_positive_windows"] = stability.get("pct_positive_windows")
    return out


def record_backtest_run(
    repo: SqliteStrategyRepository,
    name: str,
    result: BacktestResult,
    *,
    params: dict[str, Any] | None = None,
) -> int:
    """Record one `backtest` run. Recorded UNCONDITIONALLY — even for a not-yet-registered
    strategy, for the same reason `record_search_breadth` is: exploration precedes registration
    and that evidence must not be discarded.

    `params` is the strategy's config params, passed EXPLICITLY: `BacktestResult` carries
    `config_hash` but not the config itself, so there is nothing to read off the result.
    """
    return repo.record_run(
        "backtest", name,
        provenance=provenance_of(result),
        config=dict(params or {}),
        metrics=backtest_metrics(result),
        components=list(_components_of(result)),
    )


def record_walk_forward_run(
    repo: SqliteStrategyRepository, name: str, result: WalkForwardResult,
) -> int:
    """Record one `walk_forward` run."""
    return repo.record_run(
        "walk_forward", name,
        provenance=provenance_of(result),
        metrics=walk_forward_metrics(result),
        components=list(_components_of(result)),
    )


def record_sweep_run(
    repo: SqliteStrategyRepository, name: str, result: SweepResult,
) -> dict[str, Any]:
    """Record one `sweep` parent run plus a child `sweep_trial` per ranked combo.

    `SweepResult.ranked` already carries every combo's `{params, config_hash, stability, score}`,
    so no re-computation is needed. The children are written in ONE batched transaction (see
    `record_sweep_trials`), and a truncated trial set is stamped back onto the parent.
    """
    parent = repo.record_run(
        "sweep", name,
        provenance=provenance_of(result),
        config={"grid": result.grid, "rank_by": result.rank_by,
                "windows": result.windows, "holdout_frac": result.holdout_frac},
        metrics={"mean_window_sharpe": result.trial_sharpe_mean},
    )
    trials = [
        {
            "config": record["params"],
            "config_hash": record.get("config_hash"),
            "metrics": {
                "mean_window_sharpe": record["stability"].get("mean_sharpe"),
                "std_window_sharpe": record["stability"].get("std_sharpe"),
                "min_window_sharpe": record["stability"].get("min_sharpe"),
                "pct_positive_windows": record["stability"].get("pct_positive_windows"),
            },
        }
        for record in result.ranked
    ]
    n_written, truncated_at = repo.record_sweep_trials(parent, name, trials)
    if truncated_at is not None:
        repo.stamp_trials_truncated(parent, truncated_at)
    return {"run_id": parent, "trials_written": n_written, "trials_truncated_at": truncated_at}


def _components_of(result: Any) -> list[dict[str, Any]]:
    """Model-layer lineage as a LIST, even though `model_ref` is singular today (spec §2.1)."""
    ref = getattr(result, "model_ref", None)
    return [ref] if isinstance(ref, dict) else []
