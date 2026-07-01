from __future__ import annotations

import math
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from algua.backtest.result import BacktestResult, series_frame
from algua.backtest.sweep import SweepResult
from algua.backtest.walkforward import WalkForwardResult
from algua.data.files import frame_to_parquet_bytes

# ---------------------------------------------------------------------------
# Protocol (#45)
# ---------------------------------------------------------------------------

class ExperimentTracker(Protocol):
    """Structural protocol for experiment loggers."""

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str: ...

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str: ...

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into `prefix.key` entries."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{full}."))
        else:
            out[full] = value
    return out


def _is_finite_number(v: Any) -> bool:
    """Return True iff *v* is a real, finite number (int/float, not bool, not NaN/inf)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _numeric_metrics(d: dict[str, Any]) -> dict[str, float]:
    """Keep only real, finite numeric values — NaN/inf are dropped; safe for log_metrics."""
    return {k: float(v) for k, v in d.items() if _is_finite_number(v)}


def _universe_param(universe_name: str | None) -> str:
    """MLflow param value for a run's PIT universe (#333).

    A named run logs its universe name; a static-universe run (``universe_name is None``)
    logs the literal ``"static"`` sentinel — never absent — so the knowledge-doc RESULTS
    block can tell an explicitly-static run apart from a legacy run that predates this stamp
    (which has no ``universe_name`` param at all). Together with ``snapshot_id`` and the
    period this is the canonical resolver key for point-in-time membership (the same key
    ``research promote --universe NAME --start --end`` reproduces from)."""
    return universe_name if universe_name is not None else "static"


def _stamp_params(result: Any) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "snapshot_id": result.snapshot_id,
        "seed": result.seed,
        "period_start": result.period["start"],
        "period_end": result.period["end"],
        "timeframe": result.timeframe,
        "code_hash": result.code_hash,
        "dependency_hash": result.dependency_hash,
        "universe_name": _universe_param(result.universe_name),
    }


def _log_series_artifact(result: BacktestResult) -> None:
    """Log the backtest's daily return series as a `series.parquet` MLflow artifact (#181).

    The report-experiments skill can then plot the LOGGED run's own series (no re-run, no
    code/input drift). Only `log_backtest` calls this — `log_sweep`/`log_walk_forward` must
    NOT, because their return vectors contain the reserved single-use holdout tail.
    Best-effort: an absent/empty/non-finite series is skipped — do NOT raise."""
    import mlflow

    if (
        result.returns is None
        or len(result.returns) == 0
        or not np.isfinite(result.returns.to_numpy(dtype=float)).all()
    ):
        return
    frame, metadata = series_frame(result)
    data = frame_to_parquet_bytes(frame, metadata)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "series.parquet"  # mlflow logs the artifact under its basename
        p.write_bytes(data)
        mlflow.log_artifact(str(p))


@contextmanager
def _run(experiment: str, tracking_uri: str) -> Generator[Any, None, None]:
    """Import mlflow, set tracking URI + experiment once, yield a started run."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        yield run


# ---------------------------------------------------------------------------
# Public loggers
# ---------------------------------------------------------------------------

def log_backtest(result: BacktestResult, params: dict[str, Any], *, tracking_uri: str) -> str:
    """Log a single backtest as an MLflow run (experiment = strategy name). Returns run_id."""
    import mlflow

    with _run(result.strategy, tracking_uri) as run:
        mlflow.log_params({
            **{f"param.{k}": v for k, v in params.items()},
            **_stamp_params(result),
        })
        mlflow.log_metrics(_numeric_metrics(result.metrics))
        mlflow.set_tags({
            "kind": "backtest", "strategy": result.strategy,
            "config_hash": result.config_hash, "snapshot_id": str(result.snapshot_id),
            "timeframe": result.timeframe,
        })
        mlflow.log_dict(result.to_dict(), "result.json")
        _log_series_artifact(result)  # logs series.parquet iff returns is non-empty (#181)
        return run.info.run_id


def log_sweep(result: SweepResult, *, tracking_uri: str) -> str:
    """Log a sweep as a parent run with a nested child run per ranked combo.

    Returns parent run_id.
    n_combos is logged as a param (it is a config-level count, not a result metric).
    """
    import mlflow

    with _run(result.strategy, tracking_uri) as parent:
        mlflow.log_params({
            **{f"grid.{k}": ",".join(str(x) for x in vals) for k, vals in result.grid.items()},
            "n_combos": result.n_combos, "rank_by": result.rank_by,
            "windows": result.windows, "holdout_frac": result.holdout_frac,
            "snapshot_id": result.snapshot_id, "seed": result.seed,
            "period_start": result.period["start"], "period_end": result.period["end"],
            "timeframe": result.timeframe, "code_hash": result.code_hash,
            "dependency_hash": result.dependency_hash,
            "universe_name": _universe_param(result.universe_name),
        })
        if result.best is not None:
            _best_score = result.best["score"]
            if _is_finite_number(_best_score):
                mlflow.log_metric("best_score", float(_best_score))
        mlflow.set_tags({"kind": "sweep", "strategy": result.strategy})
        mlflow.log_dict(result.to_dict(), "sweep.json")

        for entry in result.ranked:
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    **{f"param.{k}": v for k, v in entry["params"].items()},
                    "config_hash": entry["config_hash"],
                    # Shared sweep context so a child run is reproducible without the parent.
                    "snapshot_id": result.snapshot_id, "seed": result.seed,
                    "period_start": result.period["start"], "period_end": result.period["end"],
                    "timeframe": result.timeframe, "windows": result.windows,
                    "holdout_frac": result.holdout_frac, "code_hash": result.code_hash,
                    "dependency_hash": result.dependency_hash,
                    "universe_name": _universe_param(result.universe_name),
                })
                _entry_score = entry["score"]
                _score_metric: dict[str, float] = (
                    {"score": float(_entry_score)} if _is_finite_number(_entry_score) else {}
                )
                # No holdout metrics: the sweep withholds the OOS holdout (reserved for
                # `research promote`), so there is nothing per-combo to log here.
                mlflow.log_metrics({
                    **_score_metric,
                    **_numeric_metrics(_flatten(entry["stability"])),
                })
                mlflow.set_tags({"kind": "sweep_combo", "strategy": result.strategy})
        return parent.info.run_id


def log_walk_forward(
    result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
) -> str:
    """Log a walk-forward evaluation as one MLflow run. Returns run_id.

    The OOS holdout is WITHHELD here, exactly as the `backtest walk-forward` command withholds it:
    it is revealed (and burned) only by `research promote`, so neither the logged metrics nor the
    result.json artifact carry the holdout segment."""
    import mlflow

    with _run(result.strategy, tracking_uri) as run:
        mlflow.log_params({
            **{f"param.{k}": v for k, v in params.items()},
            "windows": result.windows, "holdout_frac": result.holdout_frac,
            "embargo": result.embargo,
            **_stamp_params(result),
        })
        mlflow.log_metrics(_numeric_metrics(result.stability))
        mlflow.set_tags({
            "kind": "walk_forward", "strategy": result.strategy,
            "config_hash": result.config_hash, "snapshot_id": str(result.snapshot_id),
            "timeframe": result.timeframe,
        })
        # Strip the holdout from the persisted artifact too (it lives only in `research promote`).
        # No default: a KeyError here means the field was renamed and the strip silently no-oped,
        # which would re-leak the holdout into result.json — fail loud so the rename is caught.
        wf_dict = result.to_dict()
        wf_dict.pop("holdout_metrics")
        mlflow.log_dict(wf_dict, "result.json")
        return run.info.run_id
