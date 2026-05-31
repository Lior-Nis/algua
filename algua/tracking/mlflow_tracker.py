from __future__ import annotations

from typing import Any

from algua.backtest.result import BacktestResult
from algua.backtest.walkforward import WalkForwardResult


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


def _numeric_metrics(d: dict[str, Any]) -> dict[str, float]:
    """Keep only real numeric values (drops None/str/bool), as floats — safe for log_metrics."""
    return {
        k: float(v)
        for k, v in d.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _stamp_params(result: Any) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "snapshot_id": result.snapshot_id,
        "seed": result.seed,
        "period_start": result.period["start"],
        "period_end": result.period["end"],
        "timeframe": result.timeframe,
    }


def log_backtest(result: BacktestResult, params: dict[str, Any], *, tracking_uri: str) -> str:
    """Log a single backtest as an MLflow run (experiment = strategy name). Returns run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(result.strategy)
    with mlflow.start_run() as run:
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
        return run.info.run_id


def log_walk_forward(
    result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
) -> str:
    """Log a walk-forward evaluation as one MLflow run. Returns run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(result.strategy)
    with mlflow.start_run() as run:
        mlflow.log_params({
            **{f"param.{k}": v for k, v in params.items()},
            "windows": result.windows, "holdout_frac": result.holdout_frac,
            **_stamp_params(result),
        })
        mlflow.log_metrics({
            **_numeric_metrics(result.stability),
            **_numeric_metrics(_flatten(result.holdout_metrics, "holdout.")),
        })
        mlflow.set_tags({
            "kind": "walk_forward", "strategy": result.strategy,
            "config_hash": result.config_hash, "snapshot_id": str(result.snapshot_id),
            "timeframe": result.timeframe,
        })
        mlflow.log_dict(result.to_dict(), "result.json")
        return run.info.run_id
