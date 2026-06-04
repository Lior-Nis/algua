from __future__ import annotations

from typing import Any


def latest_run_metrics(strategy: str, *, tracking_uri: str) -> dict[str, Any] | None:
    """Most recent MLflow run for `strategy` as a flat dict, or None if there are none.

    Reads the experiment named `strategy`. Works against a file backend — no server.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(strategy)
    if exp is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        return None
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "kind": run.data.tags.get("kind", "unknown"),
        "snapshot_id": run.data.params.get("snapshot_id"),
        "seed": run.data.params.get("seed"),
        "metrics": dict(run.data.metrics),
    }
