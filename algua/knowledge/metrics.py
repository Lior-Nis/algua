from __future__ import annotations

from typing import Any


def latest_run_metrics(strategy: str, *, tracking_uri: str) -> dict[str, Any] | None:
    """Most recent MLflow run for `strategy` as a flat dict, or None if there are none.

    Reads the experiment named `strategy`. Works against a file backend — no server.
    A missing/unreadable tracking store (e.g. no `mlruns` yet on a fresh checkout) degrades
    to None ("no tracked runs"), never an error — this is a doc projection, not a query.
    """
    from mlflow.tracking import MlflowClient

    # An explicitly-scoped client avoids mlflow's process-global tracking-uri state, so a
    # relative `mlruns` always resolves against the caller's setting, not a stale one.
    # Metrics are an optional decoration on the doc: any read failure (missing store, stale
    # global state, unreadable backend) degrades to "no tracked runs", never crashes the
    # command.
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        exp = client.get_experiment_by_name(strategy)
        if exp is None:
            return None
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    except Exception:  # noqa: BLE001 - best-effort projection; absence of metrics is not fatal
        return None
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
