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
    # The OOS holdout is revealed in exactly ONE place: `research promote` (which burns it).
    # The knowledge doc is an operator-facing surface and must withhold holdout metrics,
    # even for runs logged before the writers were sealed (defense-in-depth).
    # Drop any key that is "holdout_metrics" or starts with "holdout." (the flattened shape
    # produced by the tracker's _flatten helper, e.g. "holdout.sharpe").
    raw_metrics = run.data.metrics
    filtered_metrics = {
        k: v
        for k, v in raw_metrics.items()
        if k != "holdout_metrics" and not k.lower().startswith("holdout.")
    }
    params = run.data.params
    return {
        "run_id": run.info.run_id,
        "kind": run.data.tags.get("kind", "unknown"),
        "snapshot_id": params.get("snapshot_id"),
        "seed": params.get("seed"),
        # Reproduction provenance (#333): the identity + coordinates needed to re-run this
        # exact experiment from the kb doc alone. Each degrades to None when the param is
        # absent (a run logged before the stamp existed); the render layer marks the gap.
        "config_hash": params.get("config_hash"),
        "code_hash": params.get("code_hash"),
        "dependency_hash": params.get("dependency_hash"),
        "period_start": params.get("period_start"),
        "period_end": params.get("period_end"),
        "timeframe": params.get("timeframe"),
        "universe_name": params.get("universe_name"),
        "metrics": filtered_metrics,
    }
