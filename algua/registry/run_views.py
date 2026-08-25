"""Pure projections behind the run-ledger read surface (`runs list`/`show`/`series`, slice 2).

Mirrors the #165 domain-extraction convention `algua/cli/ops_cmd.py`'s docstring states: the
shaping logic lives here, not in the CLI, so it is unit-testable without a subprocess. All three
functions take the ``SqliteStrategyRepository`` ALONE and reach sqlite through ``repo.connection``
(the read-only handle the repo already exposes for exactly this) — never a second, separately
passed connection.

Pure reads only: no writes, no locks, no subprocess.

Payload-size contract (#349): ``run_list_payload`` is scalars-only and must NEVER include a return
series — ``run_series_payload`` is the one function that does. ``run_detail_payload`` adds the
``run_metrics`` overflow tail, parsed lineage, and — for a `gate` run — the gate decision, but
still never a return series.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from algua.registry.gate_history import GATE_DECISION_ALLOWLIST, _project_decision
from algua.registry.store import SqliteStrategyRepository

__all__ = ["run_detail_payload", "run_list_payload", "run_series_payload"]


def _parsed_row(row: sqlite3.Row) -> dict[str, Any]:
    """A run row as a dict with its three JSON-TEXT columns parsed in place. Column names are
    kept identical to the schema (`derived_from` / `components` / `config_json`) — this is a
    type change on the same key, not a rename."""
    out = dict(row)
    out["derived_from"] = json.loads(out["derived_from"])
    out["components"] = json.loads(out["components"])
    out["config_json"] = json.loads(out["config_json"])
    return out


def run_list_payload(
    repo: SqliteStrategyRepository,
    *,
    kind: str | None,
    strategy: str | None,
    family: str | None,
    sort: str | None,
    limit: int,
) -> dict[str, Any]:
    """`runs list` — scalar run rows, newest-first (or best-first when `sort` names a metric).

    `sort` is passed straight to `list_runs`, whose `METRIC_COLUMNS` allow-list is the SINGLE gate
    for it (raises `ValueError` on a non-vocabulary value) — this function adds no second, looser
    check of its own.

    `family` is resolved to its member strategy names FIRST (via `repo.list_strategies(family=...)`
    — the same accessor `registry list --family` uses, so the semantics match exactly) and passed
    into `list_runs` as a parameterized `strategy_name IN (...)` clause, so the SQL-side LIMIT
    applies to the already-family-filtered set. Filtering post-hoc in Python over rows `list_runs`
    already truncated to `limit` would invert `--sort`'s semantics: it would return the globally
    best N runs that happen to be in the family, which for a small family is systematically empty
    rather than merely truncated. A run for a strategy with no `strategies` row at all (exploration
    precedes registration) has no family and is correctly excluded by any family filter, since it
    can never appear in `list_strategies(family=...)`'s result.

    NEVER returns a series: each row is the run's scalar columns plus its three JSON-TEXT columns
    parsed (`derived_from`/`components`/`config_json`) — the `runs series` payload-size contract
    (#349) that a subprocess-JSON CLI seam depends on.
    """
    strategy_names = (
        [rec.name for rec in repo.list_strategies(family=family)] if family is not None else None
    )
    rows = [_parsed_row(row) for row in repo.list_runs(
        kind=kind, strategy_name=strategy, strategy_names=strategy_names, sort=sort, limit=limit)]
    return {"runs": rows, "count": len(rows)}


def run_detail_payload(repo: SqliteStrategyRepository, run_id: int) -> dict[str, Any]:
    """`runs show` — one run plus its `run_metrics` overflow tail, parsed lineage, and — for a
    `gate` run with a `gate_id` — the allow-list-projected gate decision.

    Raises `ValueError(f"no run {run_id}")` when the run does not exist; the CLI's `@json_errors`
    turns that into the standard error envelope.
    """
    row = repo.get_run(run_id)
    if row is None:
        raise ValueError(f"no run {run_id}")
    payload = _parsed_row(row)
    metric_rows = repo.connection.execute(
        "SELECT key, value FROM run_metrics WHERE run_id = ?", (run_id,)
    ).fetchall()
    payload["extra_metrics"] = {r["key"]: r["value"] for r in metric_rows}
    if payload["kind"] == "gate" and payload["gate_id"] is not None:
        # `decision_json` is NEVER emitted raw — reuse gate_history.py's two-layer allowlist
        # projection (the same one `registry gates` uses) rather than re-deriving it: a
        # hand-rolled second copy is exactly how the allowlist would drift.
        gate_row = repo.connection.execute(
            "SELECT decision_json FROM gate_evaluations WHERE id = ?", (payload["gate_id"],)
        ).fetchone()
        if gate_row is not None:
            projected = _project_decision(gate_row["decision_json"], GATE_DECISION_ALLOWLIST)
            payload["gate_decision"] = projected["decision"]
            if "decision_dropped_keys" in projected:
                payload["gate_decision_dropped_keys"] = projected["decision_dropped_keys"]
            if "decision_error" in projected:
                payload["gate_decision_error"] = projected["decision_error"]
    return payload


def run_series_payload(repo: SqliteStrategyRepository, run_ids: list[int]) -> dict[str, Any]:
    """`runs series` — the ONLY one of the three that returns a per-bar return series, and only
    for the IN-SAMPLE backtest leg.

    Resolves each run's `series_backtest_id`/`series_holdout_id` pointer. A run with NO pointer
    maps to `None`, not an empty list — the two mean different things ("this run has no series"
    vs "this run's series is empty"). A run id that does not exist at all is a caller bug, not a
    legitimately-absent series, so it raises the same `ValueError(f"no run {run_id}")` shape
    `run_detail_payload` does.

    A `series_holdout_id` pointer does NOT yield a per-bar vector: `holdout_returns.returns_blob`
    is SENSITIVE (see the DDL comment in `algua/registry/db/holdout.py` and the "ONLY method that
    reads returns_blob" docstring on `overlapping_holdout_return_streams` in
    `algua/registry/store/holdout.py`) — exposing a strategy's own OOS vector would re-open the
    single-use best-of-N surface `sweep()`'s holdout burn exists to prevent. So the holdout branch
    returns only the interval and `n_bars` (safe to shade a chart's OOS region with); the scalar
    OOS metrics already live on the gate run row (`run_detail_payload`).
    """
    series: dict[str, Any] = {}
    for run_id in run_ids:
        row = repo.get_run(run_id)
        if row is None:
            raise ValueError(f"no run {run_id}")
        series[str(run_id)] = _series_entry(repo.connection, row)
    return {"series": series}


def _series_entry(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any] | None:
    if row["series_backtest_id"] is not None:
        br = conn.execute(
            "SELECT period_start, period_end, returns_json FROM backtest_returns WHERE id = ?",
            (row["series_backtest_id"],),
        ).fetchone()
        if br is None:
            return None
        return {
            "kind": "backtest",
            "period_start": br["period_start"],
            "period_end": br["period_end"],
            "returns": json.loads(br["returns_json"]),
        }
    if row["series_holdout_id"] is not None:
        # SENSITIVE per algua/registry/db/holdout.py's holdout_returns DDL comment: "no CLI
        # accessor and no 'get my own vector' API may read returns_blob — sibling-only
        # cross-strategy." algua/registry/store/holdout.py's overlapping_holdout_return_streams
        # is documented as "the ONLY method that reads returns_blob" and must stay that way — a
        # per-bar OOS vector is single-use selection surface (the thing sweep()'s holdout burn
        # exists to protect); a scalar sharpe_oos leaks far less than the full vector. So this
        # branch returns ONLY the interval/count context (safe to shade a chart's OOS region
        # with), never returns_blob or bar_dates_blob. The scalar OOS metrics (sharpe_oos,
        # total_return_oos, n_obs_oos) already live on the gate run row — see run_detail_payload.
        hr = conn.execute(
            "SELECT holdout_start, holdout_end, n_bars FROM holdout_returns WHERE id = ?",
            (row["series_holdout_id"],),
        ).fetchone()
        if hr is None:
            return None
        return {
            "kind": "holdout",
            "holdout_start": hr["holdout_start"],
            "holdout_end": hr["holdout_end"],
            "n_bars": hr["n_bars"],
        }
    return None
