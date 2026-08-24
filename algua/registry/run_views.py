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

import numpy as np

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

    NEVER returns a series: each row is the run's scalar columns plus its three JSON-TEXT columns
    parsed (`derived_from`/`components`/`config_json`) — the `runs series` payload-size contract
    (#349) that a subprocess-JSON CLI seam depends on.
    """
    rows = [_parsed_row(row) for row in repo.list_runs(
        kind=kind, strategy_name=strategy, sort=sort, limit=limit)]
    if family is not None:
        # `list_runs` has no `family` parameter and deliberately stays that way: `runs` is keyed
        # by free-text strategy_name with a NULLABLE strategy_id (exploration precedes
        # registration — see algua/registry/db/runs.py), so there is no FK to join `family` off
        # inside list_runs's own query without complicating the one thing it must stay the single
        # gate for (`sort`). `registry list --family` resolves family via a plain
        # `strategies.family = ?` filter (algua/registry/store/crud.py `list_strategies`); reused
        # here as ONE cheap SELECT for the strategy names in that family, then filtered in Python
        # over the rows `list_runs` already returned. Honest tradeoff: this filter runs AFTER
        # list_runs's own LIMIT, so a narrow `--limit` combined with a rare family can under-return
        # relative to a true server-side join — acceptable for a display cap, not a pagination
        # guarantee. A run for a strategy with no `strategies` row at all (also exploration-before-
        # registration) has no family and is correctly excluded by any family filter.
        names = {
            r["name"] for r in repo.connection.execute(
                "SELECT name FROM strategies WHERE family = ?", (family,)
            ).fetchall()
        }
        rows = [r for r in rows if r["strategy_name"] in names]
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
    """`runs series` — the ONLY one of the three that returns a return series.

    Resolves each run's `series_backtest_id`/`series_holdout_id` pointer. A run with NO pointer
    maps to `None`, not an empty list — the two mean different things ("this run has no series"
    vs "this run's series is empty"). A run id that does not exist at all is a caller bug, not a
    legitimately-absent series, so it raises the same `ValueError(f"no run {run_id}")` shape
    `run_detail_payload` does.
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
        hr = conn.execute(
            "SELECT holdout_start, holdout_end, n_bars, returns_blob, bar_dates_blob"
            " FROM holdout_returns WHERE id = ?",
            (row["series_holdout_id"],),
        ).fetchone()
        if hr is None:
            return None
        vec = np.frombuffer(hr["returns_blob"], dtype="<f8")
        dates = hr["bar_dates_blob"].decode("utf-8").split("\n")
        return {
            "kind": "holdout",
            "holdout_start": hr["holdout_start"],
            "holdout_end": hr["holdout_end"],
            "returns": [[d, float(v)] for d, v in zip(dates, vec, strict=True)],
        }
    return None
