"""``research run-all`` — a long-lived batch worker for the research funnel (#326).

Every research CLI action (``backtest run``, ``backtest sweep``, ``research promote``) is normally
one ``uv run algua`` process, each paying a ~1.5 CPU-s cold start (Python interpreter + importing
the vectorbt/numba backtest stack + opening the DB). At hundreds of strategies x 3-4 actions/day
that per-call overhead — not the statistics — becomes the throughput ceiling.

``research run-all --tasks tasks.json`` imports the heavy stack ONCE and loops over a queue of
tasks in ONE warm process, amortizing the cold-start tax. It mirrors the live/paper ``run-all``
batch-driver shape with PER-TASK FAULT ISOLATION (#374): one bad task becomes an ``ok:false`` error
marker and the loop CONTINUES — a single failing strategy never starves its siblings.

Warm-process state hygiene (so a batch is result-identical to running the tasks as separate cold
processes): each task (a) reloads the strategy module (``reload=True``) so a strategy's own
module-level globals never leak forward, and (b) restores ``construction._POLICIES`` to its
pristine snapshot, the one mutable shared strategy-dependency global. The heavy vectorbt/numba
stack stays warm. The holdout single-use guard and gate-token minting are DB rows (reserved under
BEGIN IMMEDIATE), so process reuse reuses NOTHING there — a second promote on an already-burned
window fails closed exactly as two separate processes would.

Isolation contract (what a warm task is guaranteed vs a cold process, and what it is NOT): a task's
RESULT is bit-identical to a cold run because the backtest path is deterministic in a LOCAL seed
(``SyntheticProvider`` uses ``np.random.default_rng(seed)``, never global RNG; real data is
immutable snapshot bars) and the strategy dependency surface — ``algua.features`` / ``contracts`` /
``portfolio`` — is import-linter-enforced PURE (no mutable first-party globals) EXCEPT the one
mutable ``construction._POLICIES`` registry, which is reset per task here. Beyond that, the worker
does NOT sandbox arbitrary process state (cwd, env vars, ``sys.path``, pandas/numpy options, or a
strategy that adversarially rebinds another module's ``__code__``): a strategy that reaches out and
mutates unrelated process globals is out of scope — the same way it would be for any in-process
host — and full subprocess isolation would defeat the amortization this issue exists to deliver.

This module is deliberately NOT one of the `independence`-contracted cli command modules: it is a
composition point that imports the reusable task bodies from ``backtest_cmd``/``research_cmd`` (the
same one-way direction ``main.py`` uses to mount sub-typers), so no listed command module imports a
sibling.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.backtest_cmd import run_backtest_task, sweep_task
from algua.cli.errors import error_code, json_errors
from algua.cli.research_cmd import promote_task
from algua.portfolio import construction

run_all_app = typer.Typer(help="Research batch worker", no_args_is_help=True)

# Per-action dispatch: action name -> the reusable task body. Each returns the payload dict the
# single command would have wrapped in ok(...). The set is CLOSED (an unknown action fails
# validation up front) and each entry ALWAYS runs with reload=True (warm-worker hygiene).
_DISPATCH: dict[str, Callable[..., dict]] = {
    "backtest": run_backtest_task,
    "sweep": sweep_task,
    "promote": promote_task,
}

# Allowed parameter keys per action (besides the required "action" + "name"). Rejecting unknown
# keys up front keeps JSON tasks from silently drifting from the single-command flag surface, and
# structurally excludes side-effecting flags the batch must not honor: `track` (would leak an
# MLflow active run in a warm process) and the CLI-only `summary` projection.
_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "backtest": frozenset({
        "start", "end", "demo", "snapshot", "universe", "fundamentals_snapshot",
        "news_snapshot", "delistings", "assume_terminal_last_close", "register", "emit_series",
    }),
    "sweep": frozenset({
        "start", "end", "demo", "snapshot", "universe", "windows", "holdout_frac", "param",
        "rank_by", "top", "fundamentals_snapshot", "news_snapshot", "delistings",
        "assume_terminal_last_close",
    }),
    "promote": frozenset({
        "start", "end", "demo", "snapshot", "fundamentals_snapshot", "news_snapshot", "universe",
        "windows", "holdout_frac", "min_holdout_sharpe", "min_holdout_return", "min_pct_positive",
        "min_window_sharpe", "n_combos", "allow_holdout_reuse", "allow_non_pit", "delistings",
        "assume_terminal_last_close", "actor", "new_family",
    }),
}


def _validate_tasks(raw: Any) -> list[dict[str, Any]]:
    """Structurally validate the parsed tasks file BEFORE running any task. A malformed file fails
    the WHOLE batch (a clean JSON error envelope), rather than surfacing as inconsistent per-task
    markers. Requires a JSON array of objects, each with a known `action` and a string `name`, and
    only allowed per-action keys (unknown keys — including `track`/`summary` — are rejected)."""
    if not isinstance(raw, list):
        raise ValueError("tasks file must be a JSON array of task objects")
    if not raw:
        raise ValueError("tasks file is empty (no tasks to run)")
    tasks: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"task {i} is not an object")
        action = item.get("action")
        if action not in _DISPATCH:
            raise ValueError(
                f"task {i} has unknown action {action!r}; expected one of {sorted(_DISPATCH)}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"task {i} ({action}) is missing a non-empty string 'name'")
        extra = set(item) - {"action", "name"} - _ALLOWED_KEYS[action]
        if extra:
            raise ValueError(
                f"task {i} ({action} {name}) has unsupported keys {sorted(extra)}; "
                f"allowed: {sorted(_ALLOWED_KEYS[action])} (note: `track`/`summary` are "
                f"single-command-only and rejected in a batch)")
        tasks.append(item)
    return tasks


@run_all_app.callback(invoke_without_command=True)
@json_errors
def run_all(
    tasks: str = typer.Option(
        ..., "--tasks",
        help="path to a JSON file: an array of {action, name, ...params} research tasks"),
) -> None:
    """Run a batch of research tasks in ONE warm process, amortizing the cold-start import tax.

    ``--tasks`` is a JSON array; each object is ``{"action": "backtest"|"sweep"|"promote",
    "name": "<strategy>", ...params}`` where params mirror the single command's flags (`track` and
    `summary` are single-command-only and rejected here). Tasks run IN FILE ORDER and share the
    funnel-breadth / FDR / holdout-burn ledgers EXACTLY as running them back-to-back as separate
    processes would (a promote task sees breadth a sibling sweep just recorded). Each task is
    fault-isolated: a task-domain failure becomes an ``{"ok": false, ..., "code": ...}`` marker and
    the loop continues. The envelope's ``ok`` is true iff EVERY task succeeded; the process exits
    non-zero when any task erred (meaningful-exit-code convention)."""
    path = Path(tasks)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"tasks file not found: {tasks}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"tasks file is not valid JSON: {exc}") from exc
    parsed = _validate_tasks(raw)

    # Snapshot the one mutable shared strategy-dependency global ONCE, before any task runs, so
    # each task starts from a pristine construction registry even if a prior task's strategy
    # mutated it (Codex GATE-1: construction._POLICIES is reachable+mutable from any strategy).
    pristine_policies = copy.copy(construction._POLICIES)

    results: list[dict[str, Any]] = []
    error_count = 0
    for task in parsed:
        action = task["action"]
        name = task["name"]
        params = {k: v for k, v in task.items() if k not in ("action", "name")}
        # Per-task reset boundary: restore the construction policy registry to pristine (undoes any
        # in-process mutation a prior task's strategy made). The strategy module itself is reloaded
        # inside the task via reload=True.
        construction._POLICIES.clear()
        construction._POLICIES.update(pristine_policies)
        try:
            payload = _DISPATCH[action](name, reload=True, **params)
            results.append({"ok": True, "action": action, "name": name, **payload})
        except (KeyboardInterrupt, SystemExit):
            raise  # process-integrity signals abort the batch (never a per-task marker)
        except Exception as exc:  # noqa: BLE001 - per-task isolation: one bad task must not abort the batch (#374)
            error_count += 1
            results.append({
                "ok": False, "action": action, "name": name,
                "error": str(exc), "code": error_code(exc), "kind": "task_error",
            })

    envelope = ok({
        "tasks_total": len(parsed),
        "ok_count": len(parsed) - error_count,
        "error_count": error_count,
        "results": results,
    })
    if error_count:
        envelope["ok"] = False  # meaningful exit code + machine-readable partial-failure signal
        emit(envelope)
        raise typer.Exit(code=1)
    emit(envelope)
