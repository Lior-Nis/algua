"""`algua runs list` — the read surface over the run ledger (spec:
2026-08-24-strategy-run-tracking-slice-2).

Pure read: no broker call, no writes, no locks. Thin command body — the shaping logic lives in
``algua.registry.run_views`` so it is unit-testable without a CLI, matching the domain-extraction
convention (#165, see ``algua/cli/ops_cmd.py``).
"""

from __future__ import annotations

import typer

from algua.cli._common import ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.registry.db import registry_conn
from algua.registry.run_views import run_detail_payload, run_list_payload, run_series_payload
from algua.registry.store import SqliteStrategyRepository

#: `runs series` payload-size guardrail (#349 lesson): this is the one command whose output can
#: carry a per-bar backtest return vector through a subprocess pipe that gets JSON-parsed. Cap the
#: id count rather than the byte count — cheap to check before any query runs.
MAX_SERIES_RUN_IDS = 16

runs_app = typer.Typer(help="The run ledger: every backtest/walk-forward/sweep/gate evaluation",
                       no_args_is_help=True)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
@json_errors
def runs_list(
    kind: str | None = typer.Option(None, "--kind", help="Filter to one run kind."),
    strategy: str | None = typer.Option(None, "--strategy", help="Filter to one strategy name."),
    family: str | None = typer.Option(None, "--family", help="Filter to one strategy family."),
    sort: str | None = typer.Option(
        None, "--sort",
        help="Order best-first by a metric column (NULLs last); default is newest-first.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, help="Maximum rows to return."),
) -> None:
    """List run-ledger rows, newest-first (or best-first when `--sort` names a metric).

    An empty ledger is `ok` with zero rows, not an error — the store ships empty and accumulates.
    Scalars only: never returns a per-bar return series (`runs series` is the one command that
    does).
    """
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        payload = run_list_payload(
            repo, kind=kind, strategy=strategy, family=family, sort=sort, limit=limit)
    emit(ok(payload))


@runs_app.command("show")
@json_errors
def runs_show(
    run_id: int = typer.Argument(..., help="The run id to show."),
) -> None:
    """Show one run row plus its `run_metrics` overflow tail, parsed lineage, and — for a `gate`
    run — its allow-list-projected gate decision.

    A missing run id raises `ValueError` in `run_detail_payload`, which `@json_errors` renders as
    the standard JSON error envelope and a non-zero exit — no pre-check, no different error shape.
    """
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        payload = run_detail_payload(repo, run_id)
    emit(ok(payload))


@runs_app.command("series")
@json_errors
def runs_series(
    run_id: int = typer.Argument(..., help="A run id to fetch the series pointer for."),
    extra_run_ids: list[int] = typer.Option(  # noqa: B008 — typer's documented Option-as-default
        [], "--run-id", help="Additional run ids (repeatable)."),
) -> None:
    """Resolve each run's series pointer (backtest returns or holdout interval context).

    A run with no series pointer maps to `null`, never an omitted key — an omitted key is
    indistinguishable from a typo'd id. Ids are de-duplicated and capped at
    `MAX_SERIES_RUN_IDS`: this is the one command whose payload can carry a per-bar return vector
    through a subprocess pipe that gets JSON-parsed, so an unbounded id list is a payload-size
    footgun (#349), not just a slow query.
    """
    ids = list(dict.fromkeys([run_id, *extra_run_ids]))
    if len(ids) > MAX_SERIES_RUN_IDS:
        raise ValueError(
            f"too many run ids: got {len(ids)}, max {MAX_SERIES_RUN_IDS} per call")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        payload = run_series_payload(repo, ids)
    emit(ok(payload))
