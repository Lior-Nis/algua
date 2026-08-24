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
from algua.registry.run_views import run_list_payload
from algua.registry.store import SqliteStrategyRepository

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
