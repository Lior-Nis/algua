from __future__ import annotations

import sqlite3
from contextlib import closing

import typer

from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.approvals import record_approval
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


def _conn() -> sqlite3.Connection:
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


@registry_app.command("add")
@json_errors(ValueError, LookupError)
def add(name: str) -> None:
    """Register a new strategy at stage 'idea'."""
    with closing(_conn()) as conn:
        rec = SqliteStrategyRepository(conn).add(name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("list")
@json_errors(ValueError, LookupError)
def list_(stage: str = typer.Option(None, "--stage", help="filter by stage")) -> None:
    """List strategies, optionally filtered by stage."""
    st = Stage(stage) if stage else None
    with closing(_conn()) as conn:
        recs = SqliteStrategyRepository(conn).list_strategies(st)
    emit([{"id": r.id, "name": r.name, "stage": r.stage.value} for r in recs])


@registry_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    with closing(_conn()) as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)
        transitions = repo.list_transitions(name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value,
          "transitions": transitions})


@registry_app.command("transition")
@json_errors(ValueError, LookupError)
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
) -> None:
    """Advance a strategy to a new lifecycle stage.

    Going live pins the *recomputed* code+config hash of the loaded strategy and requires a
    matching human approval; callers cannot supply the hashes.
    """
    with closing(_conn()) as conn:
        repo = SqliteStrategyRepository(conn)
        rec = transition_strategy(repo, name, Stage(to), Actor(actor), reason)
    emit({"ok": True, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("approve")
@json_errors(ValueError, LookupError)
def approve(
    name: str,
    by: str = typer.Option(..., "--by", help="human approver identity"),
) -> None:
    """Record a human approval pinning the strategy's current code+config (required for live).

    The approved hashes are computed from the live strategy source and config, so the approval
    binds to the exact artifact rather than to operator-supplied strings.
    """
    with closing(_conn()) as conn:
        aid = record_approval(SqliteStrategyRepository(conn), name, by)
    emit({"ok": True, "approval_id": aid})
