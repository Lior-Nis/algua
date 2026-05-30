from __future__ import annotations

import functools
import sqlite3
from collections.abc import Callable

import typer

from algua.cli.app import app, emit
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import store
from algua.registry.approvals import record_approval
from algua.registry.db import connect, migrate

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


def _conn() -> sqlite3.Connection:
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _json_errors(fn: Callable[..., None]) -> Callable[..., None]:
    """Emit expected domain errors as JSON ({"ok": false, "error": ...}) + exit 1,
    preserving the agent-parseable stdout contract instead of leaking tracebacks."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, LookupError) as exc:
            emit({"ok": False, "error": str(exc)})
            raise typer.Exit(code=1) from exc
    return wrapper


@registry_app.command("add")
@_json_errors
def add(name: str) -> None:
    """Register a new strategy at stage 'idea'."""
    rec = store.add_strategy(_conn(), name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("list")
@_json_errors
def list_(stage: str = typer.Option(None, "--stage", help="filter by stage")) -> None:
    """List strategies, optionally filtered by stage."""
    st = Stage(stage) if stage else None
    recs = store.list_strategies(_conn(), st)
    emit([{"id": r.id, "name": r.name, "stage": r.stage.value} for r in recs])


@registry_app.command("show")
@_json_errors
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    conn = _conn()
    rec = store.get_strategy(conn, name)
    emit({"id": rec.id, "name": rec.name, "stage": rec.stage.value,
          "transitions": store.list_transitions(conn, name)})


@registry_app.command("transition")
@_json_errors
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
    code_hash: str = typer.Option(None, "--code-hash"),
    config_hash: str = typer.Option(None, "--config-hash"),
) -> None:
    """Advance a strategy to a new lifecycle stage."""
    rec = store.transition(
        _conn(), name, Stage(to), Actor(actor), reason, code_hash, config_hash
    )
    emit({"ok": True, "name": rec.name, "stage": rec.stage.value})


@registry_app.command("approve")
@_json_errors
def approve(
    name: str,
    code_hash: str = typer.Option(..., "--code-hash"),
    config_hash: str = typer.Option(..., "--config-hash"),
    by: str = typer.Option(..., "--by", help="human approver identity"),
) -> None:
    """Record a human approval binding code+config hashes (required for going live)."""
    aid = record_approval(_conn(), name, code_hash, config_hash, by)
    emit({"ok": True, "approval_id": aid})
