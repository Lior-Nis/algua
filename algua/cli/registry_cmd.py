from __future__ import annotations

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import store
from algua.registry.approvals import record_approval

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


@registry_app.command("add")
@json_errors(ValueError, LookupError)
def add(name: str) -> None:
    """Register a new strategy at stage 'idea'."""
    with registry_conn() as conn:
        rec = store.add_strategy(conn, name)
    emit(ok({"id": rec.id, "name": rec.name, "stage": rec.stage.value}))


@registry_app.command("list")
@json_errors(ValueError, LookupError)
def list_(stage: str = typer.Option(None, "--stage", help="filter by stage")) -> None:
    """List strategies, optionally filtered by stage. Emits a bare JSON array (collection)."""
    st = Stage(stage) if stage else None
    with registry_conn() as conn:
        recs = store.list_strategies(conn, st)
    emit([{"id": r.id, "name": r.name, "stage": r.stage.value} for r in recs])


@registry_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    with registry_conn() as conn:
        rec = store.get_strategy(conn, name)
        transitions = store.list_transitions(conn, name)
    emit(ok({"id": rec.id, "name": rec.name, "stage": rec.stage.value,
             "transitions": transitions}))


@registry_app.command("transition")
@json_errors(ValueError, LookupError)
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
    code_hash: str = typer.Option(None, "--code-hash"),
    config_hash: str = typer.Option(None, "--config-hash"),
) -> None:
    """Advance a strategy to a new lifecycle stage."""
    with registry_conn() as conn:
        rec = store.transition(conn, name, Stage(to), Actor(actor), reason, code_hash, config_hash)
    emit(ok({"name": rec.name, "stage": rec.stage.value}))


@registry_app.command("approve")
@json_errors(ValueError, LookupError)
def approve(
    name: str,
    code_hash: str = typer.Option(..., "--code-hash"),
    config_hash: str = typer.Option(..., "--config-hash"),
    by: str = typer.Option(..., "--by", help="human approver identity"),
) -> None:
    """Record a human approval binding code+config hashes (required for going live)."""
    with registry_conn() as conn:
        aid = record_approval(conn, name, code_hash, config_hash, by)
    emit(ok({"approval_id": aid}))
