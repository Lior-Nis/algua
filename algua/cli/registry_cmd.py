from __future__ import annotations

from pathlib import Path

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.approvals import record_approval
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

ALLOWED_SIGNERS_PATH = Path("approvers/allowed_signers")

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


@registry_app.command("add")
@json_errors(ValueError, LookupError)
def add(name: str) -> None:
    """Register a new strategy at stage 'idea'."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).add(name)
    emit(ok({"id": rec.id, "name": rec.name, "stage": rec.stage.value}))


@registry_app.command("list")
@json_errors(ValueError, LookupError)
def list_(stage: str = typer.Option(None, "--stage", help="filter by stage")) -> None:
    """List strategies, optionally filtered by stage. Emits a bare JSON array (collection)."""
    st = Stage(stage) if stage else None
    with registry_conn() as conn:
        recs = SqliteStrategyRepository(conn).list_strategies(st)
    emit([{"id": r.id, "name": r.name, "stage": r.stage.value} for r in recs])


@registry_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)
        transitions = repo.list_transitions(name)
    emit(ok({"id": rec.id, "name": rec.name, "stage": rec.stage.value,
             "transitions": transitions}))


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
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = transition_strategy(repo, name, Stage(to), Actor(actor), reason)
    emit(ok({"name": rec.name, "stage": rec.stage.value}))


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
    with registry_conn() as conn:
        aid = record_approval(SqliteStrategyRepository(conn), name, by)
    emit(ok({"approval_id": aid}))


@registry_app.command("enroll-approver")
@json_errors(ValueError)
def enroll_approver(
    name: str = typer.Option(..., "--name", help="approver identity (allowed_signers principal)"),
    pubkey: str = typer.Option(..., "--pubkey", help="SSH public key (ssh-ed25519 AAAA...)"),
) -> None:
    """Enroll a go-live approver PUBLIC key. The trust comes from committing this through code-owner
    review — the live gate uses the reviewed copy on main."""
    if not name.strip():
        raise ValueError("--name must not be empty")
    parts = pubkey.split()
    if len(parts) < 2 or not parts[0].startswith("ssh-"):
        raise ValueError("--pubkey must be an SSH public key, e.g. 'ssh-ed25519 AAAA... comment'")
    keytype, keyblob = parts[0], parts[1]
    existing = ALLOWED_SIGNERS_PATH.read_text() if ALLOWED_SIGNERS_PATH.exists() else ""
    if keyblob in existing:
        raise ValueError("that public key is already enrolled")
    line = f'{name} namespaces="algua-go-live" {keytype} {keyblob}\n'
    with ALLOWED_SIGNERS_PATH.open("a") as fh:
        fh.write(line)
    emit(ok({"enrolled": name, "keytype": keytype}))
