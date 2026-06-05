from __future__ import annotations

import re
from pathlib import Path

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry import live_gate
from algua.registry.approvals import compute_artifact_hashes, record_approval
from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, SignatureError
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

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
@json_errors(ValueError, LookupError, TransitionError, SignatureError)
def transition(
    name: str,
    to: str = typer.Option(..., "--to"),
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(None, "--reason"),
    signature: str = typer.Option(
        None, "--signature",
        help="path to the SSH signature over the printed go-live challenge"),
) -> None:
    """Advance a strategy's lifecycle stage. Going live is a two-step signed ceremony.

    Run with no --signature to print a challenge, sign it with your enrolled key,
    then re-run with --signature."""
    target = Stage(to)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        if target is Stage.LIVE and signature is None:
            if Actor(actor) is not Actor.HUMAN:
                raise TransitionError("transition to live requires a human actor")
            rec = repo.get(name)
            validate_transition(rec.stage, Stage.LIVE)  # reject non-paper before issuing
            identity = compute_artifact_hashes(name)
            issued = live_gate.issue_challenge(
                conn, rec.id, name, identity.code_hash, identity.config_hash,
                identity.dependency_hash)
            emit(ok({
                "action": "go_live_challenge", "strategy": name, **issued,
                "instructions": ("sign the 'challenge' value with your enrolled key: "
                                 "ssh-keygen -Y sign -n algua-go-live -f <key> <file>; "
                                 "then re-run this command with --signature <file>.sig"),
            }))
            return

        verifier = None
        approver: dict[str, str] = {}
        if target is Stage.LIVE:
            sig_bytes = Path(signature).read_bytes()

            def _verify(_repo: object, sid: int, ch: str, cfg: str, dep: str | None) -> bool:
                principal = live_gate.verify_and_consume(
                    conn, name, sid, ch, cfg, dep, sig_bytes, ALLOWED_SIGNERS_PATH)
                if principal is None:
                    return False
                approver["id"] = principal
                return True

            verifier = _verify

        rec = transition_strategy(repo, name, target, Actor(actor), reason,
                                  approval_verifier=verifier)
        if target is Stage.LIVE:
            record_approval(repo, name, approver["id"])
    emit(ok({"name": rec.name, "stage": rec.stage.value}))


@registry_app.command("enroll-approver")
@json_errors(ValueError)
def enroll_approver(
    name: str = typer.Option(..., "--name", help="approver identity (allowed_signers principal)"),
    pubkey: str = typer.Option(..., "--pubkey", help="SSH public key (ssh-ed25519 AAAA...)"),
) -> None:
    """Enroll a go-live approver PUBLIC key. The trust comes from committing this through code-owner
    review — the live gate uses the reviewed copy on main."""
    # Strict principal: a single token, so a crafted --name can't inject a second allowed_signers
    # line (e.g. a newline + an extra key) into the trust anchor (codex review).
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", name):
        raise ValueError("--name must be one token of [A-Za-z0-9_.@-] (no whitespace/newlines)")
    parts = pubkey.split()
    if len(parts) < 2 or not parts[0].startswith("ssh-"):
        raise ValueError("--pubkey must be an SSH public key, e.g. 'ssh-ed25519 AAAA... comment'")
    keytype, keyblob = parts[0], parts[1]
    # Dup check on the EXACT enrolled key blobs (parse each line), not a substring of the file —
    # a comment or a prefix blob must not cause a false match either way (codex review).
    enrolled: set[str] = set()
    if ALLOWED_SIGNERS_PATH.exists():
        for ln in ALLOWED_SIGNERS_PATH.read_text().splitlines():
            fields = ln.split()
            for i, tok in enumerate(fields):
                if tok.startswith("ssh-") and i + 1 < len(fields):
                    enrolled.add(fields[i + 1])
    if keyblob in enrolled:
        raise ValueError("that public key is already enrolled")
    line = f'{name} namespaces="algua-go-live" {keytype} {keyblob}\n'
    with ALLOWED_SIGNERS_PATH.open("a") as fh:
        fh.write(line)
    emit(ok({"enrolled": name, "keytype": keytype}))
