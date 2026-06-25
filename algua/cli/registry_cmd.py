from __future__ import annotations

import re
from pathlib import Path

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.knowledge.sync import sync_strategy_doc
from algua.registry import live_gate, transitions
from algua.registry.approvals import compute_artifact_hashes, record_approval
from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, SignatureError
from algua.registry.repository import StrategyRecord, kb_metadata
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

registry_app = typer.Typer(help="Strategy lifecycle registry", no_args_is_help=True)
app.add_typer(registry_app, name="registry")


def _record_json(r: StrategyRecord) -> dict:
    return {
        "id": r.id, "name": r.name, "stage": r.stage.value,
        "family": r.family, "tags": r.tags, "author": r.author.value,
        "hypothesis_status": r.hypothesis_status.value,
        "derived_from": r.derived_from, "description": r.description,
    }



@registry_app.command("add")
@json_errors(ValueError, LookupError)
def add(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family slug"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    author: Author = typer.Option(Author.AGENT, "--author", help="agent|human"),
    hypothesis_status: HypothesisStatus = typer.Option(
        HypothesisStatus.UNTESTED, "--hypothesis-status"),
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),
    description: str = typer.Option(None, "--description"),
) -> None:
    """Register a new strategy at stage 'idea' with organizational metadata."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).add(
            name, family=family, tags=tag or [], author=author,
            hypothesis_status=hypothesis_status, derived_from=derived_from,
            description=description,
        )
    emit(ok(_record_json(rec)))


@registry_app.command("list")
@json_errors(ValueError, LookupError)
def list_(
    stage: str = typer.Option(None, "--stage", help="filter by stage"),
    family: str = typer.Option(None, "--family", help="filter by thesis family"),
    tag: list[str] = typer.Option(None, "--tag", help="require this tag (repeatable, all-of)"),
    author: Author = typer.Option(None, "--author", help="filter by author (agent|human)"),
    hypothesis_status: HypothesisStatus = typer.Option(
        None, "--hypothesis-status", help="filter by hypothesis status"),
) -> None:
    """List strategies with optional filters (AND-ed). Emits a bare JSON array."""
    st = Stage(stage) if stage else None
    with registry_conn() as conn:
        recs = SqliteStrategyRepository(conn).list_strategies(
            st, family=family, tags=tag or [], author=author,
            hypothesis_status=hypothesis_status,
        )
    emit([_record_json(r) for r in recs])


@registry_app.command("show")
@json_errors(ValueError, LookupError)
def show(name: str) -> None:
    """Show a strategy and its transition history."""
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)
        transitions = repo.list_transitions(name)
    emit(ok({**_record_json(rec), "transitions": transitions}))


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
        if target is Stage.LIVE:
            from algua.registry import allocations
            rec = repo.get(name)
            if allocations.active_allocation(conn, rec.id) is None:
                raise TransitionError(
                    f"{name} has no live allocation; run `algua live allocate {name} --capital X` "
                    "before going live.")
        if target is Stage.LIVE and signature is None:
            if Actor(actor) is not Actor.HUMAN:
                raise TransitionError("transition to live requires a human actor")
            rec = repo.get(name)
            validate_transition(rec.stage, Stage.LIVE)  # reject non-paper before issuing
            # The certificate check BEFORE any challenge is issued, so the human signs with
            # the evidence in front of them (#124). Built via the same transitions helper the
            # signature-completion wall uses — no duplicate logic. Module-attribute access on
            # purpose: it is the one monkeypatch seam shared by both paths. ONE identity
            # computation serves both the verifier and the challenge — no drift window
            # between what the certificate was judged against and what gets signed (#124 GATE-2).
            identity = compute_artifact_hashes(name)
            certificate = transitions._default_forward_certificate_verifier()(
                repo, name, rec.id, identity)
            issued = live_gate.issue_challenge(
                conn, rec.id, name, identity.code_hash, identity.config_hash,
                identity.dependency_hash)
            emit(ok({
                "action": "go_live_challenge", "strategy": name, **issued,
                "forward_certificate": certificate,
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


@registry_app.command("set")
@json_errors(ValueError, LookupError)
def set_(
    name: str,
    family: str = typer.Option(None, "--family"),
    author: Author = typer.Option(None, "--author"),
    hypothesis_status: HypothesisStatus = typer.Option(None, "--hypothesis-status"),
    derived_from: str = typer.Option(None, "--derived-from"),
    description: str = typer.Option(None, "--description"),
    add_tag: list[str] = typer.Option(None, "--add-tag", help="add a tag (repeatable)"),
    remove_tag: list[str] = typer.Option(None, "--remove-tag", help="remove a tag (repeatable)"),
) -> None:
    """Update a strategy's organizational metadata (never its stage); re-syncs the kb doc."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    fields = ("family", "author", "hypothesis_status", "derived_from", "description", "tags")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        before = repo.get(name)
        after = repo.update_metadata(
            name, family=family, author=author, hypothesis_status=hypothesis_status,
            derived_from=derived_from, description=description,
            add_tags=add_tag or [], remove_tags=remove_tag or [],
        )
    changed = {}
    for f in fields:
        b, a = getattr(before, f), getattr(after, f)
        if isinstance(b, (Author, HypothesisStatus)):
            b = b.value
        if isinstance(a, (Author, HypothesisStatus)):
            a = a.value
        if b != a:
            changed[f] = {"before": b, "after": a}
    # Re-sync the kb doc so frontmatter reflects the new registry truth.
    # Best-effort: absent doc is ok (sync_strategy_doc returns False).
    sync_strategy_doc(get_settings(), name, stage=after.stage.value, metadata=kb_metadata(after))
    emit(ok({**_record_json(after), "changed": changed}))


@registry_app.command("backfill-from-kb")
@json_errors(ValueError, LookupError)
def backfill_from_kb() -> None:
    """One-shot: recover kb-authored metadata into NULL registry columns; report conflicts.

    Fills only currently-NULL columns. kb hypothesis_status/author values that aren't valid enum
    members are reported as 'unmappable' and left for the operator. Finally default-fills any row
    still NULL on author/hypothesis_status/tags."""
    from algua.knowledge.frontmatter import parse_doc
    from algua.knowledge.sync import _unwikilink, strategy_doc_path

    settings = get_settings()
    processed: list[str] = []
    unmappable: list[dict] = []
    kb_without_row: list[str] = []
    rows_without_kb: list[str] = []
    frontmatter_name_mismatches: list[dict] = []
    valid_status = {h.value for h in HypothesisStatus}
    valid_author = {a.value for a in Author}
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        names = {r.name for r in repo.list_strategies()}
        # Rows with no kb doc
        for name in names:
            if not strategy_doc_path(settings, name).exists():
                rows_without_kb.append(name)
        # kb docs -> registry backfill
        strat_dir = settings.knowledge_dir / "strategies"
        if strat_dir.exists():
            for doc in sorted(strat_dir.glob("*.md")):
                if doc.name.startswith("_"):
                    continue
                fm, _ = parse_doc(doc.read_text())
                if fm.get("type") == "family":
                    continue
                # The filename (doc.stem) is the authoritative registry key.
                # A hand-edited frontmatter `name:` that differs from the filename would
                # mis-attribute the backfill, so we always use doc.stem.
                doc_name = doc.stem
                fm_name = fm.get("name")
                if fm_name is not None and str(fm_name) != doc_name:
                    frontmatter_name_mismatches.append(
                        {"file": doc.name, "frontmatter_name": str(fm_name)}
                    )
                if doc_name not in names:
                    kb_without_row.append(doc_name)
                    continue
                status = fm.get("hypothesis_status")
                author = fm.get("author")
                if status is not None and status not in valid_status:
                    unmappable.append({"name": doc_name, "field": "hypothesis_status",
                                       "value": status})
                    status = None
                if author is not None and author not in valid_author:
                    unmappable.append({"name": doc_name, "field": "author", "value": author})
                    author = None
                derived_from = _unwikilink(fm.get("derived_from"))
                # Validate derived_from: reject self-reference or unknown strategy name.
                if derived_from:
                    if derived_from == doc_name or derived_from not in names:
                        unmappable.append(
                            {"name": doc_name, "field": "derived_from", "value": derived_from}
                        )
                        derived_from = None
                tags = fm.get("tags")
                repo.backfill_metadata(
                    doc_name,
                    family=_unwikilink(fm.get("family")),
                    derived_from=derived_from,
                    hypothesis_status=status,
                    author=author,
                    description=fm.get("description"),
                    tags=list(tags) if isinstance(tags, list) else None,
                )
                processed.append(doc_name)
        # Final default-fill of any remaining NULLs — delegated to the repository.
        repo.default_fill_metadata_nulls()
    # `processed` = strategies whose kb doc was found and reconciled into the registry.
    # Fill-only-NULL means already-populated columns are left untouched (not a "changed" list).
    emit(ok({
        "processed": sorted(processed),
        "unmappable": unmappable,
        "kb_docs_without_registry_row": sorted(kb_without_row),
        "registry_rows_without_kb_doc": sorted(rows_without_kb),
        "frontmatter_name_mismatches": frontmatter_name_mismatches,
    }))


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
