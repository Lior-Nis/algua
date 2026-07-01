from __future__ import annotations

import re

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.contracts.idea import DataCapability, IdeaStatus, SourceType
from algua.data.capabilities import supported_capabilities
from algua.registry.ideas import Collision, IdeaRepository
from algua.registry.store import SqliteStrategyRepository
from algua.research.ideas import classify_status

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

idea_app = typer.Typer(
    help="Idea pool: source, dedup, and park research hypotheses", no_args_is_help=True)


def _idea_json(idea) -> dict:
    return {
        "id": idea.id, "title": idea.title, "hypothesis": idea.hypothesis,
        "family": idea.family, "tags": idea.tags, "source_type": idea.source_type.value,
        "source_ref": idea.source_ref, "source_date": idea.source_date,
        "source_note": idea.source_note,
        "required_data": [c.value for c in idea.required_data],
        "status": idea.status.value, "signature": idea.signature,
        "authored_strategy_id": idea.authored_strategy_id,
        "duplicate_of_idea_id": idea.duplicate_of_idea_id,
        "override_reason": idea.override_reason,
        "created_at": idea.created_at, "updated_at": idea.updated_at,
    }


def _collision_json(c: Collision) -> dict:
    return {"id": c.idea.id, "title": c.idea.title, "family": c.idea.family,
            "status": c.idea.status.value, "effective_status": c.effective_status.value}


def _parse_required_data(raw: str | None) -> list[DataCapability]:
    if not raw:
        return []
    caps: list[DataCapability] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            caps.append(DataCapability(token))
        except ValueError as exc:
            allowed = ", ".join(c.value for c in DataCapability)
            raise ValueError(
                f"unknown required-data capability {token!r}; allowed: {allowed}") from exc
    return caps


@idea_app.command("add")
@json_errors
def add(
    title: str = typer.Option(..., "--title"),
    hypothesis: str = typer.Option(..., "--hypothesis"),
    family: str = typer.Option(None, "--family", help="thesis family slug"),
    source_type: SourceType = typer.Option(..., "--source-type"),
    source_ref: str = typer.Option(None, "--source-ref", help="url / citation / doi"),
    source_date: str = typer.Option(None, "--source-date", help="ISO date of the source"),
    source_note: str = typer.Option(None, "--source-note"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    required_data: str = typer.Option(
        None, "--required-data", help="comma-separated DataCapability values"),
    allow_duplicate: bool = typer.Option(False, "--allow-duplicate"),
    reason: str = typer.Option(None, "--reason", help="required with --allow-duplicate"),
) -> None:
    """Add a sourced idea. Auto-parks (needs_data) when it needs unsupported data. Fails closed on
    a dedup collision unless --allow-duplicate --reason."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    caps = _parse_required_data(required_data)
    status = classify_status(caps, supported_capabilities())
    with registry_conn() as conn:
        repo = IdeaRepository(conn)
        collisions = repo.find_collisions(title=title, hypothesis=hypothesis, family=family)
        refuted = [c for c in collisions if c.effective_status is IdeaStatus.REFUTED]
        if refuted:
            emit({
                "ok": False,
                "error": "refuted collision: a refuted idea/strategy cannot be re-added "
                         "(the refuted wall is not overridable)",
                "collisions": [_collision_json(c) for c in collisions],
            })
            raise typer.Exit(code=1)
        dup_of: int | None = None
        if collisions:
            if not allow_duplicate:
                emit({
                    "ok": False,
                    "error": "dedup collision; pass --allow-duplicate --reason to override",
                    "collisions": [_collision_json(c) for c in collisions],
                })
                raise typer.Exit(code=1)
            if not reason:
                raise ValueError("--allow-duplicate requires --reason")
            dup_of = collisions[0].idea.id
        idea = repo.add(
            title=title, hypothesis=hypothesis, family=family, tags=tag or [],
            source_type=source_type, source_ref=source_ref, source_date=source_date,
            source_note=source_note, required_data=caps, status=status,
            duplicate_of_idea_id=dup_of, override_reason=reason if dup_of else None)
    emit(ok(_idea_json(idea)))


@idea_app.command("list")
@json_errors
def list_(
    status: str = typer.Option(None, "--status", help="filter by idea status"),
    family: str = typer.Option(None, "--family", help="filter by thesis family"),
) -> None:
    """List ideas (optional filters). Emits a bare JSON array (collection convention)."""
    st = IdeaStatus(status) if status else None
    with registry_conn() as conn:
        ideas = IdeaRepository(conn).list(status=st, family=family)
    emit([_idea_json(i) for i in ideas])


@idea_app.command("show")
@json_errors
def show(idea_id: int = typer.Argument(..., metavar="ID")) -> None:
    """Show one idea by id."""
    with registry_conn() as conn:
        idea = IdeaRepository(conn).get(idea_id)
    emit(ok(_idea_json(idea)))


@idea_app.command("dedup-check")
@json_errors
def dedup_check(
    title: str = typer.Option(..., "--title"),
    hypothesis: str = typer.Option(..., "--hypothesis"),
    family: str = typer.Option(None, "--family"),
) -> None:
    """Preflight a candidate against the pool; no write. Reports collisions (incl. refuted)."""
    with registry_conn() as conn:
        collisions = IdeaRepository(conn).find_collisions(
            title=title, hypothesis=hypothesis, family=family)
    emit(ok({"is_novel": not collisions,
             "collisions": [_collision_json(c) for c in collisions]}))


@idea_app.command("set-status")
@json_errors
def set_status(
    idea_id: int = typer.Argument(..., metavar="ID"),
    to: IdeaStatus = typer.Option(..., "--to"),
    strategy: str = typer.Option(
        None, "--strategy", help="strategy name (required for --to authored)"),
) -> None:
    """Move an idea along its lifecycle (state-machine checked). --to authored links a strategy."""
    with registry_conn() as conn:
        strat_id: int | None = None
        if to is IdeaStatus.AUTHORED:
            if not strategy:
                raise ValueError("--to authored requires --strategy <name>")
            strat_id = SqliteStrategyRepository(conn).get(strategy).id
        idea = IdeaRepository(conn).set_status(idea_id, to=to, authored_strategy_id=strat_id)
    emit(ok(_idea_json(idea)))


@idea_app.command("stats")
@json_errors
def stats(window_days: int = typer.Option(90, "--window-days")) -> None:
    """Funnel-breadth signal: idea counts by status in the trailing window. EXPOSED for the future
    (human, CODEOWNERS) gate change; NOT yet consumed by the promotion gate."""
    with registry_conn() as conn:
        counts = IdeaRepository(conn).windowed_idea_counts(window_days)
    emit(ok({"window_days": window_days, "counts": counts}))
