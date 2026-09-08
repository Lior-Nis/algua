from __future__ import annotations

import re

import typer

from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.cli.idea_json import _parse_inspirations, _parse_required_data, collision_json, idea_json
from algua.contracts.idea import Horizon, IdeaStatus, Market, SourceType
from algua.data.capabilities import (
    supported_capabilities,
    supported_horizons,
    supported_markets,
)
from algua.registry.db import registry_conn
from algua.registry.ideas import IdeaRepository
from algua.registry.store import SqliteStrategyRepository
from algua.research.ideas import classify_idea

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

idea_app = typer.Typer(
    help="Idea pool: source, dedup, and park research hypotheses", no_args_is_help=True)


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
    category: str = typer.Option(None, "--category", help="controlled-vocabulary idea category"),
    market: Market = typer.Option(None, "--market"),
    horizon: Horizon = typer.Option(None, "--horizon"),
    falsification: str = typer.Option(None, "--falsification", help="what would refute it"),
    inspiration: list[str] = typer.Option(
        None, "--inspiration", help="id|venue|obscurity (repeatable)"),
) -> None:
    """Add a sourced idea. Auto-parks (needs_data) when it needs unsupported data/market/horizon.
    Fails closed on a dedup collision unless --allow-duplicate --reason. `--source-type inspiration`
    requires --category, --market, --horizon and --falsification (spec §5/§6)."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    if source_type is SourceType.INSPIRATION and (
        category is None or market is None or horizon is None or falsification is None
    ):
        raise ValueError(
            "--source-type inspiration requires --category, --market, --horizon and "
            "--falsification")
    if category is not None and not _CATEGORY_RE.match(category):
        raise ValueError(
            f"invalid category {category!r}: must be a lowercase slug (a-z, 0-9, underscore)")
    caps = _parse_required_data(required_data)
    links = _parse_inspirations(inspiration)
    status, parked = classify_idea(
        caps, supported_capabilities(), market=market, supported_markets=supported_markets(),
        horizon=horizon, supported_horizons=supported_horizons())
    with registry_conn() as conn:
        repo = IdeaRepository(conn)
        collisions = repo.find_collisions(title=title, hypothesis=hypothesis, family=family)
        refuted = [c for c in collisions if c.effective_status is IdeaStatus.REFUTED]
        if refuted:
            emit({
                "ok": False,
                "error": "refuted collision: a refuted idea/strategy cannot be re-added "
                         "(the refuted wall is not overridable)",
                "collisions": [collision_json(c) for c in collisions],
            })
            raise typer.Exit(code=1)
        dup_of: int | None = None
        if collisions:
            if not allow_duplicate:
                emit({
                    "ok": False,
                    "error": "dedup collision; pass --allow-duplicate --reason to override",
                    "collisions": [collision_json(c) for c in collisions],
                })
                raise typer.Exit(code=1)
            if not reason:
                raise ValueError("--allow-duplicate requires --reason")
            dup_of = collisions[0].idea.id
        idea = repo.add(
            title=title, hypothesis=hypothesis, family=family, tags=tag or [],
            source_type=source_type, source_ref=source_ref, source_date=source_date,
            source_note=source_note, required_data=caps, status=status,
            duplicate_of_idea_id=dup_of, override_reason=reason if dup_of else None,
            category=category, market=market, horizon=horizon, falsification=falsification,
            parked_reason=parked, inspirations=links, created_by_run="cli")
        payload = idea_json(idea, repo)
    emit(ok(payload))


@idea_app.command("list")
@json_errors
def list_(
    status: str = typer.Option(None, "--status", help="filter by idea status"),
    family: str = typer.Option(None, "--family", help="filter by thesis family"),
    limit: int = typer.Option(None, "--limit", min=1, help="most-recent-first cap"),
) -> None:
    """List ideas (optional filters). Emits a bare JSON array (collection convention)."""
    st = IdeaStatus(status) if status else None
    with registry_conn() as conn:
        repo = IdeaRepository(conn)
        ideas = repo.list(status=st, family=family, limit=limit)
        payload = [idea_json(i, repo) for i in ideas]
    emit(payload)


@idea_app.command("show")
@json_errors
def show(idea_id: int = typer.Argument(..., metavar="ID")) -> None:
    """Show one idea by id."""
    with registry_conn() as conn:
        repo = IdeaRepository(conn)
        idea = repo.get(idea_id)
        payload = idea_json(idea, repo)
    emit(ok(payload))


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
             "collisions": [collision_json(c) for c in collisions]}))


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
        repo = IdeaRepository(conn)
        idea = repo.set_status(idea_id, to=to, authored_strategy_id=strat_id)
        payload = idea_json(idea, repo)
    emit(ok(payload))


@idea_app.command("stats")
@json_errors
def stats(window_days: int = typer.Option(90, "--window-days")) -> None:
    """Funnel-breadth signal: idea counts by status in the trailing window. EXPOSED for the future
    (human, CODEOWNERS) gate change; NOT yet consumed by the promotion gate."""
    with registry_conn() as conn:
        counts = IdeaRepository(conn).windowed_idea_counts(window_days)
    emit(ok({"window_days": window_days, "counts": counts}))
