"""`research inspirations` — the vault-side CLI over `algua.knowledge.inspirations` (spec
2026-09-08 §4/§5). Thin typer wrappers: all validation/IO lives in the knowledge module; this
module only parses flags, resolves the categories file, and shapes the JSON envelope.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import typer

from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.config.settings import Settings, get_settings
from algua.knowledge.inspirations import (
    SOURCE_KINDS,
    SourcesRegistry,
    accept_new_notes,
    list_notes,
    mark_exhausted,
    mark_used,
)
from algua.primitives.timeparse import now_iso

inspirations_app = typer.Typer(
    help="Inspirations vault domain: accept forage notes, browse them, mark them leapt/exhausted, "
         "and curate the sources registry.",
    no_args_is_help=True,
)

_DEFAULT_CATEGORIES_REL = Path(".codex/categories.txt")
_KEY_RE = re.compile(r"^[a-z0-9_]+/[A-Za-z0-9_.-]+$")
_OBSCURITY_RANK = {"rare": 0, "niche": 1, "common": 2, "canon": 3}
_MIN_VENUE_N = 5


def _repo_root() -> Path:
    """Walk up from this file to the directory containing `pyproject.toml` (the repo root).

    Never `Path.cwd()` — this module may run from any working directory, but the categories file
    lives at a fixed repo-relative location.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not locate pyproject.toml walking up from {here}")


def _resolve_categories_file(categories_file: Path | None) -> Path:
    path = categories_file if categories_file is not None else _DEFAULT_CATEGORIES_REL
    return path if path.is_absolute() else _repo_root() / path


def _load_categories(path: Path) -> set[str]:
    """Slugs = the first whitespace-separated token of each non-comment, non-blank line."""
    if not path.exists():
        raise FileNotFoundError(f"categories file not found: {path}")
    cats: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cats.add(stripped.split()[0])
    return cats


def _sources_path(settings: Settings) -> Path:
    return settings.knowledge_dir / "inspirations" / "_sources.yaml"


@inspirations_app.command("accept")
@json_errors
def accept(
    from_dir: Path = typer.Option(
        ..., "--from", exists=True, file_okay=False, help="staged notes directory"),
    run: str = typer.Option(..., "--run", help="run stamp recorded in the seen file"),
    max_notes: int = typer.Option(50, "--max", min=1, help="max notes accepted this run"),
    categories_file: Path = typer.Option(
        None, "--categories-file", help="default: .codex/categories.txt at the repo root"),
    seen_file: Path = typer.Option(
        None, "--seen-file", help="default: <data_dir>/inspirations-seen.jsonl"),
) -> None:
    """Trusted acceptance of what a forage agent staged: validate, copy into the vault, and
    record the source URL as seen. Never touches a note the forage agent didn't stage this run."""
    settings = get_settings()
    categories = _load_categories(_resolve_categories_file(categories_file))
    seen_path = seen_file if seen_file is not None else (
        settings.data_dir / "inspirations-seen.jsonl")
    result = accept_new_notes(
        staged_dir=from_dir, settings=settings, seen_path=seen_path, categories=categories,
        run_stamp=run, max_notes=max_notes)
    emit(ok(result))


@inspirations_app.command("list")
@json_errors
def list_cmd(
    status: str = typer.Option(None, "--status", help="filter: fresh | used | exhausted"),
    limit: int = typer.Option(None, "--limit", min=1, help="cap after any sort"),
    rare_first: bool = typer.Option(
        False, "--rare-first", help="sort rare>niche>common>canon, then newest id first"),
) -> None:
    """Bare JSON array of frontmatter dicts (collection convention)."""
    settings = get_settings()
    if rare_first:
        notes = list_notes(settings, status=status, limit=None)
        notes.sort(key=lambda n: str(n.get("id") or ""), reverse=True)
        notes.sort(key=lambda n: _OBSCURITY_RANK.get(
            str(n.get("obscurity")), len(_OBSCURITY_RANK)))
        if limit is not None:
            notes = notes[:limit]
    else:
        notes = list_notes(settings, status=status, limit=limit)
    emit(notes)


@inspirations_app.command("mark-used")
@json_errors
def mark_used_cmd(
    inspiration_id: str = typer.Argument(..., metavar="ID"),
    idea: int = typer.Option(..., "--idea", help="idea id this note leapt into"),
) -> None:
    """Trusted leap-driver edit: append the idea id to `leaps` and flip fresh -> used."""
    fm = mark_used(get_settings(), inspiration_id, idea_id=idea)
    emit(ok(fm))


@inspirations_app.command("mark-exhausted")
@json_errors
def mark_exhausted_cmd(inspiration_id: str = typer.Argument(..., metavar="ID")) -> None:
    """Trusted leap-driver edit: flip status -> exhausted (no more leaps to try)."""
    fm = mark_exhausted(get_settings(), inspiration_id)
    emit(ok(fm))


@inspirations_app.command("propose")
@json_errors
def propose(
    key: str = typer.Option(..., "--key", help="venue key, e.g. blog/example"),
    kind: str = typer.Option(..., "--kind", help="one of the six source kinds"),
    url: str = typer.Option(..., "--url", help="https:// venue URL"),
    categories: str = typer.Option(..., "--categories", help="comma-separated category slugs"),
    categories_file: Path = typer.Option(
        None, "--categories-file", help="default: .codex/categories.txt at the repo root"),
) -> None:
    """Propose a new venue into the sources registry (a no-op if the key already exists)."""
    if not _KEY_RE.match(key):
        raise ValueError(f"invalid key {key!r}: must match {_KEY_RE.pattern!r}")
    if kind not in SOURCE_KINDS:
        raise ValueError(f"invalid kind {kind!r}: must be one of {sorted(SOURCE_KINDS)}")
    if not url.startswith("https://"):
        raise ValueError(f"invalid url {url!r}: must start with https://")
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    allowed = _load_categories(_resolve_categories_file(categories_file))
    unknown = sorted(set(cats) - allowed)
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")

    settings = get_settings()
    registry = SourcesRegistry(_sources_path(settings))
    registry.propose({"key": key, "kind": kind, "url": url, "categories": cats})
    emit(ok({"key": key, "kind": kind, "url": url, "categories": cats}))


@inspirations_app.command("write-yield")
@json_errors
def write_yield_cmd(
    from_scorecard: str = typer.Option(
        ..., "--from-scorecard", help="path to a `research idea scorecard` JSON payload, or -"),
) -> None:
    """Feed `research idea scorecard`'s per-venue yields back into the sources registry: every
    `by_venue` key with `n >= 5` observations gets a fresh yield row."""
    raw = sys.stdin.read() if from_scorecard == "-" else Path(from_scorecard).read_text(
        encoding="utf-8")
    data = json.loads(raw)
    days = data.get("days")
    computed_at = now_iso()

    settings = get_settings()
    registry = SourcesRegistry(_sources_path(settings))
    updated = []
    for venue_key, stats in (data.get("by_venue") or {}).items():
        if (stats or {}).get("n", 0) < _MIN_VENUE_N:
            continue
        registry.write_yield(venue_key, {
            "window_days": days,
            "n": stats["n"],
            "integrity_yield": stats.get("integrity_yield"),
            "walkforward_yield": stats.get("walkforward_yield"),
            "survival_yield": stats.get("survival_yield"),
            "computed_at": computed_at,
        })
        updated.append(venue_key)
    emit(ok({"updated": updated}))
