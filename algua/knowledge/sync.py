from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from algua.config.settings import Settings
from algua.contracts.lifecycle import Stage
from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block
from algua.knowledge.metrics import latest_run_metrics

# Canonical lifecycle order for grouping rosters/axis pages by stage. Derived from the Stage enum
# (definition order == lifecycle order) so adding a stage needs no edit here. Unknown/legacy stage
# strings sort AFTER all known stages (and alphabetically among themselves) and are never dropped —
# a corrupted stage stays visible to a human instead of vanishing from the hub.
_STAGE_ORDER = {s.value: i for i, s in enumerate(Stage)}


def _stage_sort_key(stage: str) -> tuple[int, str]:
    return (_STAGE_ORDER.get(stage, len(_STAGE_ORDER)), stage)


def _created_month(value: object) -> str:
    """Normalize a frontmatter ``created`` value to a ``YYYY-MM`` bucket; ``undated`` otherwise.

    YAML may load the date as a ``date``/``datetime`` or as an ISO string; anything missing or
    malformed buckets to ``undated`` so the date axis never raises.
    """
    if isinstance(value, datetime | date):
        return f"{value.year:04d}-{value.month:02d}"
    if isinstance(value, str):
        for parse in (date.fromisoformat, datetime.fromisoformat):
            try:
                parsed = parse(value)
            except ValueError:
                continue
            return f"{parsed.year:04d}-{parsed.month:02d}"
    return "undated"


def _safe_path(base: Path, *parts: str) -> Path:
    """Join `parts` under `base`, refusing any name that escapes the vault root.

    Doc names flow in from CLI arguments and registry rows, so a `../escape` must never
    resolve a write outside `knowledge_dir`.
    """
    candidate = base.joinpath(*parts)
    base_resolved = base.resolve()
    resolved = candidate.resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(f"unsafe knowledge-base path: {'/'.join(parts)!r}")
    return candidate


def strategies_dir(settings: Settings) -> Path:
    """The strategy domain of the vault: `<knowledge_dir>/strategies/`."""
    return settings.knowledge_dir / "strategies"


def strategy_doc_path(settings: Settings, name: str) -> Path:
    return _safe_path(strategies_dir(settings), f"{name}.md")


def family_doc_path(settings: Settings, name: str) -> Path:
    # Contain to families/ itself, so a stray name can't even land elsewhere in the vault.
    return _safe_path(strategies_dir(settings) / "families", f"{name}.md")


def _unwikilink(value: object) -> str | None:
    """`"[[momentum]]"` -> `"momentum"`; passthrough for a bare slug; None otherwise."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.startswith("[[") and stripped.endswith("]]"):
        stripped = stripped[2:-2]
    return stripped or None


def strategy_family(settings: Settings, name: str) -> str | None:
    """The family slug a strategy doc declares, or None if the doc/field is absent."""
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return None
    fm, _ = parse_doc(path.read_text())
    return _unwikilink(fm.get("family"))


def render_results_block(metrics: dict[str, Any] | None) -> str:
    """Markdown content (no markers) for the synced RESULTS block."""
    if not metrics:
        return "_No tracked runs yet._"
    head = (
        f"Latest run: `{metrics['kind']}` · snapshot `{metrics.get('snapshot_id')}` · "
        f"seed `{metrics.get('seed')}` · run `{metrics['run_id'][:8]}`"
    )
    lines = [head, ""]
    if metrics["metrics"]:
        lines += ["| metric | value |", "|---|---|"]
        for key in sorted(metrics["metrics"]):
            lines.append(f"| {key} | {metrics['metrics'][key]:.4g} |")
    return "\n".join(lines)


def render_members_block(members: list[tuple[str, str]]) -> str:
    """Markdown content (no markers) for a family's synced MEMBERS block.

    ``members`` is a list of ``(strategy_stem, stage)`` pairs. Renders a true MOC: a total-count
    summary line followed by per-stage ``### {stage} ({n})`` sections of ``[[stem]]`` wikilinks
    (stems sorted), stages in canonical lifecycle order (unknown stages last, never dropped). Links
    target the canonical filename ``stem``, so a drifted frontmatter ``name`` can't break them.
    """
    if not members:
        return "_No members yet._"
    by_stage: dict[str, list[str]] = {}
    for stem, stage in members:
        by_stage.setdefault(stage, []).append(stem)
    lines = [f"**{len(members)} members**", ""]
    for stage in sorted(by_stage, key=_stage_sort_key):
        stems = sorted(by_stage[stage])
        lines.append(f"### {stage} ({len(stems)})")
        lines += [f"- [[{stem}]]" for stem in stems]
        lines.append("")
    return "\n".join(lines).rstrip()


def _apply_owned_metadata(fm: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Write the registry-owned frontmatter keys from a registry metadata dict, wrapping
    ``family``/``derived_from`` as Obsidian wikilinks. NULL/None values clear the key.

    These keys are registry-owned: edit them via ``registry set``, not by hand in the kb doc.
    Every sync overwrites them, so hand edits to these keys are lost on the next sync.
    """
    for key in ("family", "derived_from"):
        val = metadata.get(key)
        if val:
            fm[key] = f"[[{val}]]"
        else:
            fm.pop(key, None)
    for key in ("tags", "author", "hypothesis_status", "description"):
        val = metadata.get(key)
        if val is not None:
            fm[key] = val
        else:
            fm.pop(key, None)


def sync_strategy_doc(
    settings: Settings,
    name: str,
    *,
    stage: str | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Rewrite the synced parts of one strategy doc. Returns False if the doc is absent.

    ``stage`` and ``metadata`` are read by the caller at the CLI seam so this layer never touches
    the registry. ``metadata`` carries the registry-owned organizational fields; only those keys
    (plus ``stage``/``mlflow_run``) are overwritten — any other frontmatter key is preserved.
    """
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return False
    fm, body = parse_doc(path.read_text())
    if stage is not None:
        fm["stage"] = stage
    if metadata is not None:
        _apply_owned_metadata(fm, metadata)
    metrics = latest_run_metrics(name, tracking_uri=settings.mlflow_tracking_uri)
    if metrics:
        fm["mlflow_run"] = metrics["run_id"][:8]
    body = replace_block(body, "RESULTS", render_results_block(metrics))
    path.write_text(render_doc(fm, body))
    return True


def sync_family_doc(settings: Settings, name: str) -> bool:
    """Rewrite the synced MEMBERS roster of one family doc. False if the doc is absent.

    The roster is a true MOC — a stage-grouped list of ``[[member]]`` wikilinks, not a count string.
    """
    path = family_doc_path(settings, name)
    if not path.exists():
        return False
    members: list[tuple[str, str]] = []
    for doc in strategies_dir(settings).glob("*.md"):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if _unwikilink(fm.get("family")) == name:
            members.append((doc.stem, str(fm.get("stage", "?"))))
    fm, body = parse_doc(path.read_text())
    body = replace_block(body, "MEMBERS", render_members_block(members))
    path.write_text(render_doc(fm, body))
    return True


def _strategy_entries(base: Path) -> list[dict[str, Any]]:
    """Frontmatter of every strategy doc (one non-recursive ``*.md`` scan, DRY across axis pages).

    Skips ``_``-prefixed roll-ups and ``type: family`` docs; family docs live in ``families/`` so
    the non-recursive glob already excludes them. Each entry carries the canonical link ``stem``.
    """
    entries: list[dict[str, Any]] = []
    for doc in sorted(base.glob("*.md")):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if fm.get("type") == "family":
            continue
        entries.append(
            {
                "stem": doc.stem,
                "family": _unwikilink(fm.get("family")),
                "stage": str(fm.get("stage", "?")),
                "hypothesis_status": str(fm.get("hypothesis_status", "?")),
                "created": fm.get("created"),
            }
        )
    return entries


def _family_link(family: str | None) -> str:
    return f"[[{family}]]" if family else "—"


def _render_by_stage(entries: list[dict[str, Any]]) -> str:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_stage.setdefault(e["stage"], []).append(e)
    lines = ["# Strategies by stage", ""]
    for stage in sorted(by_stage, key=_stage_sort_key):
        rows = sorted(by_stage[stage], key=lambda e: e["stem"])
        lines.append(f"## {stage} ({len(rows)})")
        lines += [
            f"- [[{e['stem']}]] — {e['hypothesis_status']} · {_family_link(e['family'])}"
            for e in rows
        ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_by_date(entries: list[dict[str, Any]]) -> str:
    by_month: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_month.setdefault(_created_month(e["created"]), []).append(e)
    lines = ["# Strategies by created date", ""]
    # Newest month first; the ``undated`` bucket sorts last (it is not a real month).
    ordered = sorted((m for m in by_month if m != "undated"), reverse=True)
    if "undated" in by_month:
        ordered.append("undated")
    for month in ordered:
        rows = sorted(by_month[month], key=lambda e: e["stem"])
        lines.append(f"## {month} ({len(rows)})")
        lines += [
            f"- [[{e['stem']}]] — {e['stage']} · {_family_link(e['family'])}" for e in rows
        ]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_indexes(settings: Settings) -> None:
    """(Re)generate the strategies roll-ups: a bounded ``_index.md`` router over per-axis pages
    (``_by-stage.md`` status, ``_by-date.md`` date, ``_families.md`` thesis)."""
    base = strategies_dir(settings)
    base.mkdir(parents=True, exist_ok=True)
    entries = _strategy_entries(base)

    fam_dir = base / "families"
    fam_lines: list[str] = []
    n_families = 0
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            n_families += 1
            fam_lines.append(f"- [[{doc.stem}]] — {fm.get('status', '?')}")
    (base / "_families.md").write_text("# Thesis families\n\n" + "\n".join(fam_lines) + "\n")

    (base / "_by-stage.md").write_text(_render_by_stage(entries))
    (base / "_by-date.md").write_text(_render_by_date(entries))

    router = (
        "# Strategies\n\n"
        f"{len(entries)} strategies across {n_families} families. Navigate by axis:\n\n"
        "- [[_by-stage]] — by lifecycle stage (status)\n"
        "- [[_by-date]] — by created month (date)\n"
        "- [[_families]] — by thesis family\n"
    )
    (base / "_index.md").write_text(router)


def sync_all(
    settings: Settings,
    stages: dict[str, str],
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Sync each registered strategy's doc (stage + optional metadata), every family doc, then
    indexes. ``metadata`` maps strategy name -> its registry metadata dict.

    Strategy docs are synced before family docs so member rosters count freshly-synced stages.
    """
    metadata = metadata or {}
    synced: list[str] = []
    for name, stage in stages.items():
        if sync_strategy_doc(settings, name, stage=stage, metadata=metadata.get(name)):
            synced.append(name)
    families: list[str] = []
    fam_dir = strategies_dir(settings) / "families"
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            fam_name = str(fm.get("name", doc.stem))
            if sync_family_doc(settings, fam_name):
                families.append(fam_name)
    generate_indexes(settings)
    return {"strategies": synced, "families": families}


def kb_check(settings: Settings, stages: dict[str, str]) -> tuple[bool, str]:
    """Flag registered strategies with no doc or a stale synced stage. For `doctor`.

    `stages` is the registry name->stage mapping, read by the caller at the CLI seam.
    """
    issues: list[str] = []
    for name, stage in stages.items():
        path = strategy_doc_path(settings, name)
        if not path.exists():
            issues.append(f"{name}: no doc")
            continue
        fm, _ = parse_doc(path.read_text())
        if fm.get("stage") != stage:
            issues.append(f"{name}: doc stage {fm.get('stage')} != registry {stage}")
    if issues:
        return False, "; ".join(issues)
    return True, f"{len(stages)} strategies in sync"
