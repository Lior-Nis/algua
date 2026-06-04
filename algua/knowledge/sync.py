from __future__ import annotations

from pathlib import Path
from typing import Any

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block
from algua.knowledge.metrics import latest_run_metrics


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


def strategy_doc_path(settings: Settings, name: str) -> Path:
    return _safe_path(settings.knowledge_dir, f"{name}.md")


def family_doc_path(settings: Settings, name: str) -> Path:
    return _safe_path(settings.knowledge_dir, "families", f"{name}.md")


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


def sync_strategy_doc(settings: Settings, name: str, *, stage: str | None) -> bool:
    """Rewrite the synced parts of one strategy doc. Returns False if the doc is absent.

    `stage` is the registry lifecycle stage (None if the strategy isn't registered yet);
    the caller reads it at the CLI seam so this layer never touches the registry.
    """
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return False
    fm, body = parse_doc(path.read_text())
    if stage is not None:
        fm["stage"] = stage
    metrics = latest_run_metrics(name, tracking_uri=settings.mlflow_tracking_uri)
    if metrics:
        fm["mlflow_run"] = metrics["run_id"][:8]
    body = replace_block(body, "RESULTS", render_results_block(metrics))
    path.write_text(render_doc(fm, body))
    return True


def sync_family_doc(settings: Settings, name: str) -> bool:
    """Rewrite the synced MEMBERS roster of one family doc. False if the doc is absent."""
    path = family_doc_path(settings, name)
    if not path.exists():
        return False
    counts: dict[str, int] = {}
    for doc in settings.knowledge_dir.glob("*.md"):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if _unwikilink(fm.get("family")) == name:
            stage = str(fm.get("stage", "?"))
            counts[stage] = counts.get(stage, 0) + 1
    total = sum(counts.values())
    detail = f"{total} members"
    if counts:
        detail += ": " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    fm, body = parse_doc(path.read_text())
    body = replace_block(body, "MEMBERS", detail)
    path.write_text(render_doc(fm, body))
    return True


def generate_indexes(settings: Settings) -> None:
    """(Re)generate _index.md (strategies) and _families.md from vault frontmatter."""
    vault = settings.knowledge_dir
    vault.mkdir(parents=True, exist_ok=True)
    strat_lines: list[str] = []
    for doc in sorted(vault.glob("*.md")):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if fm.get("type") == "family":
            continue
        name = fm.get("name", doc.stem)
        strat_lines.append(
            f"- [[{name}]] — {fm.get('family', '—')} · "
            f"{fm.get('stage', '?')} · {fm.get('hypothesis_status', '?')}"
        )
    (vault / "_index.md").write_text("# Strategies\n\n" + "\n".join(strat_lines) + "\n")

    fam_dir = vault / "families"
    fam_lines: list[str] = []
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            fam_lines.append(f"- [[{fm.get('name', doc.stem)}]] — {fm.get('status', '?')}")
    (vault / "_families.md").write_text("# Thesis families\n\n" + "\n".join(fam_lines) + "\n")


def sync_all(settings: Settings, stages: dict[str, str]) -> dict[str, list[str]]:
    """Sync each registered strategy's doc (stage from `stages`), every family doc, then indexes.

    Strategy docs are synced before family docs so member rosters count freshly-synced stages.
    """
    synced: list[str] = []
    for name, stage in stages.items():
        if sync_strategy_doc(settings, name, stage=stage):
            synced.append(name)
    families: list[str] = []
    fam_dir = settings.knowledge_dir / "families"
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
