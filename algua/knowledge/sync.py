from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block
from algua.knowledge.metrics import latest_run_metrics
from algua.registry import store
from algua.registry.db import connect, migrate


def strategy_doc_path(settings: Settings, name: str) -> Path:
    return settings.knowledge_dir / f"{name}.md"


def family_doc_path(settings: Settings, name: str) -> Path:
    return settings.knowledge_dir / "families" / f"{name}.md"


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


def sync_strategy_doc(settings: Settings, conn, name: str) -> bool:
    """Rewrite the synced parts of one strategy doc. Returns False if the doc is absent."""
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return False
    fm, body = parse_doc(path.read_text())
    try:
        rec = store.get_strategy(conn, name)
        fm["stage"] = rec.stage.value
    except store.StrategyNotFound:
        pass  # doc scaffolded before registry add — leave authored stage untouched
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
        if fm.get("family") == f"[[{name}]]":
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


def sync_all(settings: Settings, conn) -> dict[str, list[str]]:
    """Sync every registry strategy that has a doc, every family doc, then the indexes."""
    synced: list[str] = []
    for rec in store.list_strategies(conn):
        if sync_strategy_doc(settings, conn, rec.name):
            synced.append(rec.name)
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


def kb_check(settings: Settings) -> tuple[bool, str]:
    """Flag registry strategies with no doc or a stale synced stage. For `doctor`."""
    with closing(connect(settings.db_path)) as conn:
        migrate(conn)
        records = store.list_strategies(conn)
    issues: list[str] = []
    for rec in records:
        path = strategy_doc_path(settings, rec.name)
        if not path.exists():
            issues.append(f"{rec.name}: no doc")
            continue
        fm, _ = parse_doc(path.read_text())
        if fm.get("stage") != rec.stage.value:
            issues.append(f"{rec.name}: doc stage {fm.get('stage')} != registry {rec.stage.value}")
    if issues:
        return False, "; ".join(issues)
    return True, f"{len(records)} strategies in sync"
