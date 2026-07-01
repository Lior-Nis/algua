"""Vault-resident experience notes for the negative-result log (#332).

The advisory ledger (``algua.registry.negative_results``) is the queryable source of truth; this
module writes a human-browsable, graph-linked Obsidian note per captured negative result under
``<knowledge_dir>/experience/`` so a person curating the vault sees refuted hypotheses next to the
strategy docs. It is a BEST-EFFORT secondary surface — the caller wraps it so a note failure never
breaks the thing being logged, and never masquerades as a ledger loss.

No maintained ``_index.md``: Obsidian's own graph/search plus ``research log list`` are the
discovery paths, which sidesteps the index-rewrite race two concurrent promotes would create.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from algua.config.settings import Settings
from algua.knowledge.frontmatter import render_doc
from algua.knowledge.sync import _safe_path

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def experience_dir(settings: Settings) -> Path:
    """The experience-log domain of the vault: ``<knowledge_dir>/experience/``."""
    return settings.knowledge_dir / "experience"


def _slug(value: str | None) -> str:
    """Filesystem-safe slug; ``unattributed`` when a strategy name is absent."""
    if not value:
        return "unattributed"
    s = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return s or "unattributed"


def experience_note_path(settings: Settings, *, record_id: int, strategy: str | None,
                         created_at: str) -> Path:
    """Unique, containment-checked note path: ``{date}-{strategy-slug}-{id}.md``.

    The ledger id guarantees uniqueness (no collision under parallel promotes); the date/slug make
    it human-scannable. ``_safe_path`` refuses any name that would escape the vault root.
    """
    date = created_at[:10] if created_at else ""
    return _safe_path(experience_dir(settings), f"{date}-{_slug(strategy)}-{record_id}.md")


def render_experience_note(record: dict[str, Any], *, record_id: int) -> str:
    """Render one negative-result note (frontmatter + body). Pure."""
    strategy = record.get("strategy_name")
    fm: dict[str, Any] = {
        "type": "negative-result",
        "id": record_id,
        "kind": record.get("kind"),
        "verdict": record.get("verdict"),
        "actor": record.get("actor"),
        "source": record.get("source"),
        "created": record.get("created_at"),
    }
    if strategy:
        fm["strategy"] = f"[[{strategy}]]"
    if record.get("gate_evaluation_id") is not None:
        fm["gate_evaluation_id"] = record["gate_evaluation_id"]
    if record.get("tags"):
        fm["tags"] = record["tags"]

    parts = [f"## Verdict\n\n{record.get('verdict', '?')} — {record.get('reason', '')}\n"]
    if record.get("hypothesis"):
        parts.append(f"## Hypothesis\n\n{record['hypothesis']}\n")
    params = record.get("params")
    if params:
        parts.append(
            "## Evidence\n\n```json\n"
            + json.dumps(params, indent=2, sort_keys=True, default=str)
            + "\n```\n"
        )
    parts.append(
        "> Advisory negative-result capture (#332). Queryable via `algua research log list`.\n"
    )
    return render_doc(fm, "\n".join(parts))


def write_experience_note(settings: Settings, record: dict[str, Any], *, record_id: int) -> Path:
    """Write the note atomically (exclusive create — never clobbers). Returns the path.

    ``record`` is the recorder's field dict augmented with ``created_at``/``gate_evaluation_id``.
    """
    path = experience_note_path(
        settings, record_id=record_id, strategy=record.get("strategy_name"),
        created_at=record.get("created_at", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: the ledger id makes the name unique, so an existing file signals a real
    # collision we must not silently overwrite.
    with path.open("x", encoding="utf-8") as fh:
        fh.write(render_experience_note(record, record_id=record_id))
    return path
