# algua/cli/idea_json.py
"""Shared JSON-projection helpers for the idea-pool CLI surface (spec 2026-09-08 §7).

Carved out up front (not waited on until the size ratchet forced it) so `idea_cmd.py` (agent-
facing: add/list/show/dedup-check/set-status/stats) and `idea_ops_cmd.py` (driver-facing:
claim/record-outcome/link/depth/refuted/import/reclassify/scorecard) share one JSON shape for an
`Idea` instead of drifting. Deliberately NOT listed in the cli-independence import-linter contract
(`pyproject.toml`) -- it is shared infra like `cli._common`/`cli.app`/`cli.errors`, not a command
module in its own right.
"""
from __future__ import annotations

from algua.contracts.idea import DataCapability
from algua.registry.ideas import Collision, IdeaRepository, InspirationLink, Obscurity


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


def _parse_inspirations(raw: list[str] | None) -> list[InspirationLink]:
    out = []
    for token in raw or []:
        parts = token.split("|")
        if len(parts) != 3:
            raise ValueError(f"--inspiration must be id|venue|obscurity, got {token!r}")
        out.append(InspirationLink(parts[0].strip(), parts[1].strip(), Obscurity(parts[2].strip())))
    return out


def idea_json(idea, repo: IdeaRepository) -> dict:
    """Project an `Idea` (+ its inspiration links, read via `repo`) to its CLI JSON shape."""
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
        "category": idea.category,
        "market": idea.market.value if idea.market else None,
        "horizon": idea.horizon.value if idea.horizon else None,
        "falsification": idea.falsification, "parked_reason": idea.parked_reason,
        "claimed_by": idea.claimed_by, "claim_token": idea.claim_token,
        "claimed_at": idea.claimed_at,
        "inspirations": [
            {"inspiration_id": link.inspiration_id, "venue": link.venue,
             "obscurity": link.obscurity.value}
            for link in repo.inspirations_of(idea.id)
        ],
    }


def collision_json(c: Collision) -> dict:
    return {"id": c.idea.id, "title": c.idea.title, "family": c.idea.family,
            "status": c.idea.status.value, "effective_status": c.effective_status.value}
