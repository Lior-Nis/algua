# algua/registry/idea_import.py
"""Trusted-driver writes that move agent output into the authoritative pool (spec §6/§7).

`import_ideas`: scratch rows (id > seeded max) -> authority under a fresh dedup + eligibility
check, one BEGIN IMMEDIATE per row. `import_critic_rejections`: DB-only negative-result rows.
`refuted_with_reasons`: what leap reads. `reclassify`: re-open parked rows after a capability flip.

EVERY field crossing this seam is MODEL OUTPUT written into a scratch DB the agent owns, so this
module is where it stops being trusted: the category must be in the human's `.codex/categories.txt`
vocabulary, each inspiration link's `venue`/`obscurity` is re-derived from the vault note itself
(the agent's own labels are discarded), and every free-text field is length-capped.
"""
from __future__ import annotations

import sqlite3

from algua.config.settings import Settings
from algua.contracts.idea import Idea, IdeaStatus, Obscurity
from algua.data.capabilities import (
    supported_capabilities,
    supported_horizons,
    supported_markets,
)
from algua.knowledge.inspirations import load_note
from algua.registry.ideas import IdeaRepository, InspirationLink
from algua.registry.negative_results import record_negative_result
from algua.research.ideas import classify_idea

#: Per-field character budgets for imported free text. A field over its limit is TRUNCATED with a
#: `…` marker (an over-long title is still a usable idea); a field over `ABSURD_FACTOR` x its limit
#: is not a long idea but a runaway generation, and skips the whole row as `oversized`.
MAX_TITLE = 200
MAX_HYPOTHESIS = 4000
MAX_FALSIFICATION = 1000
MAX_SOURCE_NOTE = 2000
MAX_TAG = 60
MAX_TAGS = 20
ABSURD_FACTOR = 4


def _eligibility(idea: Idea) -> tuple[IdeaStatus, str | None]:
    return classify_idea(idea.required_data, supported_capabilities(), market=idea.market,
                         supported_markets=supported_markets(), horizon=idea.horizon,
                         supported_horizons=supported_horizons())


def _truncate(value: str, limit: int) -> str:
    """Cap `value` at `limit` CHARACTERS INCLUDING the `…` marker, so the stored length is exactly
    the budget — never the budget plus a marker."""
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _truncate_opt(value: str | None, limit: int) -> str | None:
    return None if value is None else _truncate(value, limit)


def _oversized(idea: Idea) -> bool:
    fields = ((idea.title, MAX_TITLE), (idea.hypothesis, MAX_HYPOTHESIS),
              (idea.falsification, MAX_FALSIFICATION), (idea.source_note, MAX_SOURCE_NOTE))
    if any(v is not None and len(v) > limit * ABSURD_FACTOR for v, limit in fields):
        return True
    return any(len(t) > MAX_TAG * ABSURD_FACTOR for t in idea.tags)


def _resolve_links(settings: Settings, links: list[InspirationLink]
                   ) -> tuple[list[InspirationLink], list[str]]:
    """Re-derive each link's venue/obscurity from its vault note; drop links with no readable note.

    The agent supplied `id|venue|obscurity` itself, and those two fields feed the per-venue yield
    scorecard that steers future foraging — trusting them would let a mislabeled `rare` inflate a
    venue's standing. The note (written by the forage driver, validated on acceptance) is the
    authority; a link whose note is gone is reported, never silently kept with agent metadata.
    """
    kept: list[InspirationLink] = []
    dropped: list[str] = []
    for link in links:
        fm = load_note(settings, link.inspiration_id)
        venue = str((fm or {}).get("venue") or "")
        raw_obscurity = str((fm or {}).get("obscurity") or "")
        if not fm or not venue or raw_obscurity not in {o.value for o in Obscurity}:
            dropped.append(link.inspiration_id)
            continue
        kept.append(InspirationLink(link.inspiration_id, venue, Obscurity(raw_obscurity)))
    return kept, dropped


def import_ideas(auth: sqlite3.Connection, scratch: sqlite3.Connection, *, settings: Settings,
                 categories: set[str], run_stamp: str, max_new: int, ceiling: int,
                 seeded_max_id: int) -> dict:
    if auth.in_transaction:
        raise RuntimeError(
            "import_ideas must run at top level, not inside an open transaction")
    src = IdeaRepository(scratch)
    dst = IdeaRepository(auth)
    new_rows = [i for i in src.list() if i.id > seeded_max_id]
    imported: list[int] = []
    skipped: list[dict] = []
    for idea in new_rows:
        if len(imported) >= max_new:
            skipped.append({"scratch_id": idea.id, "reason": "max_new"})
            continue
        if idea.category not in categories:
            skipped.append({"scratch_id": idea.id, "reason": "unknown_category"})
            continue
        if _oversized(idea):
            skipped.append({"scratch_id": idea.id, "reason": "oversized"})
            continue
        links, dropped = _resolve_links(settings, src.inspirations_of(idea.id))
        source_note = idea.source_note
        if dropped:
            marker = f"dropped_links: {','.join(dropped)}"
            source_note = f"{source_note}\n{marker}" if source_note else marker
        auth.execute("BEGIN IMMEDIATE")
        try:
            open_count = auth.execute(
                "SELECT COUNT(*) FROM ideas WHERE status=? AND claimed_by IS NULL",
                (IdeaStatus.OPEN.value,)).fetchone()[0]
            if open_count >= ceiling:
                auth.rollback()
                skipped.append({"scratch_id": idea.id, "reason": "ceiling"})
                continue
            if dst.find_collisions(title=idea.title, hypothesis=idea.hypothesis,
                                   family=idea.family):
                auth.rollback()
                skipped.append({"scratch_id": idea.id, "reason": "dedup_collision"})
                continue
            status, parked = _eligibility(idea)
            new_id = dst.insert_locked(
                title=_truncate(idea.title, MAX_TITLE),
                hypothesis=_truncate(idea.hypothesis, MAX_HYPOTHESIS),
                family=idea.family,
                tags=[_truncate(t, MAX_TAG) for t in idea.tags[:MAX_TAGS]],
                source_type=idea.source_type, source_ref=idea.source_ref,
                source_date=idea.source_date,
                source_note=_truncate_opt(source_note, MAX_SOURCE_NOTE),
                required_data=idea.required_data, status=status, category=idea.category,
                market=idea.market, horizon=idea.horizon,
                falsification=_truncate_opt(idea.falsification, MAX_FALSIFICATION),
                parked_reason=parked, inspirations=links, created_by_run=run_stamp)
            auth.commit()
        except BaseException:
            auth.rollback()
            raise
        imported.append(new_id)
    return {"imported": imported, "skipped": skipped}


def import_critic_rejections(auth: sqlite3.Connection, rows: list[dict], *, run_stamp: str,
                             max_rows: int) -> tuple[int, int]:
    """File the leap critic's rejections into the negative-result ledger.

    Returns `(recorded, errors)`. Each row is validated and written INSIDE its own try/except: the
    rows are model output read off `leap-critic.jsonl`, and one malformed row used to raise out of
    the whole import — AFTER the ideas had already been committed, so the caller's CLI reported a
    failure for a run whose authoritative work had actually succeeded. A row that cannot state
    WHY the critic rejected it (blank/non-string `reason` or `reason_kind`) is counted as an error
    rather than filed under a manufactured reason: a negative-result row with no real reason is
    noise the future leap prompt would read back as evidence.
    """
    recorded = 0
    errors = 0
    for row in rows[:max_rows]:
        try:
            kind = row.get("reason_kind") if isinstance(row, dict) else None
            reason = row.get("reason") if isinstance(row, dict) else None
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError(f"reason_kind is not a non-blank string: {kind!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"reason is not a non-blank string: {reason!r}")
            record_negative_result(
                auth, kind="discard", verdict=f"CRITIC:{kind.strip()[:40]}", actor="agent",
                reason=reason.strip(), source="auto:leap_critic",
                hypothesis=str(row.get("hypothesis") or ""), tags=f"leap:{run_stamp}")
            recorded += 1
        except Exception:
            errors += 1
    return recorded, errors


def refuted_with_reasons(conn: sqlite3.Connection, *, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT i.id, i.title, i.hypothesis, i.category, i.status, s.name AS strategy_name,"
        " a.outcome, a.reason, a.outcome_at FROM ideas i"
        " LEFT JOIN strategies s ON s.id = i.authored_strategy_id"
        " LEFT JOIN idea_attempts a ON a.id = (SELECT MAX(id) FROM idea_attempts"
        "   WHERE idea_id = i.id AND outcome IS NOT NULL)"
        " WHERE i.status = ? OR s.hypothesis_status = 'refuted'"
        " ORDER BY COALESCE(a.outcome_at, i.updated_at) DESC LIMIT ?",
        (IdeaStatus.REFUTED.value, limit)).fetchall()
    return [dict(r) for r in rows]


def reclassify(conn: sqlite3.Connection) -> dict:
    repo = IdeaRepository(conn)
    reopened: list[int] = []
    for idea in repo.list(status=IdeaStatus.NEEDS_DATA):
        status, parked = _eligibility(idea)
        if status is IdeaStatus.OPEN:
            repo.set_status(idea.id, to=IdeaStatus.OPEN)
            conn.execute("UPDATE ideas SET parked_reason=NULL WHERE id=?", (idea.id,))
            reopened.append(idea.id)
        elif parked != idea.parked_reason:
            conn.execute("UPDATE ideas SET parked_reason=? WHERE id=?", (parked, idea.id))
    conn.commit()
    return {"reopened": reopened}
