# algua/registry/idea_import.py
"""Trusted-driver writes that move agent output into the authoritative pool (spec §6/§7).

`import_ideas`: scratch rows (id > seeded max) -> authority under a fresh dedup + eligibility
check, one BEGIN IMMEDIATE per row. `import_critic_rejections`: DB-only negative-result rows.
`refuted_with_reasons`: what leap reads. `reclassify`: re-open parked rows after a capability flip.
"""
from __future__ import annotations

import sqlite3

from algua.contracts.idea import Idea, IdeaStatus
from algua.data.capabilities import (
    supported_capabilities,
    supported_horizons,
    supported_markets,
)
from algua.registry.ideas import IdeaRepository
from algua.registry.negative_results import record_negative_result
from algua.research.ideas import classify_idea


def _eligibility(idea: Idea) -> tuple[IdeaStatus, str | None]:
    return classify_idea(idea.required_data, supported_capabilities(), market=idea.market,
                         supported_markets=supported_markets(), horizon=idea.horizon,
                         supported_horizons=supported_horizons())


def import_ideas(auth: sqlite3.Connection, scratch: sqlite3.Connection, *, run_stamp: str,
                 max_new: int, ceiling: int, seeded_max_id: int) -> dict:
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
            links = src.inspirations_of(idea.id)
            new_id = dst._insert_locked(
                title=idea.title, hypothesis=idea.hypothesis, family=idea.family,
                tags=idea.tags, source_type=idea.source_type, source_ref=idea.source_ref,
                source_date=idea.source_date, source_note=idea.source_note,
                required_data=idea.required_data, status=status, category=idea.category,
                market=idea.market, horizon=idea.horizon, falsification=idea.falsification,
                parked_reason=parked, inspirations=links, created_by_run=run_stamp)
            auth.commit()
        except BaseException:
            auth.rollback()
            raise
        imported.append(new_id)
    return {"imported": imported, "skipped": skipped}


def import_critic_rejections(auth: sqlite3.Connection, rows: list[dict], *, run_stamp: str,
                             max_rows: int) -> int:
    n = 0
    for row in rows[:max_rows]:
        kind = str(row.get("reason_kind") or "unspecified")[:40]
        record_negative_result(
            auth, kind="discard", verdict=f"CRITIC:{kind}", actor="agent",
            reason=str(row.get("reason") or f"leap critic rejected: {kind}"),
            source="auto:leap_critic", hypothesis=str(row.get("hypothesis") or ""),
            tags=f"leap:{run_stamp}")
        n += 1
    return n


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
