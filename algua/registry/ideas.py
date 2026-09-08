# algua/registry/ideas.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from algua.contracts.idea import (
    DataCapability,
    Horizon,
    Idea,
    IdeaStatus,
    Market,
    Obscurity,
    SourceType,
    can_change_status,
)
from algua.registry.metadata import dump_tags, load_tags
from algua.research.idea_dedup import is_collision, signature

# `IdeaRepository.list` shadows the builtin inside the class namespace, so annotations on later
# methods that say `list[...]` would resolve to the method, not the builtin. Reference the builtin
# through this alias in those annotations.
_list = list


class IdeaNotFound(LookupError):
    def __init__(self, idea_id: int) -> None:
        super().__init__(f"idea {idea_id} not found")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dump_caps(caps: list[DataCapability]) -> str:
    return json.dumps([c.value for c in caps])


def _load_caps(value: str | None) -> list[DataCapability]:
    if not value:
        return []
    return [DataCapability(c) for c in json.loads(value)]


def _row_to_idea(row: sqlite3.Row) -> Idea:
    # `row.keys()` guard: pre-v46 rows in tests that hand-build a bare `ideas` table (without the
    # eight new columns) must never KeyError here.
    keys = set(row.keys())

    def _opt(name: str) -> str | None:
        return row[name] if name in keys else None

    market = _opt("market")
    horizon = _opt("horizon")
    return Idea(
        id=row["id"], title=row["title"], hypothesis=row["hypothesis"],
        family=row["family"], tags=load_tags(row["tags"]),
        source_type=SourceType(row["source_type"]),
        source_ref=row["source_ref"], source_date=row["source_date"],
        source_note=row["source_note"], required_data=_load_caps(row["required_data"]),
        status=IdeaStatus(row["status"]), signature=row["signature"],
        authored_strategy_id=row["authored_strategy_id"],
        duplicate_of_idea_id=row["duplicate_of_idea_id"],
        override_reason=row["override_reason"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        category=_opt("category"),
        market=Market(market) if market else None,
        horizon=Horizon(horizon) if horizon else None,
        falsification=_opt("falsification"), parked_reason=_opt("parked_reason"),
        claimed_by=_opt("claimed_by"), claim_token=_opt("claim_token"),
        claimed_at=_opt("claimed_at"),
    )


@dataclass(frozen=True)
class InspirationLink:
    """One inspiration an idea leaped from (kb/inspirations note id + venue + obscurity)."""

    inspiration_id: str
    venue: str
    obscurity: Obscurity


@dataclass
class Collision:
    """A colliding existing idea, with its effective status: AUTHORED is downgraded to REFUTED
    when the linked strategy is currently refuted (the refuted wall)."""

    idea: Idea
    effective_status: IdeaStatus


class IdeaRepository:
    """sqlite-backed idea pool: the only module that embeds idea SQL. Shares the registry
    connection (same DB as strategies), so the refuted-aware dedup can join across the two."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self, *, title: str, hypothesis: str, family: str | None, tags: list[str],
        source_type: SourceType, source_ref: str | None, source_date: str | None,
        source_note: str | None, required_data: list[DataCapability], status: IdeaStatus,
        duplicate_of_idea_id: int | None = None, override_reason: str | None = None,
        category: str | None = None, market: Market | None = None,
        horizon: Horizon | None = None, falsification: str | None = None,
        parked_reason: str | None = None, inspirations: list[InspirationLink] | None = None,
        created_by_run: str | None = None,
    ) -> Idea:
        # `with self._conn:` commits on exit, so `add()` cannot be nested inside a caller's own
        # BEGIN IMMEDIATE (it would commit the caller's transaction early). Task 4's bulk import
        # calls `_insert_locked` directly inside its own transaction instead.
        with self._conn:
            rowid = self._insert_locked(
                title=title, hypothesis=hypothesis, family=family, tags=tags,
                source_type=source_type, source_ref=source_ref, source_date=source_date,
                source_note=source_note, required_data=required_data, status=status,
                duplicate_of_idea_id=duplicate_of_idea_id, override_reason=override_reason,
                category=category, market=market, horizon=horizon, falsification=falsification,
                parked_reason=parked_reason, inspirations=inspirations,
                created_by_run=created_by_run,
            )
        return self.get(rowid)

    def _insert_locked(
        self, *, title: str, hypothesis: str, family: str | None, tags: list[str],
        source_type: SourceType, source_ref: str | None, source_date: str | None,
        source_note: str | None, required_data: list[DataCapability], status: IdeaStatus,
        duplicate_of_idea_id: int | None = None, override_reason: str | None = None,
        category: str | None = None, market: Market | None = None,
        horizon: Horizon | None = None, falsification: str | None = None,
        parked_reason: str | None = None, inspirations: list[InspirationLink] | None = None,
        created_by_run: str | None = None,
    ) -> int:
        """Insert the idea row (+ its inspiration links) WITHOUT committing — the caller owns the
        transaction (either `add()`'s `with self._conn:` or a caller's own BEGIN IMMEDIATE)."""
        # add() always inserts authored_strategy_id=None, so an AUTHORED idea created here would
        # violate the AUTHORED<->authored_strategy_id invariant set_status enforces (and silently
        # defeat the refuted-wall join in find_collisions). Authoring goes through
        # set_status(to=AUTHORED, authored_strategy_id=...); reject it at this write site (#267).
        if status is IdeaStatus.AUTHORED:
            raise ValueError(
                "add() cannot create an AUTHORED idea (it has no strategy link); "
                "author via set_status(to=AUTHORED, authored_strategy_id=...)"
            )
        sig = signature(title, hypothesis)
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO ideas(title, hypothesis, family, tags, source_type, source_ref,"
            " source_date, source_note, required_data, status, signature,"
            " authored_strategy_id, duplicate_of_idea_id, override_reason,"
            " created_at, updated_at, category, market, horizon, falsification, parked_reason)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (title, hypothesis, family, dump_tags(tags), source_type.value, source_ref,
             source_date, source_note, _dump_caps(required_data), status.value, sig,
             None, duplicate_of_idea_id, override_reason, now, now, category,
             market.value if market else None, horizon.value if horizon else None,
             falsification, parked_reason),
        )
        rowid = cur.lastrowid
        assert rowid is not None
        for link in inspirations or []:
            self._conn.execute(
                "INSERT INTO idea_inspirations(idea_id, inspiration_id, venue, obscurity,"
                " created_by_run) VALUES (?,?,?,?,?)",
                (rowid, link.inspiration_id, link.venue, link.obscurity.value,
                 created_by_run or "manual"),
            )
        return int(rowid)

    def inspirations_of(self, idea_id: int) -> _list[InspirationLink]:
        rows = self._conn.execute(
            "SELECT inspiration_id, venue, obscurity FROM idea_inspirations WHERE idea_id=?"
            " ORDER BY inspiration_id", (idea_id,))
        return [InspirationLink(r["inspiration_id"], r["venue"], Obscurity(r["obscurity"]))
                for r in rows]

    def get(self, idea_id: int) -> Idea:
        row = self._conn.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
        if row is None:
            raise IdeaNotFound(idea_id)
        return _row_to_idea(row)

    def list(
        self, *, status: IdeaStatus | None = None, family: str | None = None,
        limit: int | None = None,
    ) -> list[Idea]:
        sql = "SELECT * FROM ideas"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit is not None:
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
        else:
            sql += " ORDER BY id"
        return [_row_to_idea(r) for r in self._conn.execute(sql, params)]

    def find_collisions(
        self, *, title: str, hypothesis: str, family: str | None, threshold: float = 0.6
    ) -> _list[Collision]:
        """Non-discarded ideas colliding with the candidate (family-safe + Jaccard). Each
        collision's effective_status downgrades AUTHORED -> REFUTED when the idea's linked
        strategy is currently hypothesis_status='refuted' — the refuted wall, via a read-only
        join (no idea-row mutation, no protected-file change)."""
        cand_sig = signature(title, hypothesis)
        rows = self._conn.execute(
            "SELECT i.*, s.hypothesis_status AS strat_status FROM ideas i"
            " LEFT JOIN strategies s ON s.id = i.authored_strategy_id"
            " WHERE i.status != ?",
            (IdeaStatus.DISCARDED.value,),
        ).fetchall()
        out: list[Collision] = []
        for row in rows:
            if is_collision(cand_sig, family, row["signature"], row["family"],
                            threshold=threshold):
                idea = _row_to_idea(row)
                effective = idea.status
                if idea.status is IdeaStatus.AUTHORED and row["strat_status"] == "refuted":
                    effective = IdeaStatus.REFUTED
                out.append(Collision(idea=idea, effective_status=effective))
        return out

    def set_status(
        self, idea_id: int, *, to: IdeaStatus, authored_strategy_id: int | None = None
    ) -> Idea:
        idea = self.get(idea_id)
        if not can_change_status(idea.status, to):
            raise ValueError(
                f"illegal idea status change {idea.status.value} -> {to.value}")
        if to is IdeaStatus.AUTHORED and authored_strategy_id is None:
            raise ValueError("authored status requires a strategy link")
        if to is not IdeaStatus.AUTHORED and authored_strategy_id is not None:
            raise ValueError("authored_strategy_id is only valid for the authored status")
        with self._conn:
            self._conn.execute(
                "UPDATE ideas SET status=?,"
                " authored_strategy_id=COALESCE(?, authored_strategy_id), updated_at=?"
                " WHERE id=?",
                (to.value, authored_strategy_id, _now(), idea_id),
            )
        return self.get(idea_id)

    def windowed_idea_counts(self, window_days: int) -> dict[str, int]:
        """Idea counts BY STATUS created within the trailing window — the funnel-breadth signal
        the later (human, CODEOWNERS) gate change will consume. By-status (not one number) so the
        gate can pick the right denominator. ISO-8601 UTC strings compare chronologically, so a
        string >= on created_at is correct."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        counts: dict[str, int] = {s.value: 0 for s in IdeaStatus}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ideas WHERE created_at >= ? GROUP BY status",
            (cutoff,),
        ):
            counts[row["status"]] = int(row["n"])
        counts["total"] = sum(counts[s.value] for s in IdeaStatus)
        return counts
