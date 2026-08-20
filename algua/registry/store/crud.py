"""``StrategyStore`` (StrategyReader + StrategyLister) — strategy CRUD, organizational
metadata, and the stage-transition entry points."""
from __future__ import annotations

import sqlite3

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.contracts.types import ExitLaneGuard, PendingLiveAuthorization
from algua.registry.metadata import canonicalize_tags, dump_tags, load_tags
from algua.registry.repository import StrategyExists, StrategyNotFound, StrategyRecord
from algua.registry.store._util import _now
from algua.registry.store.base import TransitionMixin


def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"], name=row["name"], stage=Stage(row["stage"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
        family=row["family"],
        tags=load_tags(row["tags"]),
        author=Author(row["author"]) if row["author"] else Author.AGENT,
        hypothesis_status=(
            HypothesisStatus(row["hypothesis_status"])
            if row["hypothesis_status"] else HypothesisStatus.UNTESTED
        ),
        derived_from=row["derived_from"],
        description=row["description"],
    )


class CrudMixin(TransitionMixin):
    _conn: sqlite3.Connection

    def add(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author = Author.AGENT,
        hypothesis_status: HypothesisStatus = HypothesisStatus.UNTESTED,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)  # raises StrategyNotFound if the parent is unknown
        now = _now()
        with self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO strategies"
                    "(name, stage, created_at, updated_at, family, tags, author,"
                    " hypothesis_status, derived_from, description)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name, Stage.IDEA.value, now, now, family, dump_tags(tags or []),
                     author.value, hypothesis_status.value, derived_from, description),
                )
            except sqlite3.IntegrityError as exc:
                raise StrategyExists(name) from exc
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
            )
        return self.get(name)

    def get(self, name: str) -> StrategyRecord:
        row = self._conn.execute(
            "SELECT * FROM strategies WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise StrategyNotFound(f"strategy not found: {name}")
        return _row_to_record(row)

    def update_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
        derived_from: str | None = None,
        description: str | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> StrategyRecord:
        rec = self.get(name)
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)
        # One dict drives both the clause list and the param list, so a column and its value can
        # never drift out of lockstep — adding a field is a single line, not two parallel edits.
        updates: dict[str, object] = {}
        if family is not None:
            updates["family"] = family
        if author is not None:
            updates["author"] = author.value
        if hypothesis_status is not None:
            updates["hypothesis_status"] = hypothesis_status.value
        if derived_from is not None:
            updates["derived_from"] = derived_from
        if description is not None:
            updates["description"] = description
        if add_tags or remove_tags:
            tags = set(rec.tags)
            tags |= set(canonicalize_tags(add_tags or []))
            tags -= set(canonicalize_tags(remove_tags or []))
            updates["tags"] = dump_tags(tags)
        if updates:
            updates["updated_at"] = _now()
            clauses = ", ".join(f"{col} = ?" for col in updates)
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {clauses} WHERE id = ?",
                    [*updates.values(), rec.id],
                )
        return self.get(name)

    def list_strategies(
        self,
        stage: Stage | None = None,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
    ) -> list[StrategyRecord]:
        # Each clause carries its OWN params as a co-located tuple, so a multi-placeholder clause
        # (e.g. the COALESCE pairs below) can never fall out of sync with a separate param list.
        clauses: list[tuple[str, tuple[object, ...]]] = []
        if stage is not None:
            clauses.append(("stage = ?", (stage.value,)))
        if family is not None:
            clauses.append(("family = ?", (family,)))
        if author is not None:
            # COALESCE so legacy NULL rows (pre-metadata schema) match the default 'agent'.
            clauses.append(("COALESCE(author, ?) = ?", (Author.AGENT.value, author.value)))
        if hypothesis_status is not None:
            # Same NULL-legacy treatment; hypothesis_status defaults to 'untested'.
            clauses.append((
                "COALESCE(hypothesis_status, ?) = ?",
                (HypothesisStatus.UNTESTED.value, hypothesis_status.value),
            ))
        for tag in canonicalize_tags(tags or []):
            clauses.append((
                "EXISTS (SELECT 1 FROM json_each("
                "CASE WHEN json_valid(tags) THEN tags ELSE '[]' END"
                ") WHERE value = ?)",
                (tag,),
            ))
        where = f" WHERE {' AND '.join(c for c, _ in clauses)}" if clauses else ""
        params = [p for _, clause_params in clauses for p in clause_params]
        rows = self._conn.execute(
            f"SELECT * FROM strategies{where} ORDER BY id", params
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def backfill_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: str | None = None,
        hypothesis_status: str | None = None,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        """Fill only currently-NULL metadata columns (one-shot recovery). Uses COALESCE so any
        column already holding a value is left untouched. Idempotent: re-running is a no-op."""
        cols: dict[str, object] = {
            "family": family,
            "tags": dump_tags(tags) if tags is not None else None,
            "author": author,
            "hypothesis_status": hypothesis_status,
            "derived_from": derived_from,
            "description": description,
        }
        # Filter to columns where the caller provided a non-None value.
        to_fill = {c: v for c, v in cols.items() if v is not None}
        if to_fill:
            rec = self.get(name)
            # COALESCE keeps any existing non-NULL value; only NULLs are filled.
            assignments = ", ".join(f"{c} = COALESCE({c}, ?)" for c in to_fill)
            params: list[object] = [*to_fill.values(), rec.id]
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {assignments} WHERE id = ?", params
                )
        return self.get(name)

    def default_fill_metadata_nulls(self) -> None:
        """Fill every strategy row's author/hypothesis_status/tags column from its default when
        still NULL. Used as the terminal step of the backfill-from-kb command. Idempotent.

        Runs in a single transaction so a partial run is not committed.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE strategies SET author = COALESCE(author, ?)",
                (Author.AGENT.value,),
            )
            self._conn.execute(
                "UPDATE strategies SET hypothesis_status = COALESCE(hypothesis_status, ?)",
                (HypothesisStatus.UNTESTED.value,),
            )
            self._conn.execute("UPDATE strategies SET tags = COALESCE(tags, '[]')")

    def delete(self, name: str) -> None:
        """Remove a strategy row and its transition rows. ONLY for rolling back a failed
        ``strategy new`` that just created it — there is no general deletion workflow."""
        rec = self.get(name)
        with self._conn:
            self._conn.execute(
                "DELETE FROM stage_transitions WHERE strategy_id = ?", (rec.id,)
            )
            self._conn.execute("DELETE FROM strategies WHERE id = ?", (rec.id,))

    def list_transitions(self, name: str) -> list[dict]:
        rec = self.get(name)
        rows = self._conn.execute(
            "SELECT * FROM stage_transitions WHERE strategy_id = ? ORDER BY id", (rec.id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        consume_gate_id: int | None = None,
        consume_forward_gate_id: int | None = None,
        revoke_allocation: bool = False,
        live_authorization: PendingLiveAuthorization | None = None,
        exit_guard: ExitLaneGuard | None = None,
    ) -> StrategyRecord:
        if consume_gate_id is not None and consume_forward_gate_id is not None:
            raise ValueError(
                "at most one of consume_gate_id/consume_forward_gate_id may be set — a single"
                " transition spends a single token")
        if live_authorization is not None and revoke_allocation and not (
                rec.stage is Stage.FORWARD_TESTED and to is Stage.LIVE):
            # go-live (forward_tested->live, #497) is the ONE edge that legitimately carries BOTH a
            # live_authorization AND a revoke_allocation (it enters live UNALLOCATED, shedding any
            # paper-book slice). Every OTHER co-occurrence is a caller bug: the authorization write
            # belongs only to go-live, and revoke elsewhere is a plain wind-down.
            raise ValueError("live_authorization is incompatible with revoke_allocation")
        if live_authorization is not None and not (
                rec.stage is Stage.FORWARD_TESTED and to is Stage.LIVE and actor is Actor.HUMAN):
            # Defense in depth: the go-live authorization write belongs ONLY to a human
            # forward_tested->live transition, so the security invariant doesn't rest on
            # transition_strategy being the sole caller (codex #254 review).
            raise ValueError(
                "live_authorization is only valid for the human forward_tested->live transition")
        if exit_guard is not None and not revoke_allocation:
            # The source-lane open-order drain (#497 F2/H1) only makes sense on a book-exit edge
            # that sheds the allocation; wiring it onto a non-revoke transition is a caller bug.
            raise ValueError("exit_guard is only valid on a revoke_allocation transition")
        if revoke_allocation:
            # Bench wind-down (#125/#247): the live->dormant flatness check, the allocation revoke,
            # and the stage CAS must be ONE atomic critical section. Enforcing flatness in a
            # separate autocommit read (the caller) left a TOCTOU: a live fill committed between the
            # check and the CAS orphaned a position on a now-dormant strategy (run-all iterates
            # Stage.LIVE only). BEGIN IMMEDIATE takes the write lock up front so no concurrent fill
            # can land between the re-check below and the revoke+CAS. TOP-LEVEL ONLY (mirrors
            # reserve_holdout) — a manual BEGIN inside an open tx raises, and the blanket rollback
            # could roll back a surrounding tx.
            if self._conn.in_transaction:
                raise RuntimeError(
                    "apply_transition(revoke_allocation=True) must run at top level, not inside an"
                    " open transaction")
            if exit_guard is not None:
                # Cancel the strategy's own resting orders + ingest the venue feed BEFORE taking the
                # write lock (both are broker network calls + committing ingests — never hold the
                # registry write lock across them). This mirrors the `live flatten` ceremony's
                # cancel -> ingest, so a just-filled order lands in the ledger the under-lock
                # flatness check reads and a still-open order is caught by owned_open_order_ids
                # below. Runs OUTSIDE the try/BEGIN so its own commits are not swept into the
                # transaction that a flatness failure rolls back.
                exit_guard.cancel_and_ingest()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._assert_flat_for_bench(rec.name, rec.stage, exit_guard)
                result = self._apply_transition_locked(
                    rec, to, actor, reason, code_hash, config_hash, dependency_hash,
                    consume_gate_id, consume_forward_gate_id, _now(), revoke_allocation=True,
                    live_authorization=live_authorization)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return result
        with self._conn:  # consume + UPDATE + INSERT commit together or not at all
            return self._apply_transition_locked(
                rec, to, actor, reason, code_hash, config_hash, dependency_hash,
                consume_gate_id, consume_forward_gate_id, _now(),
                revoke_allocation=False, live_authorization=live_authorization)

    def intake_candidate_to_paper(
        self,
        rec: StrategyRecord,
        capital: float,
        actor: Actor,
        account_equity: float,
        max_concurrent: int,
    ) -> StrategyRecord:
        """Admit a CANDIDATE into the paper book in ONE atomic write (#317, finding #1/#3).

        Under a single top-level ``BEGIN IMMEDIATE`` write lock, in order: (1) re-check the
        max-concurrent count cap, (2) capital cap-check + allocation insert (Σ(active)+slice ≤
        equity, via the shared commit-less ``allocate_locked``), (3) the ``candidate→paper`` stage
        CAS + audit row (``_apply_transition_locked``, whose ``WHERE stage='candidate'`` re-asserts
        the source stage — closing the same ``candidate→…``-during-intake TOCTOU #246 closed for
        research-promote). Commit or roll back together, so there is no reachable
        allocated-but-still-candidate NOR transitioned-but-unallocated state. TOP-LEVEL ONLY
        (mirrors ``apply_transition(revoke_allocation=True)``): a manual ``BEGIN`` inside an open
        transaction raises, and the blanket ``BaseException`` rollback must own the whole txn.

        Raises ``CountCapReached`` (book full), ``AllocationError`` (no capital headroom), or
        ``TransitionError`` (the selection went stale — a concurrent transition moved the strategy
        out of ``candidate`` before the CAS). The count re-read is UNDER the write lock, so two
        concurrent intakes cannot both see ``count=cap-1`` and both admit.
        """
        from algua.registry import allocations

        if self._conn.in_transaction:
            raise RuntimeError(
                "intake_candidate_to_paper must run at top level, not inside an open transaction")
        if rec.stage is not Stage.CANDIDATE:
            # Friendly early error; the authoritative re-assert is the in-txn stage CAS below.
            raise TransitionError(
                f"{rec.name!r} is not a candidate (stage {rec.stage.value!r})")
        now = _now()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            count = allocations.active_paper_lane_count(self._conn)
            if count >= max_concurrent:
                raise allocations.CountCapReached(
                    f"paper book at capacity ({count}/{max_concurrent} active tenants)")
            allocations.allocate_locked(
                self._conn, rec.id, capital, actor.value, account_equity)
            result = self._apply_transition_locked(
                rec, Stage.PAPER, actor, "operator paper intake",
                None, None, None, None, None, now, revoke_allocation=False)
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return result

    def _assert_flat_for_bench(
        self, name: str, source: Stage, exit_guard: ExitLaneGuard | None = None
    ) -> None:
        """Re-check flatness on the SOURCE lane INSIDE the exit transaction (#247/#497): with the
        BEGIN IMMEDIATE write lock held, no concurrent fill can commit between this check and the
        revoke+CAS, so a strategy cannot leave its book (dormant/retired/lane-cross) while still
        holding an open — and thus orphaned — position. A LIVE source checks the live ledger; every
        other (paper-lane) source checks the paper-venue ledger. believed_positions is imported
        lazily — the registry->execution pattern transitions.py uses.

        FILLED positions are not the only orphan risk: a resting (submitted-but-unfilled) order left
        behind at the venue can fill AFTER the exit and orphan a position the source lane's run-all
        no longer iterates. When the caller injects an ``exit_guard`` (the broker-backed drain,
        wired for the LIVE lane, #497 F2/H1), its ``cancel_and_ingest`` already ran before the lock;
        we re-list the strategy's STILL-open orders UNDER the lock so a cancel that failed to remove
        one (a non-cancelable/partial state) blocks the revoke+CAS rather than orphaning it."""
        from algua.execution.live_ledger import LedgerKind, believed_positions
        kind = LedgerKind.LIVE if source is Stage.LIVE else LedgerKind.PAPER
        if believed_positions(self._conn, name, kind):
            raise TransitionError(
                f"{name} is not flat (open {kind.value} positions); flatten before this transition")
        if exit_guard is not None:
            open_ids = exit_guard.owned_open_order_ids()
            if open_ids:
                raise TransitionError(
                    f"{name} is not flat ({len(open_ids)} open {kind.value} order(s) {open_ids}); "
                    "flatten before this transition")
