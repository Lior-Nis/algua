"""``FamilyGraph`` — family registry, parentage DAG, and family-scoped breadth accounting
(#222), plus the #524 agent-NOVEL mint governance constants and bounds."""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from algua.registry.db import MAX_N_COMBOS
from algua.registry.repository import AgentMintCapError
from algua.registry.store._util import _now

# --- #524 agent-NOVEL mint governance constants (R9-M1) --------------------------------------
# CODEOWNERS-protected: these live in algua/registry/store/family.py as module constants — NOT a
# CLI flag, NOT an env var — so the autonomous loop has no surface to read or raise them. Changing
# either requires a human PR to this protected file (the same human-gate as the promote relaxation
# flags).
AGENT_NOVEL_MINT_WINDOW_DAYS = 90   # rolling window for the burst rate cap (matches FUNNEL_WINDOW)
AGENT_NOVEL_MINT_CAP = 8            # max agent mints per rolling window (SOLE automatic bound)


def _parse_canonical_utc(ts: str) -> datetime:
    """Parse a canonical UTC ISO-8601 timestamp (offset-aware, +00:00). Raises ValueError on a
    naive/local/malformed value so the #524 rate-cap read can fail closed rather than mis-bucket a
    row across the window cutoff."""
    dt = datetime.fromisoformat(ts)  # ValueError on malformed
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"non-canonical-UTC timestamp {ts!r}")
    return dt


class FamilyGraphMixin:
    _conn: sqlite3.Connection

    if TYPE_CHECKING:
        # Provided by search_breadth.py's SearchBreadthLedgerMixin. TYPE_CHECKING-only so it can
        # never shadow the real implementation in the facade's MRO (see gate.py's note).
        def funnel_lifetime_search_combos(self) -> int: ...

    def family_graph_fingerprint(self) -> tuple[int, ...]:
        # v37 (#524, R9): a monotone digest over EVERY DB table the NOVEL classifier reads. The
        # append-only triggers (db.py) make (COUNT, MAX(id)) exact: any INSERT strictly increases a
        # component; a member removal (removed_at UPDATE) strictly decreases the active-only COUNT;
        # persisted member profiles ride on immutable family_members rows so the profile axis is
        # covered here. Boundary-clean (pure SQL, no algua.research). A mismatch between two reads
        # means a concurrent family mint / member (re)assignment or removal / parentage edge /
        # member-returns refresh landed — the CAS the mint uses to detect stale-NOVEL drift.
        def _cm(sql: str) -> tuple[int, int]:
            r = self._conn.execute(sql).fetchone()
            return int(r[0]), int(r[1])

        fam = _cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM families")
        mem_all = _cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM family_members")
        mem_active = self._conn.execute(
            "SELECT COUNT(*) FROM family_members WHERE removed_at IS NULL").fetchone()[0]
        # The one-time legacy profile materialisation (member_code_hash NULL→value) is the ONLY
        # permitted UPDATE that mutates a classifier-read column WITHOUT changing COUNT/MAX(id) or
        # the active count. Include the active-NULL-profile count so that flip DOES bump the
        # fingerprint: a concurrent materialisation between fp_before and the under-lock re-check
        # then trips the CAS (FamilyGraphDriftError, fail-closed) instead of silently changing the
        # member profile the NOVEL verdict was computed against.
        mem_null_profile = self._conn.execute(
            "SELECT COUNT(*) FROM family_members"
            " WHERE removed_at IS NULL AND member_code_hash IS NULL").fetchone()[0]
        parents = _cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM family_parents")
        events = _cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM family_events")
        returns = _cm("SELECT COUNT(*), COALESCE(MAX(id),0) FROM backtest_returns")
        return (*fam, *mem_all, int(mem_active), int(mem_null_profile),
                *parents, *events, *returns)

    def create_family(
        self,
        name: str,
        actor: str,
        created_by_strategy: str | None = None,
    ) -> int:
        """Create a new family and record the family_created event. Return the new family id. The
        family carries ``seeded_prior_combos = 0`` (a fresh zero-prior family). The agent-NOVEL
        seeded family is minted by RAW INSERT inside the promote transaction (#524), NOT via this
        public helper, so there is no agent-reachable seeded-create surface."""
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO families"
                "(name, created_at, created_by_actor, created_by_strategy, seeded_prior_combos)"
                " VALUES (?,?,?,?,0)",
                (name, now, actor, created_by_strategy),
            )
            family_id = cur.lastrowid
            assert family_id is not None
            self._conn.execute(
                "INSERT INTO family_events(event_type, family_id, actor, created_at)"
                " VALUES (?,?,?,?)",
                ("family_created", family_id, actor, now),
            )
        return int(family_id)

    def assign_strategy_to_family(
        self,
        strategy_name: str,
        family_id: int,
        actor: str,
        *,
        verdict: str,
        similarity_score: float,
        clustering_version: str,
        clustering_config_json: str,
        axis_json: str,
        matched_family_id: int | None = None,
        member_code_hash: str | None = None,
        member_factors_json: str | None = None,
    ) -> None:
        """Assign a strategy to a family (append-only: old row gets removed_at set). #524: the
        joining member's classified ``(code_hash, sorted factors)`` are persisted onto the new
        ``family_members`` row so the classifier reads member profiles from immutable DB state."""
        now = _now()
        event_type = "strategy_merged" if verdict == "MERGE" else "strategy_assigned"
        with self._conn:
            # If an active membership row exists, soft-delete it first.
            self._conn.execute(
                "UPDATE family_members SET removed_at=?"
                " WHERE strategy_name=? AND removed_at IS NULL",
                (now, strategy_name),
            )
            self._conn.execute(
                "INSERT INTO family_members"
                "(family_id, strategy_name, joined_at, joined_by_actor, removed_at,"
                " member_code_hash, member_factors_json)"
                " VALUES (?,?,?,?,NULL,?,?)",
                (family_id, strategy_name, now, actor,
                 member_code_hash, member_factors_json),
            )
            self._conn.execute(
                "INSERT INTO family_events"
                "(event_type, family_id, strategy_name, actor,"
                " clustering_verdict, similarity_score, clustering_version,"
                " clustering_config_json, axis_json, matched_family_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event_type, family_id, strategy_name, actor,
                 verdict, similarity_score, clustering_version,
                 clustering_config_json, axis_json, matched_family_id, now),
            )

    def strategy_family(self, strategy_name: str) -> int | None:
        """Return the current (active) family_id for the strategy, or None."""
        row = self._conn.execute(
            "SELECT family_id FROM family_members WHERE strategy_name=? AND removed_at IS NULL",
            (strategy_name,),
        ).fetchone()
        return int(row["family_id"]) if row is not None else None

    def family_count(self) -> int:
        # #532: raw table count, NOT an active-member count. Distinguishes a true cold-start (0
        # rows) from "families exist but no active member" (COUNT>0, members soft-deleted). It is
        # fingerprint component 0 (family_graph_fingerprint's leading families COUNT), so reading it
        # at classification time inside the fp_before/fp_after window adds no TOCTOU: a concurrent
        # mint bumps the fingerprint and trips the existing CAS.
        return int(self._conn.execute("SELECT COUNT(*) FROM families").fetchone()[0])

    def family_ancestry(self, family_id: int) -> list[int]:
        """BFS-transitive list of all ancestor family_ids (cycle-safe via visited set)."""
        visited: set[int] = {family_id}
        queue: deque[int] = deque([family_id])
        ancestors: list[int] = []
        while queue:
            current = queue.popleft()
            rows = self._conn.execute(
                "SELECT parent_family_id FROM family_parents WHERE child_family_id=?",
                (current,),
            ).fetchall()
            for row in rows:
                pid = int(row[0])
                if pid not in visited:
                    visited.add(pid)
                    ancestors.append(pid)
                    queue.append(pid)
        return ancestors

    def add_parent_edge(self, child_family_id: int, parent_family_id: int) -> None:
        """Atomically add a parent edge (cycle-guarded, BEGIN IMMEDIATE, top-level-only)."""
        if self._conn.in_transaction:
            raise RuntimeError(
                "add_parent_edge must be called at top level, not inside an open transaction"
            )
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # Cycle guard: adding edge (child, parent) creates a cycle iff parent_family_id is
            # already an ancestor of child_family_id, OR parent == child (self-edge).
            # Cycle means: following PARENT edges from parent_family_id upward reaches
            # child_family_id (closing a loop: child -> parent -> ... -> child).
            # Equivalently: child_family_id must not already be an ancestor-or-self of
            # parent_family_id when we add child as parent's ancestor.
            # Correct check: BFS from parent_family_id following PARENT links (going up the
            # ancestry). If child_family_id appears, the new edge would form a cycle.
            if parent_family_id == child_family_id:
                raise ValueError(
                    f"cycle detected: cannot add self-edge {child_family_id} -> {parent_family_id}"
                )
            visited: set[int] = {parent_family_id}
            queue: deque[int] = deque([parent_family_id])
            while queue:
                current = queue.popleft()
                rows = self._conn.execute(
                    "SELECT parent_family_id FROM family_parents WHERE child_family_id=?",
                    (current,),
                ).fetchall()
                for row in rows:
                    pid = int(row[0])
                    if pid not in visited:
                        visited.add(pid)
                        queue.append(pid)
            if child_family_id in visited:
                raise ValueError(
                    f"cycle detected: adding edge {child_family_id} -> {parent_family_id}"
                    f" would create a cycle"
                )
            now = _now()
            self._conn.execute(
                "INSERT INTO family_parents(child_family_id, parent_family_id)"
                " VALUES (?,?)",
                (child_family_id, parent_family_id),
            )
            self._conn.execute(
                "INSERT INTO family_events"
                "(event_type, family_id, actor, created_at)"
                " VALUES (?,?,?,?)",
                ("parent_edge_added", child_family_id, "system", now),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def all_families_with_member_profiles(self) -> list[tuple[int, list[dict]]]:
        """Return [(family_id, members_list)] for all families that have active members.

        Each member dict: {"name": str, "code_hash": str, "factors": set[str]}.

        v37 (#524, R9-H3): the classifier's member-profile input is now IMMUTABLE DB STATE. Each
        active row's persisted ``member_code_hash``/``member_factors_json`` (materialised at
        assignment) are read directly. A pre-#524 row still carrying NULL (un-materialised legacy)
        falls back to a live ``compute_artifact_hashes``/``factors_used_by`` recompute — the
        one-time ``materialise_legacy_member_profiles`` bootstrap eliminates that fallback in steady
        state. A live recompute that cannot load the module fails closed (empty hash/factors).
        """
        rows = self._conn.execute(
            "SELECT DISTINCT family_id, strategy_name, member_code_hash, member_factors_json"
            " FROM family_members"
            " WHERE removed_at IS NULL"
            " ORDER BY family_id"
        ).fetchall()
        family_map: dict[int, list[dict]] = {}
        for row in rows:
            fid = int(row["family_id"])
            sname = row["strategy_name"]
            if row["member_code_hash"] is not None:
                code_hash = row["member_code_hash"]
                factors = set(json.loads(row["member_factors_json"] or "[]"))
            else:
                code_hash, factors = self._live_member_profile(sname)
            family_map.setdefault(fid, []).append({
                "code_hash": code_hash, "factors": factors, "name": sname,
            })
        return list(family_map.items())

    @staticmethod
    def _live_member_profile(strategy_name: str) -> tuple[str, set[str]]:
        """Recompute a member's (code_hash, factor-name set) from module source. Fail-closed to
        ('', set()) when the module cannot be loaded (a legacy/test name)."""
        from algua.registry.approvals import compute_artifact_hashes
        from algua.registry.lineage import factors_used_by

        try:
            code_hash = compute_artifact_hashes(strategy_name).code_hash
        except Exception:  # noqa: BLE001
            code_hash = ""
        try:
            factor_specs = factors_used_by(strategy_name)
            factors: set[str] = {
                f.name if hasattr(f, "name") else str(f) for f in factor_specs
            }
        except Exception:  # noqa: BLE001
            factors = set()
        return code_hash, factors

    def materialise_legacy_member_profiles(self) -> int:
        """One-time (#524): persist ``member_code_hash``/``member_factors_json`` for active
        ``family_members`` rows still carrying NULL profiles, via the trigger-permitted NULL→value
        UPDATE. Idempotent. Returns the number materialised. Loads modules — call from the store
        bootstrap, never under the promote write lock."""
        rows = self._conn.execute(
            "SELECT id, strategy_name FROM family_members"
            " WHERE removed_at IS NULL AND member_code_hash IS NULL"
        ).fetchall()
        if not rows:
            return 0  # steady state: no empty write-locked transaction, no module loads
        n = 0
        with self._conn:
            for row in rows:
                code_hash, factors = self._live_member_profile(row["strategy_name"])
                self._conn.execute(
                    "UPDATE family_members SET member_code_hash=?, member_factors_json=?"
                    " WHERE id=?",
                    (code_hash, json.dumps(sorted(factors)), int(row["id"])),
                )
                n += 1
        return n

    def agent_novel_mint_audit(self) -> dict:
        """Read-only mint-governance stats for the ``family-audit`` advisory block (#524)."""
        cutoff = datetime.now(UTC) - timedelta(days=AGENT_NOVEL_MINT_WINDOW_DAYS)
        # Bucket the window with the SAME canonical-UTC parse the enforcement path uses (a naive
        # string `created_at >= ?` compare could mis-bucket a non-canonical timestamp and disagree
        # with check_agent_novel_mint_bounds). A row whose created_at is not canonical UTC is
        # surfaced as corruption (advisory — audit never raises) rather than silently in/out of the
        # window.
        agent_rows = self._conn.execute(
            "SELECT created_at FROM families WHERE created_by_actor='agent'").fetchall()
        mints_in_window = 0
        created_at_corruption = 0
        for r in agent_rows:
            try:
                if _parse_canonical_utc(r[0]) >= cutoff:
                    mints_in_window += 1
            except ValueError:
                created_at_corruption += 1
        lifetime_consumed = len(agent_rows)
        n_rows, n_well_typed = self._conn.execute(
            "SELECT COUNT(*), COUNT(CASE WHEN typeof(n_combos)='integer'"
            " AND n_combos BETWEEN 1 AND ? THEN 1 END) FROM search_trials",
            (MAX_N_COMBOS,),
        ).fetchone()
        return {
            "mints_in_window": mints_in_window,
            "window_cap": AGENT_NOVEL_MINT_CAP,
            "window_days": AGENT_NOVEL_MINT_WINDOW_DAYS,
            "lifetime_consumed": lifetime_consumed,
            "search_trials_corruption_count": int(n_rows) - int(n_well_typed),
            "created_at_corruption_count": created_at_corruption,
        }

    def agent_novel_mint_seed(self) -> int:
        """The durable breadth prior an agent-NOVEL family is seeded with (#524): the funnel-wide
        LIFETIME search effort (WHERE-filtered, overflow-safe). Identical to
        ``funnel_lifetime_search_combos``; the mint additionally requires it be > 0."""
        return self.funnel_lifetime_search_combos()

    def check_agent_novel_mint_bounds(self) -> None:
        """Fail-closed on the SOLE automatic agent-NOVEL mint bound (#524, R9): the per-window rate
        cap (``AgentMintCapError``). Counts the canonical ``families`` table (not the derived event
        stream). Parses each counted agent row's ``created_at`` as canonical UTC and fail-closes if
        any does not parse, so a stray naive/local timestamp cannot silently mis-bucket a row across
        the cutoff. Safe to call lock-free (pre-check) OR under the promote write lock
        (authoritative re-check).

        #532 (3b) scope note: this rate cap counts live rows in the ``families`` table of the
        CURRENTLY CONNECTED registry DB file — it is per-DB, NOT global across environments.
        Staging/CI must use a separate ``registry.sqlite`` from prod (the norm); a staging promote
        consumes only staging's budget. A DB wipe/rotation resets the window with it (the cap counts
        live rows, not an append-only external ledger) — acceptable because the cap is a throughput
        bound on autonomous minting within one registry, not a cross-environment audit trail."""
        cutoff = datetime.now(UTC) - timedelta(days=AGENT_NOVEL_MINT_WINDOW_DAYS)
        rows = self._conn.execute(
            "SELECT created_at FROM families WHERE created_by_actor='agent'"
        ).fetchall()
        in_window = 0
        for r in rows:
            try:
                created = _parse_canonical_utc(r[0])
            except ValueError as exc:
                raise AgentMintCapError(
                    f"agent family created_at is not canonical UTC ({exc}); refusing to mint "
                    "to avoid mis-bucketing the rate-cap window") from exc
            if created >= cutoff:
                in_window += 1
        if in_window >= AGENT_NOVEL_MINT_CAP:
            raise AgentMintCapError(
                f"agent-NOVEL mint rate cap reached: {in_window} agent families in the last "
                f"{AGENT_NOVEL_MINT_WINDOW_DAYS}d (cap {AGENT_NOVEL_MINT_CAP}); wait out the "
                "window or promote --actor human --new-family")

    def _family_member_strategies(self, family_id: int) -> list[str]:
        """DISTINCT strategy names for a family and all its transitive ancestors."""
        ancestor_ids = [family_id] + self.family_ancestry(family_id)
        placeholders = ",".join("?" * len(ancestor_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT fm.strategy_name FROM family_members fm"
            f" WHERE fm.family_id IN ({placeholders})",
            ancestor_ids,
        ).fetchall()
        return [row[0] for row in rows]

    def windowed_family_combos(self, family_id: int, window_days: int) -> int:
        """Windowed search combos for a family + transitive ancestors.

        Like family_lifetime_combos but filtered to search_trials created within
        the trailing window_days. Used for informational output and gate_evaluations
        audit field; NOT used in the 3-way max (which uses lifetime).
        """
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        member_strategies = self._family_member_strategies(family_id)
        if not member_strategies:
            return 0
        placeholders = ",".join("?" * len(member_strategies))
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
            f" WHERE st.created_at >= ? AND st.strategy_name IN ({placeholders})",
            [cutoff, *member_strategies],
        ).fetchone()
        return int(row[0])

    def lifetime_combos_for_families(self, family_ids: Iterable[int]) -> int:
        """Lifetime search combos across the UNION of the given families + all their
        transitive ancestors. A strategy reachable via several of the families is counted
        exactly once (the union of member-strategy sets is deduped before the sum)."""
        all_strategies: set[str] = set()
        all_family_ids: set[int] = set()
        for fid in family_ids:
            all_strategies.update(self._family_member_strategies(fid))
            all_family_ids.add(fid)
            all_family_ids.update(self.family_ancestry(fid))
        search_sum = 0
        if all_strategies:
            placeholders = ",".join("?" * len(all_strategies))
            # Same well-typed-row policy as funnel_lifetime_search_combos()/the mint-seed path
            # (typeof='integer' AND 1..MAX_N_COMBOS): a legacy corrupt row (pre-#524, no CHECK
            # constraint) for a family member must be EXCLUDED, not silently coerced to 0 by SUM.
            # Without this filter the two lifetime-accounting paths could disagree and a corrupt
            # row could quietly undercount a sibling's true family_lifetime tax.
            row = self._conn.execute(
                f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
                f" WHERE st.strategy_name IN ({placeholders})"
                f" AND typeof(st.n_combos)='integer' AND st.n_combos BETWEEN 1 AND ?",
                [*all_strategies, MAX_N_COMBOS],
            ).fetchone()
            search_sum = int(row[0])
        # v37 (#524): add the durable breadth PRIOR of the family + all its transitive ancestors
        # (deduped by set over the ancestor closure). The seed is a lifetime-only prior — NOT in the
        # windowed/funnel-wide sums — so an agent-founded NOVEL family starts as if it had already
        # accumulated the funnel-wide lifetime tests, removing the reset-gaming incentive. Every
        # pre-#524 family has seed 0, so all current anti-reset numbers are unchanged. A NEGATIVE
        # seed is impossible via the mint (which asserts seed>0) and the fresh-DB CHECK; treat any
        # negative as corruption (fail closed) rather than silently subtracting from the tax.
        seed_sum = 0
        if all_family_ids:
            fam_placeholders = ",".join("?" * len(all_family_ids))
            seed_row = self._conn.execute(
                f"SELECT COALESCE(SUM(seeded_prior_combos), 0),"
                f" COALESCE(MIN(seeded_prior_combos), 0) FROM families"
                f" WHERE id IN ({fam_placeholders})",
                list(all_family_ids),
            ).fetchone()
            if int(seed_row[1]) < 0:
                raise ValueError(
                    "families.seeded_prior_combos holds a negative value (corruption); the "
                    "breadth prior must be >= 0")
            seed_sum = int(seed_row[0])
        return search_sum + seed_sum

    def family_lifetime_combos(self, family_id: int) -> int:
        """Lifetime search combos across this family + all transitive ancestors."""
        return self.lifetime_combos_for_families([family_id])

    def family_names(self) -> dict[int, str]:
        """All family ids → names (read-only)."""
        rows = self._conn.execute("SELECT id, name FROM families").fetchall()
        return {int(r[0]): r[1] for r in rows}
