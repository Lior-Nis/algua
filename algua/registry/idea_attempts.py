# algua/registry/idea_attempts.py
"""Claims, attempts and outcomes for the idea pool (spec 2026-09-08 §7).

Every write here is a driver's write: `claim` before a research run, `record_outcome` after,
`link` from the merge-back drainer. Claims are fenced by a UUID token; attempts are append-only.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

from algua.contracts.idea import (
    REFUTING_OUTCOMES,
    AttemptOutcome,
    Idea,
    IdeaStatus,
)
from algua.registry.ideas import IdeaRepository

# `IdeaAttemptsRepository.claim`'s return type says `list[Idea]`; nothing here shadows the
# builtin, but the alias mirrors ideas.py's convention for consistency in this pair of modules.
_list = list

#: How long a claim whose attempt already reached `candidate_preview_pass` is HELD before the
#: reaper abandons it — a second, much longer clock than `ttl_minutes` (see `_reap`). Mirrored by
#: `Settings.idea_preview_hold_hours`, which is what the CLI actually passes;
#: `tests/test_settings_ideation.py` asserts the two never drift.
DEFAULT_PREVIEW_HOLD_HOURS = 72

#: The outcomes an attempt already recorded as `candidate_preview_pass` may be REWRITTEN to, both
#: written only by the merge-back drainer once the authoritative run has spoken: `integrity_fail`
#: (the authoritative promote refused the strategy — a real refutation) and `run_error` (the
#: merge-back died for an infrastructure reason: diff policy, an exhausted retry budget — which
#: says nothing about the hypothesis, so it must NOT refute the idea). Every other rewrite,
#: including re-recording `candidate_preview_pass`, still raises.
PREVIEW_PASS_REWRITES: frozenset[AttemptOutcome] = frozenset({
    AttemptOutcome.INTEGRITY_FAIL, AttemptOutcome.RUN_ERROR,
})


class ClaimTokenMismatch(ValueError):
    """The (idea, token) pair does not match a live claim — stale run, wrong idea, or released."""


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class IdeaAttemptsRepository:
    """sqlite-backed claim/attempt/outcome ledger over the `ideas` pool. Shares the registry
    connection with `IdeaRepository` (same DB, same write lock)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ideas = IdeaRepository(conn)

    # -- claim ---------------------------------------------------------------------------
    def claim(self, *, run_stamp: str, limit: int, ttl_minutes: int,
              now: datetime | None = None, category: str | None = None,
              preview_hold_hours: int = DEFAULT_PREVIEW_HOLD_HOURS) -> _list[Idea]:
        """Reap expired claims, then claim up to `limit` open ideas, in ONE BEGIN IMMEDIATE.

        Selection: round-robin over categories (fewest claims in the trailing 7 days first),
        then obscurity rare>niche>common>canon (best linked inspiration), then oldest created.
        Legacy NULL-category rows are eligible only when no categorized row is. `category`, when
        given, restricts eligibility to that one category (legacy NULL rows excluded).

        `preview_hold_hours` is the reaper's SECOND clock, for claims deliberately held open by a
        `candidate_preview_pass` outcome — see `_reap`."""
        if self._conn.in_transaction:
            raise RuntimeError("claim must run at top level, not inside an open transaction")
        now = now or datetime.now(UTC)
        cutoff = _iso(now - timedelta(minutes=ttl_minutes))
        preview_cutoff = _iso(now - timedelta(hours=preview_hold_hours))
        week = _iso(now - timedelta(days=7))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._reap(cutoff=cutoff, preview_cutoff=preview_cutoff, now=now)
            picked = self._select(limit=limit, week_cutoff=week, category=category)
            claimed: list[Idea] = []
            for idea_id in picked:
                token = str(uuid.uuid4())
                cur = self._conn.execute(
                    "UPDATE ideas SET claimed_by=?, claim_token=?, claimed_at=?, updated_at=?"
                    " WHERE id=? AND claimed_by IS NULL AND status=?",
                    (run_stamp, token, _iso(now), _iso(now), idea_id, IdeaStatus.OPEN.value))
                if cur.rowcount != 1:
                    continue  # raced by another claimer inside the same instant; skip
                self._conn.execute(
                    "INSERT INTO idea_attempts(idea_id, run_stamp, claim_token, claimed_at)"
                    " VALUES (?,?,?,?)", (idea_id, run_stamp, token, _iso(now)))
                claimed.append(self._ideas.get(idea_id))
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return claimed

    def _reap(self, *, cutoff: str, preview_cutoff: str, now: datetime) -> None:
        """Release timed-out claims. TWO clocks, because a `candidate_preview_pass` DELIBERATELY
        keeps its claim so the merge-back drainer's `link` can still spend the fencing token:

        * an ordinary in-flight claim is reaped at `ttl_minutes` (sized against the research run's
          own timeout) with reason `claim_ttl_expired`;
        * a claim whose live attempt already reached `candidate_preview_pass` is EXEMPT from that
          and held for `preview_hold_hours` instead, then abandoned with reason
          `preview_hold_expired`.

        Without the exemption the TTL would strand exactly the successful ideas: the drainer
        drains ONE queue item per 30-minute fire, so a deep backlog routinely outlives the TTL,
        and reaping the claim makes the later `link` fail on its token — leaving an idea whose
        strategy really did reach authority stuck `open` with no FK, forever.

        The preview branch is the ONLY place `candidate_preview_pass` is overwritten outside
        `record_outcome`'s explicit rewrite set, and it is CAS-guarded on that exact prior value.
        `abandoned` does not refute: the idea returns to the pool for a future run."""
        # Prefilter on the LATER of the two cutoffs — a row reapable under either clock always
        # satisfies it, so this narrows the scan without ever hiding a reapable row (correct even
        # if preview_hold_hours is configured shorter than ttl_minutes).
        rows = self._conn.execute(
            "SELECT i.id AS id, i.claim_token AS claim_token, i.claimed_at AS claimed_at,"
            " a.outcome AS outcome FROM ideas i"
            " LEFT JOIN idea_attempts a ON a.idea_id = i.id AND a.claim_token = i.claim_token"
            " WHERE i.claimed_by IS NOT NULL AND i.claimed_at < ?",
            (max(cutoff, preview_cutoff),)).fetchall()
        for r in rows:
            held = r["outcome"] == AttemptOutcome.CANDIDATE_PREVIEW_PASS.value
            # A NULL claimed_at on a claimed row is corrupt; "" sorts before any cutoff, so it
            # reaps (fail toward releasing a claim nobody can account for).
            if (r["claimed_at"] or "") >= (preview_cutoff if held else cutoff):
                continue
            if held:
                self._conn.execute(
                    "UPDATE idea_attempts SET outcome=?, reason=?, outcome_at=? WHERE idea_id=?"
                    " AND claim_token=? AND outcome=?",
                    (AttemptOutcome.ABANDONED.value, "preview_hold_expired", _iso(now), r["id"],
                     r["claim_token"], AttemptOutcome.CANDIDATE_PREVIEW_PASS.value))
            else:
                self._conn.execute(
                    "UPDATE idea_attempts SET outcome=?, reason=?, outcome_at=? WHERE idea_id=?"
                    " AND claim_token=? AND outcome IS NULL",
                    (AttemptOutcome.ABANDONED.value, "claim_ttl_expired", _iso(now), r["id"],
                     r["claim_token"]))
            self._conn.execute(
                "UPDATE ideas SET claimed_by=NULL, claim_token=NULL, claimed_at=NULL,"
                " updated_at=? WHERE id=?", (_iso(now), r["id"]))

    def _select(self, *, limit: int, week_cutoff: str,
                category: str | None = None) -> _list[int]:
        sql = (
            "SELECT i.id, i.category, i.created_at,"
            " (SELECT MIN(CASE ins.obscurity WHEN 'rare' THEN 0 WHEN 'niche' THEN 1"
            "   WHEN 'common' THEN 2 ELSE 3 END) FROM idea_inspirations ins"
            "   WHERE ins.idea_id = i.id) AS obs_rank"
            " FROM ideas i WHERE i.status=? AND i.claimed_by IS NULL"
        )
        params: list[object] = [IdeaStatus.OPEN.value]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        rows = self._conn.execute(sql, params).fetchall()
        recent = {r["category"]: r["n"] for r in self._conn.execute(
            "SELECT i.category AS category, COUNT(*) AS n FROM idea_attempts a"
            " JOIN ideas i ON i.id = a.idea_id WHERE a.claimed_at >= ? GROUP BY i.category",
            (week_cutoff,))}
        by_cat: dict[str | None, list] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        for lst in by_cat.values():
            lst.sort(key=lambda r: (r["obs_rank"] if r["obs_rank"] is not None else 3,
                                    r["created_at"], r["id"]))
        legacy = by_cat.pop(None, [])
        order = sorted(by_cat, key=lambda c: (recent.get(c, 0), c))
        picked: list[int] = []
        while len(picked) < limit and any(by_cat.values()):
            for cat in order:
                if by_cat[cat] and len(picked) < limit:
                    picked.append(by_cat[cat].pop(0)["id"])
        while len(picked) < limit and legacy:
            picked.append(legacy.pop(0)["id"])
        return picked

    # -- outcomes ------------------------------------------------------------------------
    def _check_token(self, idea_id: int, token: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT id, status, claimed_by, claim_token FROM ideas WHERE id=?", (idea_id,)
        ).fetchone()
        if row is None or row["claimed_by"] is None or row["claim_token"] != token:
            raise ClaimTokenMismatch(f"idea {idea_id}: no live claim for that token")
        return row

    def record_outcome(self, idea_id: int, *, token: str, outcome: AttemptOutcome, reason: str,
                       evidence_ref: str | None = None,
                       strategy_name: str | None = None) -> Idea:
        """Write the attempt's outcome once (CAS on the token). Refuting outcomes move the idea
        to REFUTED. CANDIDATE_PREVIEW_PASS keeps the claim (the drainer's `link` releases it);
        every other outcome releases it. The ONLY permitted rewrites are from an attempt already
        recorded as candidate_preview_pass, to one of `PREVIEW_PASS_REWRITES` — integrity_fail
        (the drainer's "authoritative promote failed") or run_error (the drainer's terminal
        merge-back failure, which must NOT refute the idea). Any other rewrite, including
        re-recording candidate_preview_pass, raises ClaimTokenMismatch."""
        if self._conn.in_transaction:
            raise RuntimeError(
                "record_outcome must run at top level, not inside an open transaction")
        now = _iso(datetime.now(UTC))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._check_token(idea_id, token)
            # The permitted rewrites are candidate_preview_pass -> integrity_fail | run_error
            # (both drainer-only; see PREVIEW_PASS_REWRITES). Gating the guard on the NEW value
            # too, not just the old one, closes candidate_preview_pass -> anything: an
            # agent-judgement outcome (walkforward_refuted, holdout_negative, sweep_unstable) can
            # never be forged over preview-pass evidence after the fact.
            if outcome in PREVIEW_PASS_REWRITES:
                guard = "(outcome IS NULL OR outcome = ?)"
                guard_params: tuple[object, ...] = (AttemptOutcome.CANDIDATE_PREVIEW_PASS.value,)
            else:
                guard = "outcome IS NULL"
                guard_params = ()
            cur = self._conn.execute(
                "UPDATE idea_attempts SET outcome=?, reason=?, evidence_ref=?, strategy_name=?,"
                f" outcome_at=? WHERE idea_id=? AND claim_token=? AND {guard}",
                (outcome.value, reason[:300], evidence_ref, strategy_name, now, idea_id, token,
                 *guard_params))
            if cur.rowcount != 1:
                raise ClaimTokenMismatch(f"idea {idea_id}: attempt already has an outcome")
            sets = ["updated_at=?"]
            params: list[object] = [now]
            if outcome in REFUTING_OUTCOMES and row["status"] == IdeaStatus.OPEN.value:
                sets.append("status=?")
                params.append(IdeaStatus.REFUTED.value)
            if outcome is not AttemptOutcome.CANDIDATE_PREVIEW_PASS:
                sets.append("claimed_by=NULL, claim_token=NULL, claimed_at=NULL")
            params.append(idea_id)
            self._conn.execute(f"UPDATE ideas SET {', '.join(sets)} WHERE id=?", params)
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return self._ideas.get(idea_id)

    def link(self, idea_id: int, *, token: str, strategy_id: int, strategy_name: str) -> Idea:
        """Merge-back succeeded: AUTHORED + FK, attempt outcome -> promoted_candidate, claim
        released. The attempt rewrite here (candidate_preview_pass -> promoted_candidate) is the
        drainer's normal happy path, not the record_outcome rewrite exception."""
        if self._conn.in_transaction:
            raise RuntimeError("link must run at top level, not inside an open transaction")
        now = _iso(datetime.now(UTC))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._check_token(idea_id, token)
            cur = self._conn.execute(
                "UPDATE idea_attempts SET outcome=?, strategy_name=?, outcome_at=?"
                " WHERE idea_id=? AND claim_token=? AND (outcome = ? OR outcome IS NULL)",
                (AttemptOutcome.PROMOTED_CANDIDATE.value, strategy_name, now, idea_id, token,
                 AttemptOutcome.CANDIDATE_PREVIEW_PASS.value))
            if cur.rowcount != 1:
                # A live claim (token checked above) whose attempt row already carries a terminal
                # outcome other than candidate_preview_pass -- mirrors record_outcome's CAS check
                # so the idea is never flipped to AUTHORED while the attempt ledger diverges.
                raise ClaimTokenMismatch(
                    f"idea {idea_id}: attempt outcome is not candidate_preview_pass or NULL")
            self._conn.execute(
                "UPDATE ideas SET status=?, authored_strategy_id=?, claimed_by=NULL,"
                " claim_token=NULL, claimed_at=NULL, updated_at=? WHERE id=?",
                (IdeaStatus.AUTHORED.value, strategy_id, now, idea_id))
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return self._ideas.get(idea_id)

    # -- reads ---------------------------------------------------------------------------
    def attempts_of(self, idea_id: int) -> _list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM idea_attempts WHERE idea_id=? ORDER BY id", (idea_id,))
        return [dict(r) for r in rows]

    def depth(self, *, runs_per_day: int, hypotheses_per_run: int, floor_days: int,
              ceiling_days: int) -> dict:
        per_day = runs_per_day * hypotheses_per_run
        counts = {"open_unclaimed": 0, "claimed": 0, "needs_data": 0}
        for r in self._conn.execute(
                "SELECT status, claimed_by IS NOT NULL AS claimed, COUNT(*) AS n FROM ideas"
                " GROUP BY status, claimed"):
            if r["status"] == IdeaStatus.OPEN.value:
                counts["claimed" if r["claimed"] else "open_unclaimed"] += r["n"]
            elif r["status"] == IdeaStatus.NEEDS_DATA.value:
                counts["needs_data"] += r["n"]
        refill, ceiling = per_day * floor_days, per_day * ceiling_days
        return {**counts, "refill_at": refill, "ceiling": ceiling,
                "below_refill": counts["open_unclaimed"] < refill,
                "inputs": {"runs_per_day": runs_per_day, "hypotheses_per_run": hypotheses_per_run,
                           "floor_days": floor_days, "ceiling_days": ceiling_days}}
