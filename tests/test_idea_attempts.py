# tests/test_idea_attempts.py
from datetime import UTC, datetime, timedelta

import pytest

from algua.contracts.idea import (
    AttemptOutcome,
    DataCapability,
    Horizon,
    IdeaStatus,
    Market,
    Obscurity,
    SourceType,
)
from algua.registry.db import connect, migrate
from algua.registry.idea_attempts import ClaimTokenMismatch, IdeaAttemptsRepository
from algua.registry.ideas import IdeaRepository, InspirationLink
from algua.registry.store import SqliteStrategyRepository

DEPTH = dict(runs_per_day=12, hypotheses_per_run=3, floor_days=2, ceiling_days=7)


def _setup(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn, IdeaRepository(conn), IdeaAttemptsRepository(conn)


def _seed(repo, *, n, category, obscurity=Obscurity.COMMON, status=IdeaStatus.OPEN):
    out = []
    for i in range(n):
        out.append(repo.add(
            title=f"{category} idea number {i} unique words {category}{i}",
            hypothesis=f"{category} hypothesis {i} distinct text {i}", family=None, tags=[],
            source_type=SourceType.INSPIRATION, source_ref=None, source_date=None,
            source_note=None, required_data=[DataCapability.OHLCV], status=status,
            category=category, market=Market.US_EQUITIES, horizon=Horizon.DAILY,
            falsification="refuted if x", created_by_run="t",
            inspirations=[InspirationLink(f"insp-{category}-{i}", "blog/x", obscurity)]))
    return out


def test_claim_round_robins_categories_then_prefers_rare(tmp_path):
    _, repo, att = _setup(tmp_path)
    _seed(repo, n=3, category="momentum")
    rare = _seed(repo, n=1, category="seasonality", obscurity=Obscurity.RARE)
    _seed(repo, n=1, category="seasonality", obscurity=Obscurity.CANON)
    claimed = att.claim(run_stamp="r1", limit=3, ttl_minutes=180)
    cats = [c.category for c in claimed]
    assert sorted(cats) == ["momentum", "momentum", "seasonality"]
    assert rare[0].id in {c.id for c in claimed}  # rare beats canon inside seasonality
    assert all(c.claimed_by == "r1" and c.claim_token for c in claimed)


def test_claim_skips_claimed_and_non_open(tmp_path):
    _, repo, att = _setup(tmp_path)
    _seed(repo, n=2, category="momentum")
    _seed(repo, n=1, category="momentum", status=IdeaStatus.NEEDS_DATA)
    first = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    second = att.claim(run_stamp="r2", limit=5, ttl_minutes=180)
    assert len(first) == 1 and len(second) == 1 and first[0].id != second[0].id


def test_claim_reaps_expired_claims_as_abandoned(tmp_path):
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    old = datetime.now(UTC) - timedelta(minutes=500)
    att.claim(run_stamp="r1", limit=1, ttl_minutes=180, now=old)
    reclaimed = att.claim(run_stamp="r2", limit=1, ttl_minutes=180)
    assert [i.id for i in reclaimed] == [idea.id]
    outcomes = [a["outcome"] for a in att.attempts_of(idea.id)]
    assert outcomes == ["abandoned", None]  # oldest first: reaped, then the live claim


def test_legacy_null_category_only_when_nothing_else(tmp_path):
    _, repo, att = _setup(tmp_path)
    legacy = repo.add(title="legacy row words", hypothesis="legacy hypothesis words",
                      family=None, tags=[], source_type=SourceType.MANUAL, source_ref=None,
                      source_date=None, source_note=None,
                      required_data=[DataCapability.OHLCV], status=IdeaStatus.OPEN)
    fresh = _seed(repo, n=1, category="momentum")
    assert [i.id for i in att.claim(run_stamp="r1", limit=1, ttl_minutes=180)] == [fresh[0].id]
    assert [i.id for i in att.claim(run_stamp="r2", limit=1, ttl_minutes=180)] == [legacy.id]


def test_record_outcome_requires_token_and_writes_once(tmp_path):
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    with pytest.raises(ClaimTokenMismatch):
        att.record_outcome(idea.id, token="wrong", outcome=AttemptOutcome.RUN_ERROR, reason="x")
    done = att.record_outcome(idea.id, token=c.claim_token,
                              outcome=AttemptOutcome.WALKFORWARD_REFUTED, reason="min sharpe<0")
    assert done.status is IdeaStatus.REFUTED and done.claimed_by is None
    with pytest.raises(ClaimTokenMismatch):  # claim released: token no longer valid
        att.record_outcome(idea.id, token=c.claim_token, outcome=AttemptOutcome.RUN_ERROR,
                           reason="again")
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "walkforward_refuted" and row["outcome_at"]


def test_preview_pass_keeps_open_and_link_then_promoted(tmp_path):
    conn, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    after = att.record_outcome(idea.id, token=c.claim_token,
                               outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="ok")
    assert after.status is IdeaStatus.OPEN and after.claimed_by == "r1"  # claim HELD
    strat = SqliteStrategyRepository(conn).add("strat_a")
    linked = att.link(idea.id, token=c.claim_token, strategy_id=strat.id, strategy_name="strat_a")
    assert linked.status is IdeaStatus.AUTHORED and linked.authored_strategy_id == strat.id
    assert linked.claimed_by is None
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "promoted_candidate" and row["strategy_name"] == "strat_a"


def test_depth_converts_days_to_counts(tmp_path):
    _, repo, att = _setup(tmp_path)
    _seed(repo, n=4, category="momentum")
    att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    d = att.depth(**DEPTH)
    assert d["open_unclaimed"] == 3 and d["claimed"] == 1
    assert d["refill_at"] == 72 and d["ceiling"] == 252 and d["below_refill"] is True
    assert d["inputs"] == DEPTH


def test_claim_with_category_filters_to_that_category(tmp_path):
    _, repo, att = _setup(tmp_path)
    _seed(repo, n=2, category="momentum")
    seasonality = _seed(repo, n=2, category="seasonality")
    claimed = att.claim(run_stamp="r1", limit=5, ttl_minutes=180, category="seasonality")
    assert sorted(c.id for c in claimed) == sorted(i.id for i in seasonality)
    assert all(c.category == "seasonality" for c in claimed)


def test_record_outcome_allows_preview_pass_to_integrity_fail_rewrite(tmp_path):
    """The merge-back drainer's 'authoritative promote failed' path: an attempt already recorded
    as candidate_preview_pass may be rewritten to integrity_fail (the ONE permitted rewrite)."""
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="preview ok")
    rewritten = att.record_outcome(idea.id, token=c.claim_token,
                                   outcome=AttemptOutcome.INTEGRITY_FAIL,
                                   reason="authoritative promote failed")
    assert rewritten.status is IdeaStatus.REFUTED and rewritten.claimed_by is None
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "integrity_fail"


def test_record_outcome_allows_preview_pass_to_run_error_rewrite_without_refuting(tmp_path):
    """The drainer's TERMINAL merge-back failure path (diff policy rejected, retry budget
    exhausted): the merge-back died for an INFRASTRUCTURE reason, which says nothing about the
    hypothesis — so the claim is released but the idea must stay OPEN, not REFUTED, and must be
    re-claimable by a later run."""
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="preview ok")
    rewritten = att.record_outcome(idea.id, token=c.claim_token,
                                   outcome=AttemptOutcome.RUN_ERROR,
                                   reason="mergeback_terminal_failed:diff_policy_rejected")
    assert rewritten.status is IdeaStatus.OPEN      # run_error does not refute
    assert rewritten.claimed_by is None             # ...but the claim IS released
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "run_error"
    assert [i.id for i in att.claim(run_stamp="r2", limit=1, ttl_minutes=180)] == [idea.id]


@pytest.mark.parametrize("outcome", [
    AttemptOutcome.CANDIDATE_PREVIEW_PASS,   # re-recording the same outcome
    AttemptOutcome.WALKFORWARD_REFUTED,      # an AGENT-judgement outcome, forged after the fact
    AttemptOutcome.HOLDOUT_NEGATIVE,
    AttemptOutcome.SWEEP_UNSTABLE,
    AttemptOutcome.ABANDONED,
])
def test_record_outcome_rejects_preview_pass_rewrite_to_other_outcomes(tmp_path, outcome):
    """Only PREVIEW_PASS_REWRITES (integrity_fail | run_error, both drainer-only) may overwrite a
    candidate_preview_pass. Every other new outcome must still raise and leave the attempt row
    exactly as candidate_preview_pass recorded it — an agent's judgement of the hypothesis can
    never be rewritten after the preview."""
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="preview ok")
    (before,) = att.attempts_of(idea.id)

    with pytest.raises(ClaimTokenMismatch):
        att.record_outcome(idea.id, token=c.claim_token, outcome=outcome,
                           reason="should not land")

    (after,) = att.attempts_of(idea.id)
    assert after == before  # attempt row untouched by the rejected rewrite
    idea_after = repo.get(idea.id)
    assert idea_after.status is IdeaStatus.OPEN and idea_after.claimed_by == "r1"


# --- the reaper's two clocks: an ordinary TTL, and the preview HOLD ------------------------------


def _preview_claim(att, repo, idea, *, at, hold_hours=72):
    """Claim `idea` at instant `at` and record candidate_preview_pass on it (the state that
    deliberately keeps a claim held for the merge-back drainer's `link`)."""
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180, now=at,
                     preview_hold_hours=hold_hours)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="preview ok")
    return c


def test_preview_pass_claim_past_the_ttl_but_inside_the_hold_is_not_reaped(tmp_path):
    """The whole point of the exemption: the drainer drains one item per 30-minute fire, so a
    backlogged candidate is routinely older than the 180-minute TTL. Reaping it would break the
    later `link`'s fencing token and strand an idea whose strategy did reach authority."""
    _, repo, att = _setup(tmp_path)
    (idea, other) = _seed(repo, n=2, category="momentum")
    old = datetime.now(UTC) - timedelta(minutes=500)   # way past the 180-minute TTL...
    _preview_claim(att, repo, idea, at=old)            # ...but well inside a 72-hour hold

    claimed = att.claim(run_stamp="r2", limit=5, ttl_minutes=180, preview_hold_hours=72)

    assert [i.id for i in claimed] == [other.id]       # the held idea was NOT re-offered
    held = repo.get(idea.id)
    assert held.claimed_by == "r1" and held.claim_token is not None
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "candidate_preview_pass"  # evidence intact


def test_preview_pass_claim_past_the_hold_is_reaped_as_preview_hold_expired(tmp_path):
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    ancient = datetime.now(UTC) - timedelta(hours=100)  # past a 72-hour hold
    _preview_claim(att, repo, idea, at=ancient)

    claimed = att.claim(run_stamp="r2", limit=1, ttl_minutes=180, preview_hold_hours=72)

    assert [i.id for i in claimed] == [idea.id]        # released back into the pool
    (reaped, _live) = att.attempts_of(idea.id)
    assert reaped["outcome"] == "abandoned"
    assert reaped["reason"] == "preview_hold_expired"
    assert repo.get(idea.id).status is IdeaStatus.OPEN  # abandoning never refutes


def test_an_ordinary_in_flight_claim_still_reaps_on_the_short_ttl(tmp_path):
    """The exemption is keyed on the ATTEMPT's outcome, not on age: a claim with no outcome yet
    (a crashed run) must still be reaped at the TTL, not held for 72 hours."""
    _, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    old = datetime.now(UTC) - timedelta(minutes=500)
    att.claim(run_stamp="r1", limit=1, ttl_minutes=180, now=old, preview_hold_hours=72)

    claimed = att.claim(run_stamp="r2", limit=1, ttl_minutes=180, preview_hold_hours=72)

    assert [i.id for i in claimed] == [idea.id]
    (reaped, _live) = att.attempts_of(idea.id)
    assert (reaped["outcome"], reaped["reason"]) == ("abandoned", "claim_ttl_expired")


def test_link_rejects_when_attempt_outcome_is_not_preview_pass_or_null(tmp_path):
    """A live claim (token still valid) whose attempt row already carries a terminal outcome
    other than candidate_preview_pass/NULL must not flip the idea to AUTHORED — the
    idea_attempts UPDATE's rowcount is checked, mirroring record_outcome's CAS."""
    conn, repo, att = _setup(tmp_path)
    (idea,) = _seed(repo, n=1, category="momentum")
    (c,) = att.claim(run_stamp="r1", limit=1, ttl_minutes=180)
    # Simulate a terminal, non-preview-pass outcome landing on this claim's attempt row while the
    # claim itself is still live -- bypasses record_outcome (which would release the claim) so
    # this exercises link()'s own rowcount guard rather than _check_token's claimed_by check.
    conn.execute("UPDATE idea_attempts SET outcome=? WHERE idea_id=? AND claim_token=?",
                 (AttemptOutcome.RUN_ERROR.value, idea.id, c.claim_token))
    conn.commit()

    strat = SqliteStrategyRepository(conn).add("strat_b")
    with pytest.raises(ClaimTokenMismatch):
        att.link(idea.id, token=c.claim_token, strategy_id=strat.id, strategy_name="strat_b")

    idea_after = repo.get(idea.id)
    assert idea_after.status is IdeaStatus.OPEN and idea_after.authored_strategy_id is None
    (row,) = att.attempts_of(idea.id)
    assert row["outcome"] == "run_error"
