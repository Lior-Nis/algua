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
