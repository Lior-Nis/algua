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
from algua.registry.idea_attempts import IdeaAttemptsRepository
from algua.registry.idea_scorecard import scorecard
from algua.registry.ideas import IdeaRepository, InspirationLink


def test_scorecard_groups_by_venue_and_reports_rates_only_at_n5(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo, att = IdeaRepository(conn), IdeaAttemptsRepository(conn)
    for i in range(6):
        repo.add(title=f"venue idea {i} distinct words", hypothesis=f"mech {i} distinct words",
                 family=None, tags=[], source_type=SourceType.INSPIRATION, source_ref=None,
                 source_date=None, source_note=None, required_data=[DataCapability.OHLCV],
                 status=IdeaStatus.OPEN, category="momentum", market=Market.US_EQUITIES,
                 horizon=Horizon.DAILY, falsification="f", created_by_run="t",
                 inspirations=[InspirationLink(f"n{i}", "reddit/algotrading", Obscurity.NICHE)])
    claimed = att.claim(run_stamp="r", limit=6, ttl_minutes=180)
    outcomes = [AttemptOutcome.INTEGRITY_FAIL] * 2 + [AttemptOutcome.WALKFORWARD_REFUTED] * 3 \
        + [AttemptOutcome.CANDIDATE_PREVIEW_PASS]
    for idea, oc in zip(claimed, outcomes, strict=True):
        att.record_outcome(idea.id, token=idea.claim_token, outcome=oc, reason="x")
    sc = scorecard(conn, days=90)
    venue = sc["by_venue"]["reddit/algotrading"]
    assert venue["n"] == 6 and venue["outcomes"]["integrity_fail"] == 2
    assert venue["integrity_yield"] == 4 / 6 and venue["walkforward_yield"] == 1 / 6
    assert venue["survival_yield"] == 0.0
    assert sc["by_category"]["momentum"]["n"] == 6
    assert sc["by_obscurity"]["niche"]["n"] == 6
    assert "n1" in sc["by_inspiration"]


def test_scorecard_withholds_rates_below_n5(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo, att = IdeaRepository(conn), IdeaAttemptsRepository(conn)
    repo.add(title="one idea distinct words", hypothesis="one mech distinct words", family=None,
             tags=[], source_type=SourceType.INSPIRATION, source_ref=None, source_date=None,
             source_note=None, required_data=[DataCapability.OHLCV], status=IdeaStatus.OPEN,
             category="momentum", market=Market.US_EQUITIES, horizon=Horizon.DAILY,
             falsification="f", created_by_run="t",
             inspirations=[InspirationLink("n", "blog/y", Obscurity.RARE)])
    (c,) = att.claim(run_stamp="r", limit=1, ttl_minutes=180)
    att.record_outcome(c.id, token=c.claim_token, outcome=AttemptOutcome.RUN_ERROR, reason="x")
    v = scorecard(conn, days=90)["by_venue"]["blog/y"]
    assert v["n"] == 1 and v["integrity_yield"] is None


def test_scorecard_counts_two_attempts_on_the_same_idea_separately(tmp_path):
    """A second attempt on the SAME idea (re-claimed after a run_error released it) must count
    as its own attempt, not collapse into the first — the per-(group,key) dedup marker must key
    on the attempt's OWN id, not (idea_id, outcome), which two same-outcome attempts share."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo, att = IdeaRepository(conn), IdeaAttemptsRepository(conn)
    repo.add(title="retried idea distinct words", hypothesis="retried mech distinct words",
             family=None, tags=[], source_type=SourceType.INSPIRATION, source_ref=None,
             source_date=None, source_note=None, required_data=[DataCapability.OHLCV],
             status=IdeaStatus.OPEN, category="momentum", market=Market.US_EQUITIES,
             horizon=Horizon.DAILY, falsification="f", created_by_run="t",
             inspirations=[InspirationLink("n", "blog/retry", Obscurity.COMMON)])
    (c1,) = att.claim(run_stamp="r", limit=1, ttl_minutes=180)
    att.record_outcome(c1.id, token=c1.claim_token, outcome=AttemptOutcome.RUN_ERROR, reason="x")
    (c2,) = att.claim(run_stamp="r", limit=1, ttl_minutes=180)
    assert c2.id == c1.id  # run_error released the claim; the only OPEN idea is re-claimed
    att.record_outcome(c2.id, token=c2.claim_token, outcome=AttemptOutcome.RUN_ERROR, reason="x")
    v = scorecard(conn, days=90)["by_venue"]["blog/retry"]
    assert v["n"] == 2
    assert v["outcomes"]["run_error"] == 2


def test_scorecard_survival_counts_promote_and_forward_tested_stage_once(tmp_path):
    """An idea that was directly promoted AND whose strategy later reached forward_tested must
    count as ONE survival, not two — survival is a union over the attempt, not a sum of the
    'promoted_candidate' outcome count and the 'forward_survivor' stage count."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo, att = IdeaRepository(conn), IdeaAttemptsRepository(conn)
    idea = repo.add(
        title="promoted idea distinct words", hypothesis="promoted mech distinct words",
        family=None, tags=[], source_type=SourceType.INSPIRATION, source_ref=None,
        source_date=None, source_note=None, required_data=[DataCapability.OHLCV],
        status=IdeaStatus.OPEN, category="momentum", market=Market.US_EQUITIES,
        horizon=Horizon.DAILY, falsification="f", created_by_run="t",
        inspirations=[InspirationLink("n", "blog/survivor", Obscurity.RARE)])
    (c,) = att.claim(run_stamp="r", limit=1, ttl_minutes=180)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.CANDIDATE_PREVIEW_PASS, reason="x")
    now = "2026-01-01T00:00:00+00:00"
    cur = conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
        ("promoted_strat", "forward_tested", now, now))
    strategy_id = cur.lastrowid
    conn.commit()
    att.link(idea.id, token=c.claim_token, strategy_id=strategy_id, strategy_name="promoted_strat")

    v = scorecard(conn, days=90)["by_venue"]["blog/survivor"]
    assert v["n"] == 1
    assert v["outcomes"]["promoted_candidate"] == 1
    assert v["survivals"] == 1
