import sqlite3

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
from algua.registry.idea_attempts import IdeaAttemptsRepository
from algua.registry.idea_import import (
    import_critic_rejections,
    import_ideas,
    reclassify,
    refuted_with_reasons,
)
from algua.registry.ideas import IdeaRepository, InspirationLink
from algua.registry.negative_results import list_negative_results


def _db(path):
    conn = connect(path)
    migrate(conn)
    return conn


def _scratch_from(auth_path, scratch_path):
    src, dst = sqlite3.connect(auth_path), sqlite3.connect(scratch_path)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return _db(scratch_path)


def _add(repo, title, hyp, **kw):
    base = dict(family=None, tags=[], source_type=SourceType.INSPIRATION, source_ref=None,
                source_date=None, source_note=None, required_data=[DataCapability.OHLCV],
                status=IdeaStatus.OPEN, category="momentum", market=Market.US_EQUITIES,
                horizon=Horizon.DAILY, falsification="refuted if y", created_by_run="t",
                inspirations=[InspirationLink("i-1", "blog/x", Obscurity.NICHE)])
    base.update(kw)
    return repo.add(title=title, hypothesis=hyp, **base)


def test_import_copies_new_scratch_rows_with_links_and_recheck(tmp_path):
    auth = _db(tmp_path / "auth.db")
    _add(IdeaRepository(auth), "seeded idea words here", "seeded hypothesis words here")
    seeded_max = auth.execute("SELECT MAX(id) FROM ideas").fetchone()[0]
    scratch = _scratch_from(tmp_path / "auth.db", tmp_path / "scratch.db")
    srepo = IdeaRepository(scratch)
    # alpha/beta are lexically DISTINCT from each other and from the seeded idea (verified
    # against algua.research.idea_dedup.is_collision — jaccard 0.0 for every pair below) so this
    # test isolates the "new content imports, a duplicate of PRE-EXISTING content is rejected"
    # behavior from same-batch sibling dedup (covered separately, see
    # test_import_dedups_near_duplicate_siblings_within_one_call).
    _add(srepo, "overnight gap fade after quiet opens", "fade opens that gapped quietly overnight")
    _add(srepo, "turnover decline predicts drift", "declining turnover predicts price drift",
         market=Market.CRYPTO)  # will park on import: market unsupported
    _add(srepo, "seeded idea words here again", "seeded hypothesis words here again")  # dup
    res = import_ideas(auth, scratch, run_stamp="leap-1", max_new=10, ceiling=100,
                       seeded_max_id=seeded_max)
    assert len(res["imported"]) == 2
    assert [s["reason"] for s in res["skipped"]] == ["dedup_collision"]
    arepo = IdeaRepository(auth)
    ideas = {i.title: i for i in arepo.list()}
    assert ideas["turnover decline predicts drift"].status is IdeaStatus.NEEDS_DATA
    assert ideas["turnover decline predicts drift"].parked_reason == "market:crypto"
    assert arepo.inspirations_of(
        ideas["overnight gap fade after quiet opens"].id)[0].venue == "blog/x"


def test_import_dedups_near_duplicate_siblings_within_one_call(tmp_path):
    """Two near-duplicate proposals from the SAME leap batch must dedup against each other, not
    just against pre-existing authority content: the collision check re-runs against everything
    already in authority, including what this same import already inserted (spec §6). "brand new
    leap idea alpha/beta" is a real near-duplicate pair under the token-Jaccard dedup (0.778 >=
    the 0.6 threshold), verified via algua.research.idea_dedup.is_collision."""
    auth = _db(tmp_path / "auth.db")
    scratch = _scratch_from(tmp_path / "auth.db", tmp_path / "scratch.db")
    srepo = IdeaRepository(scratch)
    _add(srepo, "brand new leap idea alpha", "alpha mechanism text distinct")
    _add(srepo, "brand new leap idea beta", "beta mechanism text distinct")
    res = import_ideas(auth, scratch, run_stamp="leap-2", max_new=10, ceiling=100,
                       seeded_max_id=0)
    assert len(res["imported"]) == 1
    assert [s["reason"] for s in res["skipped"]] == ["dedup_collision"]
    arepo = IdeaRepository(auth)
    assert [i.title for i in arepo.list()] == ["brand new leap idea alpha"]


def test_import_ideas_refuses_inside_open_transaction(tmp_path):
    """import_ideas owns its own per-row BEGIN IMMEDIATE/commit; called from inside a caller's
    already-open transaction it must refuse rather than silently nest (matching the guard on
    IdeaAttemptsRepository.claim/record_outcome/link)."""
    auth = _db(tmp_path / "auth.db")
    scratch = _scratch_from(tmp_path / "auth.db", tmp_path / "scratch.db")
    auth.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError):
            import_ideas(auth, scratch, run_stamp="l", max_new=1, ceiling=100, seeded_max_id=0)
    finally:
        auth.rollback()


def test_import_respects_max_and_ceiling(tmp_path):
    auth = _db(tmp_path / "auth.db")
    scratch = _scratch_from(tmp_path / "auth.db", tmp_path / "scratch.db")
    srepo = IdeaRepository(scratch)
    # Five LEXICALLY DISTINCT ideas (no pair collides under is_collision) so this test isolates
    # max_new/ceiling counting from dedup behavior.
    distinct_ideas = [
        ("overnight gap fade after quiet opens", "fade opens that gapped quietly overnight"),
        ("turnover decline predicts drift", "declining turnover predicts price drift"),
        ("insider cluster buying signals reversal",
         "clustered insider buying predicts reversal moves"),
        ("sector rotation breadth thrust indicator",
         "breadth thrust flags sector rotation shifts"),
        ("earnings drift persists past surprise", "post surprise earnings drift continues weeks"),
    ]
    for title, hyp in distinct_ideas:
        _add(srepo, title, hyp)
    res = import_ideas(auth, scratch, run_stamp="l", max_new=3, ceiling=100, seeded_max_id=0)
    assert len(res["imported"]) == 3
    res2 = import_ideas(auth, scratch, run_stamp="l", max_new=10, ceiling=4, seeded_max_id=0)
    assert len(res2["imported"]) == 1 and res2["skipped"][-1]["reason"] == "ceiling"


def test_import_critic_rejections_db_only_with_quota(tmp_path):
    auth = _db(tmp_path / "auth.db")
    rows = [{"title": f"rej {i}", "hypothesis": f"h {i}", "reason_kind": "beta_in_disguise"}
            for i in range(5)]
    n = import_critic_rejections(auth, rows, run_stamp="l", max_rows=3)
    assert n == 3
    got = list_negative_results(auth, limit=10)
    assert all(r["source"] == "auto:leap_critic" for r in got) and len(got) == 3
    assert got[0]["verdict"] == "CRITIC:beta_in_disguise"


def test_refuted_with_reasons_joins_attempt_reason(tmp_path):
    auth = _db(tmp_path / "auth.db")
    repo, att = IdeaRepository(auth), IdeaAttemptsRepository(auth)
    idea = _add(repo, "will be refuted idea words", "refuted hypothesis words")
    (c,) = att.claim(run_stamp="r", limit=1, ttl_minutes=180)
    att.record_outcome(idea.id, token=c.claim_token,
                       outcome=AttemptOutcome.HOLDOUT_NEGATIVE, reason="holdout sharpe -0.3")
    (row,) = refuted_with_reasons(auth, limit=10)
    assert row["id"] == idea.id and row["reason"] == "holdout sharpe -0.3"
    assert row["outcome"] == "holdout_negative"


def test_reclassify_reopens_when_market_becomes_supported(tmp_path, monkeypatch):
    auth = _db(tmp_path / "auth.db")
    repo = IdeaRepository(auth)
    parked = _add(repo, "crypto idea words here", "crypto hypothesis words here",
                  market=Market.CRYPTO, status=IdeaStatus.NEEDS_DATA, parked_reason="market:crypto")
    import algua.registry.idea_import as mod
    monkeypatch.setattr(mod, "supported_markets", lambda: frozenset({Market.CRYPTO, Market.ANY}))
    out = reclassify(auth)
    assert out["reopened"] == [parked.id]
    assert repo.get(parked.id).status is IdeaStatus.OPEN
