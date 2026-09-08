import sqlite3

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
    _add(srepo, "brand new leap idea alpha", "alpha mechanism text distinct")
    _add(srepo, "brand new leap idea beta", "beta mechanism text distinct",
         market=Market.CRYPTO)  # will park on import: market unsupported
    _add(srepo, "seeded idea words here again", "seeded hypothesis words here again")  # dup
    res = import_ideas(auth, scratch, run_stamp="leap-1", max_new=10, ceiling=100,
                       seeded_max_id=seeded_max)
    assert len(res["imported"]) == 2
    assert [s["reason"] for s in res["skipped"]] == ["dedup_collision"]
    arepo = IdeaRepository(auth)
    ideas = {i.title: i for i in arepo.list()}
    assert ideas["brand new leap idea beta"].status is IdeaStatus.NEEDS_DATA
    assert ideas["brand new leap idea beta"].parked_reason == "market:crypto"
    assert arepo.inspirations_of(ideas["brand new leap idea alpha"].id)[0].venue == "blog/x"


def test_import_respects_max_and_ceiling(tmp_path):
    auth = _db(tmp_path / "auth.db")
    scratch = _scratch_from(tmp_path / "auth.db", tmp_path / "scratch.db")
    srepo = IdeaRepository(scratch)
    for i in range(5):
        _add(srepo, f"leap idea number {i} distinct", f"mechanism {i} distinct words")
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
