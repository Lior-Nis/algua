# tests/test_idea_repository.py
import pytest

from algua.contracts.idea import DataCapability, IdeaStatus, SourceType
from algua.contracts.registry_metadata import HypothesisStatus
from algua.registry.db import connect, migrate
from algua.registry.ideas import IdeaNotFound, IdeaRepository
from algua.registry.store import SqliteStrategyRepository


def _conns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return IdeaRepository(conn), SqliteStrategyRepository(conn)


def _add(repo, *, title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
         family="vol", status=IdeaStatus.OPEN, required_data=None):
    return repo.add(
        title=title, hypothesis=hypothesis, family=family, tags=["factor"],
        source_type=SourceType.PAPER, source_ref="http://x", source_date="2020-01-01",
        source_note=None, required_data=required_data or [DataCapability.OHLCV], status=status)


def test_add_get_roundtrip(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    assert idea.id == 1
    fetched = repo.get(idea.id)
    assert fetched.title == "low vol anomaly"
    assert fetched.required_data == [DataCapability.OHLCV]
    assert fetched.status is IdeaStatus.OPEN
    assert fetched.signature  # computed + stored


def test_get_missing_raises(tmp_path):
    repo, _ = _conns(tmp_path)
    with pytest.raises(IdeaNotFound):
        repo.get(999)


def test_add_rejects_authored_status(tmp_path):
    """add() always inserts a NULL strategy link, so creating an AUTHORED idea here would
    violate the AUTHORED<->authored_strategy_id invariant set_status enforces (#267)."""
    repo, _ = _conns(tmp_path)
    with pytest.raises(ValueError, match="cannot create an AUTHORED idea"):
        _add(repo, status=IdeaStatus.AUTHORED)


def test_list_filters_by_status_and_family(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="a", family="vol")
    _add(repo, title="b", family="mom", status=IdeaStatus.NEEDS_DATA,
         required_data=[DataCapability.FORM_13F])
    assert [i.title for i in repo.list(family="vol")] == ["a"]
    assert [i.title for i in repo.list(status=IdeaStatus.NEEDS_DATA)] == ["b"]


def test_find_collisions_same_family(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
         family="vol")
    hits = repo.find_collisions(
        title="the low vol anomaly",
        hypothesis="low vol stocks outperform on a risk adjusted basis", family="vol")
    assert len(hits) == 1
    assert hits[0].effective_status is IdeaStatus.OPEN


def test_find_collisions_ignores_discarded(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    repo.set_status(idea.id, to=IdeaStatus.DISCARDED)
    assert repo.find_collisions(
        title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
        family="vol") == []


def test_refuted_strategy_blocks_via_live_join(tmp_path):
    repo, strat = _conns(tmp_path)
    idea = _add(repo)
    s = strat.add("lowvol_v1", family="vol")
    repo.set_status(idea.id, to=IdeaStatus.AUTHORED, authored_strategy_id=s.id)
    # The authored idea's strategy gets refuted (registry set, elsewhere).
    strat.update_metadata("lowvol_v1", hypothesis_status=HypothesisStatus.REFUTED)
    hits = repo.find_collisions(
        title="low vol anomaly redux",
        hypothesis="low vol names outperform on risk adjusted returns", family="vol")
    assert len(hits) == 1
    assert hits[0].effective_status is IdeaStatus.REFUTED  # downgraded by the join


def test_set_status_rejects_illegal(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    repo.set_status(idea.id, to=IdeaStatus.DISCARDED)
    with pytest.raises(ValueError, match="illegal idea status change"):
        repo.set_status(idea.id, to=IdeaStatus.OPEN)


def test_set_status_rejects_strategy_link_on_non_authored(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    with pytest.raises(ValueError, match="authored_strategy_id is only valid"):
        repo.set_status(idea.id, to=IdeaStatus.DISCARDED, authored_strategy_id=5)


def test_windowed_counts_by_status(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="a")
    _add(repo, title="b", status=IdeaStatus.NEEDS_DATA,
         required_data=[DataCapability.FORM_13F])
    counts = repo.windowed_idea_counts(90)
    assert counts["open"] == 1
    assert counts["needs_data"] == 1
    assert counts["total"] == 2
