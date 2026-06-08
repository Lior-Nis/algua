import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.db import connect, migrate
from algua.registry.repository import StrategyExists
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def test_record_exposes_metadata_defaults(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("plain")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []
    assert rec.family is None
    assert rec.derived_from is None
    assert rec.description is None


def test_null_metadata_columns_read_as_defaults(tmp_path):
    # A row written before the columns existed (all NULL) must read as the defaults.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy', 'idea', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    rec = SqliteStrategyRepository(conn).get("legacy")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []


@pytest.fixture()
def repo(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return SqliteStrategyRepository(c)


def _transition(repo, name, to, actor, reason=None):
    return transition_strategy(repo, name, to, actor, reason)


def test_add_creates_idea_with_initial_transition(repo):
    rec = repo.add("alpha")
    assert rec.stage is Stage.IDEA
    transitions = repo.list_transitions("alpha")
    assert len(transitions) == 1
    assert transitions[0]["to_stage"] == "idea"
    assert transitions[0]["from_stage"] is None
    assert transitions[0]["actor"] == "system"


def test_duplicate_name_raises(repo):
    repo.add("alpha")
    with pytest.raises(StrategyExists):
        repo.add("alpha")


def test_legal_transition_updates_stage_and_history(repo):
    repo.add("alpha")
    rec = _transition(repo, "alpha", Stage.BACKTESTED, Actor.AGENT, "ran backtest")
    assert rec.stage is Stage.BACKTESTED
    assert len(repo.list_transitions("alpha")) == 2


def test_transition_records_true_from_stage(repo):
    repo.add("alpha")
    _transition(repo, "alpha", Stage.BACKTESTED, Actor.AGENT)
    last = repo.list_transitions("alpha")[-1]
    assert last["from_stage"] == "idea"
    assert last["to_stage"] == "backtested"


def test_transition_accepts_enum_values_as_strings(repo):
    repo.add("alpha")
    rec = _transition(repo, "alpha", "backtested", "agent")
    assert rec.stage is Stage.BACKTESTED


def test_illegal_transition_raises(repo):
    repo.add("alpha")
    with pytest.raises(TransitionError):
        _transition(repo, "alpha", Stage.LIVE, Actor.AGENT)


def test_transition_service_allows_injected_live_approval_verifier(repo):
    repo.add("cross_sectional_momentum")
    for stage in (Stage.BACKTESTED, Stage.SHORTLISTED, Stage.PAPER):
        _transition(repo, "cross_sectional_momentum", stage, Actor.AGENT)

    rec = transition_strategy(
        repo,
        "cross_sectional_momentum",
        Stage.LIVE,
        Actor.HUMAN,
        approval_verifier=lambda *_args: True,
    )

    assert rec.stage is Stage.LIVE


def test_list_filters_by_stage(repo):
    repo.add("alpha")
    repo.add("beta")
    _transition(repo, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = repo.list_strategies(Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]


# --- holdout_evaluations -----------------------------------------------------

def test_record_and_query_overlapping_holdout(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Overlapping period, same data_source + holdout_frac -> collision.
    assert repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-06-01", period_end="2024-06-01", holdout_frac=0.2,
    )


def test_holdout_non_overlap_and_frac_and_data_distinguish(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Disjoint period -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2,
    )
    # Different holdout_frac -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.3,
    )
    # Different data_source -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
    )


def test_holdout_snapshot_identity_takes_precedence(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapA",
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Same window, different snapshot id -> distinct data identity, no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapB",
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
    )
    # Same snapshot id, overlapping window -> collision.
    assert repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapA",
        period_start="2022-06-01", period_end="2023-06-01", holdout_frac=0.2,
    )
