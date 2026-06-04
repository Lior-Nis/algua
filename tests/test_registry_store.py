import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.repository import StrategyExists
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


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
