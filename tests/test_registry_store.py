import pytest
from algua.registry.db import connect, migrate
from algua.registry import store
from algua.contracts.lifecycle import Stage, Actor, TransitionError


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def test_add_creates_idea_with_initial_transition(conn):
    rec = store.add_strategy(conn, "alpha")
    assert rec.stage is Stage.IDEA
    transitions = store.list_transitions(conn, "alpha")
    assert len(transitions) == 1
    assert transitions[0]["to_stage"] == "idea"
    assert transitions[0]["actor"] == "system"


def test_duplicate_name_raises(conn):
    store.add_strategy(conn, "alpha")
    with pytest.raises(store.StrategyExists):
        store.add_strategy(conn, "alpha")


def test_legal_transition_updates_stage_and_history(conn):
    store.add_strategy(conn, "alpha")
    rec = store.transition(conn, "alpha", Stage.BACKTESTED, Actor.AGENT, "ran backtest")
    assert rec.stage is Stage.BACKTESTED
    assert len(store.list_transitions(conn, "alpha")) == 2


def test_illegal_transition_raises(conn):
    store.add_strategy(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.AGENT)


def test_list_filters_by_stage(conn):
    store.add_strategy(conn, "alpha")
    store.add_strategy(conn, "beta")
    store.transition(conn, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = store.list_strategies(conn, Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]
