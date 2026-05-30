import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.registry.transitions import transition_strategy


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


def test_transition_accepts_enum_values_as_strings(conn):
    store.add_strategy(conn, "alpha")
    rec = store.transition(conn, "alpha", "backtested", "agent")
    assert rec.stage is Stage.BACKTESTED


def test_illegal_transition_raises(conn):
    store.add_strategy(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.AGENT)


def test_transition_service_allows_injected_live_approval_verifier(conn):
    store.add_strategy(conn, "alpha")
    for stage in (Stage.BACKTESTED, Stage.SHORTLISTED, Stage.PAPER):
        store.transition(conn, "alpha", stage, Actor.AGENT)

    rec = transition_strategy(
        conn,
        "alpha",
        Stage.LIVE,
        Actor.HUMAN,
        code_hash="c1",
        config_hash="g1",
        approval_verifier=lambda _conn, _strategy_id, code_hash, config_hash: (
            code_hash == "c1" and config_hash == "g1"
        ),
    )

    assert rec.stage is Stage.LIVE


def test_list_filters_by_stage(conn):
    store.add_strategy(conn, "alpha")
    store.add_strategy(conn, "beta")
    store.transition(conn, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = store.list_strategies(conn, Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]
