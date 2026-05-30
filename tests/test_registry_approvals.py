import pytest
from algua.registry.db import connect, migrate
from algua.registry import store
from algua.registry.approvals import record_approval, has_valid_approval
from algua.contracts.lifecycle import Stage, Actor, TransitionError


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return c


def _advance_to_paper(conn, name):
    store.add_strategy(conn, name)
    store.transition(conn, name, Stage.BACKTESTED, Actor.AGENT)
    store.transition(conn, name, Stage.SHORTLISTED, Actor.AGENT)
    store.transition(conn, name, Stage.PAPER, Actor.AGENT)


def test_live_requires_approval(conn):
    _advance_to_paper(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                         code_hash="c1", config_hash="g1")


def test_live_requires_human_actor(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.AGENT,
                         code_hash="c1", config_hash="g1")


def test_live_requires_matching_hash(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                         code_hash="DIFFERENT", config_hash="g1")


def test_live_succeeds_with_human_and_matching_approval(conn):
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    rec = store.transition(conn, "alpha", Stage.LIVE, Actor.HUMAN,
                           code_hash="c1", config_hash="g1")
    assert rec.stage is Stage.LIVE


def test_has_valid_approval(conn):
    store.add_strategy(conn, "alpha")
    s = store.get_strategy(conn, "alpha")
    assert has_valid_approval(conn, s.id, "c1", "g1") is False
    record_approval(conn, "alpha", "c1", "g1", "lior")
    assert has_valid_approval(conn, s.id, "c1", "g1") is True


def test_string_live_does_not_bypass_gate(conn):
    # Passing the raw string "live" (not Stage.LIVE) must still engage the gate,
    # raising TransitionError rather than skipping it / crashing on .value.
    _advance_to_paper(conn, "alpha")
    with pytest.raises(TransitionError):
        store.transition(conn, "alpha", "live", Actor.HUMAN,
                         code_hash="c1", config_hash="g1")


def test_string_live_succeeds_with_approval(conn):
    # And with a valid approval, the string form promotes correctly (coercion works).
    _advance_to_paper(conn, "alpha")
    record_approval(conn, "alpha", "c1", "g1", "lior")
    rec = store.transition(conn, "alpha", "live", Actor.HUMAN,
                           code_hash="c1", config_hash="g1")
    assert rec.stage is Stage.LIVE
