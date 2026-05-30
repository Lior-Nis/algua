import pytest
from algua.contracts.lifecycle import (
    Stage, Actor, can_transition, validate_transition, TransitionError,
)


def test_legal_transition():
    assert can_transition(Stage.IDEA, Stage.BACKTESTED) is True
    assert can_transition(Stage.SHORTLISTED, Stage.PAPER) is True


def test_illegal_transition():
    assert can_transition(Stage.IDEA, Stage.LIVE) is False
    assert can_transition(Stage.RETIRED, Stage.IDEA) is False


def test_validate_raises_on_illegal():
    with pytest.raises(TransitionError):
        validate_transition(Stage.IDEA, Stage.LIVE)


def test_actor_values():
    assert {a.value for a in Actor} == {"human", "agent", "system"}
