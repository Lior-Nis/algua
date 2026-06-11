import pytest

from algua.contracts.lifecycle import (
    ALLOWED_TRANSITIONS,
    Actor,
    Stage,
    TransitionError,
    can_transition,
    validate_transition,
)


def test_legal_transition():
    assert can_transition(Stage.IDEA, Stage.BACKTESTED) is True
    assert can_transition(Stage.CANDIDATE, Stage.PAPER) is True


def test_illegal_transition():
    assert can_transition(Stage.IDEA, Stage.LIVE) is False
    assert can_transition(Stage.RETIRED, Stage.IDEA) is False


def test_validate_raises_on_illegal():
    with pytest.raises(TransitionError):
        validate_transition(Stage.IDEA, Stage.LIVE)


def test_actor_values():
    assert {a.value for a in Actor} == {"human", "agent", "system"}


def test_transition_table_is_total():
    """Every Stage is a key with valid Stage targets (no stage forgotten)."""
    assert set(ALLOWED_TRANSITIONS) == set(Stage)
    for frm, targets in ALLOWED_TRANSITIONS.items():
        assert all(isinstance(t, Stage) for t in targets)
        assert frm not in targets  # no self-loops


def test_every_non_retired_stage_can_retire():
    """RETIRED is reachable from every live stage; RETIRED itself is terminal."""
    for stage in Stage:
        if stage is Stage.RETIRED:
            assert can_transition(stage, Stage.RETIRED) is False
        else:
            assert can_transition(stage, Stage.RETIRED) is True


def test_forward_tested_edges():
    assert can_transition(Stage.PAPER, Stage.FORWARD_TESTED)
    assert can_transition(Stage.FORWARD_TESTED, Stage.LIVE)
    assert can_transition(Stage.FORWARD_TESTED, Stage.PAPER)
    assert can_transition(Stage.FORWARD_TESTED, Stage.RETIRED)  # derived retire edge


def test_paper_to_live_removed_for_everyone():
    assert not can_transition(Stage.PAPER, Stage.LIVE)


def test_live_demotion_still_lands_at_paper():
    assert can_transition(Stage.LIVE, Stage.PAPER)
    assert not can_transition(Stage.LIVE, Stage.FORWARD_TESTED)
