# tests/test_idea_contracts.py
from algua.contracts.idea import (
    ALLOWED_IDEA_TRANSITIONS,
    DataCapability,
    Idea,
    IdeaStatus,
    SourceType,
    can_change_status,
)


def test_enum_values():
    assert IdeaStatus.OPEN == "open"
    assert IdeaStatus.NEEDS_DATA == "needs_data"
    assert {s.value for s in IdeaStatus} == {
        "open", "needs_data", "authored", "refuted", "discarded"}
    assert SourceType.PAPER == "paper"
    assert DataCapability.OHLCV == "ohlcv"
    assert DataCapability.FORM_13F == "form_13f"


def test_status_transitions_legal():
    assert can_change_status(IdeaStatus.OPEN, IdeaStatus.AUTHORED)
    assert can_change_status(IdeaStatus.OPEN, IdeaStatus.NEEDS_DATA)
    assert can_change_status(IdeaStatus.NEEDS_DATA, IdeaStatus.OPEN)
    assert can_change_status(IdeaStatus.AUTHORED, IdeaStatus.REFUTED)


def test_status_transitions_illegal():
    # terminal states go nowhere
    assert ALLOWED_IDEA_TRANSITIONS[IdeaStatus.REFUTED] == set()
    assert ALLOWED_IDEA_TRANSITIONS[IdeaStatus.DISCARDED] == set()
    # cannot resurrect a refuted idea or re-open an authored one
    assert not can_change_status(IdeaStatus.REFUTED, IdeaStatus.OPEN)
    assert not can_change_status(IdeaStatus.AUTHORED, IdeaStatus.OPEN)
    # no-op is not a legal "change"
    assert not can_change_status(IdeaStatus.OPEN, IdeaStatus.OPEN)


def test_idea_dataclass_fields():
    idea = Idea(
        id=1, title="t", hypothesis="h", family="mom", tags=["x"],
        source_type=SourceType.PAPER, source_ref="u", source_date=None, source_note=None,
        required_data=[DataCapability.OHLCV], status=IdeaStatus.OPEN, signature="h t",
        authored_strategy_id=None, duplicate_of_idea_id=None, override_reason=None,
        created_at="2026-06-08", updated_at="2026-06-08",
    )
    assert idea.required_data == [DataCapability.OHLCV]
    assert idea.status is IdeaStatus.OPEN
