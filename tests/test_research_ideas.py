# tests/test_research_ideas.py
from algua.contracts.idea import DataCapability, IdeaStatus
from algua.research.ideas import classify_status

SUPPORTED = frozenset({DataCapability.OHLCV})


def test_ohlcv_only_is_open():
    assert classify_status([DataCapability.OHLCV], SUPPORTED) is IdeaStatus.OPEN


def test_no_data_is_open():
    assert classify_status([], SUPPORTED) is IdeaStatus.OPEN


def test_unsupported_capability_parks():
    assert classify_status(
        [DataCapability.OHLCV, DataCapability.FORM_13F], SUPPORTED) is IdeaStatus.NEEDS_DATA
    assert classify_status([DataCapability.OPTIONS_FLOW], SUPPORTED) is IdeaStatus.NEEDS_DATA
