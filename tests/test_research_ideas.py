# tests/test_research_ideas.py
from algua.contracts.idea import DataCapability, Horizon, IdeaStatus, Market
from algua.research.ideas import classify_idea, classify_status

SUPPORTED = frozenset({DataCapability.OHLCV})


def test_ohlcv_only_is_open():
    assert classify_status([DataCapability.OHLCV], SUPPORTED) is IdeaStatus.OPEN


def test_no_data_is_open():
    assert classify_status([], SUPPORTED) is IdeaStatus.OPEN


def test_unsupported_capability_parks():
    assert classify_status(
        [DataCapability.OHLCV, DataCapability.FORM_13F], SUPPORTED) is IdeaStatus.NEEDS_DATA
    assert classify_status([DataCapability.OPTIONS_FLOW], SUPPORTED) is IdeaStatus.NEEDS_DATA


_CAPS = frozenset({DataCapability.OHLCV})
_MK = frozenset({Market.US_EQUITIES, Market.ANY})
_HZ = frozenset({Horizon.DAILY, Horizon.WEEKLY, Horizon.MONTHLY, Horizon.EVENT})


def test_classify_idea_open_when_all_supported():
    assert classify_idea([DataCapability.OHLCV], _CAPS, market=Market.US_EQUITIES,
                         supported_markets=_MK, horizon=Horizon.DAILY,
                         supported_horizons=_HZ) == (IdeaStatus.OPEN, None)


def test_classify_idea_parks_unsupported_market_with_reason():
    status, reason = classify_idea([DataCapability.OHLCV], _CAPS, market=Market.CRYPTO,
                                   supported_markets=_MK, horizon=Horizon.DAILY,
                                   supported_horizons=_HZ)
    assert status is IdeaStatus.NEEDS_DATA and reason == "market:crypto"


def test_classify_idea_parks_intraday_and_names_every_gap():
    status, reason = classify_idea([DataCapability.FORM_13F], _CAPS, market=Market.FOREX,
                                   supported_markets=_MK, horizon=Horizon.INTRADAY,
                                   supported_horizons=_HZ)
    assert status is IdeaStatus.NEEDS_DATA
    assert reason == "data:form_13f;market:forex;horizon:intraday"


def test_classify_idea_legacy_none_market_and_horizon_are_open():
    assert classify_idea([DataCapability.OHLCV], _CAPS, market=None, supported_markets=_MK,
                         horizon=None, supported_horizons=_HZ) == (IdeaStatus.OPEN, None)
