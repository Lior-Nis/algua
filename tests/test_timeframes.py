# tests/test_timeframes.py
import pytest

from algua.data.timeframes import (
    DAILY,
    INTRADAY,
    KNOWN,
    is_intraday,
    validate_timeframe,
)


def test_known_set_is_daily_plus_intraday():
    assert DAILY == "1d"
    assert INTRADAY == frozenset({"1m", "5m", "30m", "1h"})
    assert KNOWN == frozenset({"1d", "1m", "5m", "30m", "1h"})


@pytest.mark.parametrize("tf", ["1d", "1m", "5m", "30m", "1h"])
def test_validate_timeframe_accepts_known(tf):
    assert validate_timeframe(tf) == tf


@pytest.mark.parametrize("tf", ["5min", "1hr", "15m", "2h", "", "1D"])
def test_validate_timeframe_rejects_unknown(tf):
    with pytest.raises(ValueError, match="unknown timeframe"):
        validate_timeframe(tf)


def test_is_intraday_classifies():
    assert is_intraday("1d") is False
    for tf in ["1m", "5m", "30m", "1h"]:
        assert is_intraday(tf) is True


def test_is_intraday_rejects_unknown():
    with pytest.raises(ValueError, match="unknown timeframe"):
        is_intraday("nope")
