# tests/test_corpactions.py
import math

import pandas as pd
import pytest

from algua.data.corpactions import Dividend, Split


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_split_and_dividend_construct():
    s = Split(ex_date=_utc("2024-01-03"), ratio=2.0)
    d = Dividend(ex_date=_utc("2024-01-03"), cash=1.5)
    assert s.ratio == 2.0 and d.cash == 1.5


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_split_ratio_must_be_finite_positive(ratio):
    with pytest.raises(ValueError, match="ratio"):
        Split(ex_date=_utc("2024-01-03"), ratio=ratio)


@pytest.mark.parametrize("cash", [0.0, -1.0, float("inf"), float("nan")])
def test_dividend_cash_must_be_finite_positive(cash):
    with pytest.raises(ValueError, match="cash"):
        Dividend(ex_date=_utc("2024-01-03"), cash=cash)


def test_tz_naive_ex_date_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        Split(ex_date=pd.Timestamp("2024-01-03"), ratio=2.0)
    with pytest.raises(ValueError, match="tz-aware"):
        Dividend(ex_date=pd.Timestamp("2024-01-03"), cash=1.0)
