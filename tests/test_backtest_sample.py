from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 3, 1, tzinfo=UTC)
COLS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]


def test_returns_bar_schema_shape():
    df = SyntheticProvider(seed=1).get_bars(["AAA", "BBB"], START, END, "1d")
    assert list(df.columns) == COLS
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert set(df["symbol"].unique()) == {"AAA", "BBB"}
    assert df.index.is_monotonic_increasing
    assert not df[["open", "high", "low", "close", "adj_close"]].isna().any().any()


def test_deterministic_for_same_seed():
    a = SyntheticProvider(seed=42).get_bars(["AAA"], START, END, "1d")
    b = SyntheticProvider(seed=42).get_bars(["AAA"], START, END, "1d")
    pd.testing.assert_frame_equal(a, b)


def test_rejects_unknown_timeframe():
    import pytest
    with pytest.raises(ValueError):
        SyntheticProvider().get_bars(["AAA"], START, END, "7h")


def test_timestamps_are_utc_session_dates():
    df = SyntheticProvider(seed=1).get_bars(["AAA"], START, END, "1d")
    assert str(df.index.tz) == "UTC"
    assert (df.index.hour == 0).all()
    assert (df.index.normalize() == df.index).all()  # exactly midnight


def test_ohlc_invariant_holds():
    """low <= open,close <= high for every bar and open != close (realistic bars)."""
    df = SyntheticProvider(seed=7).get_bars(["AAA", "BBB"], START, END, "1d")
    assert (df["low"] <= df["open"]).all(), "low > open on at least one bar"
    assert (df["low"] <= df["close"]).all(), "low > close on at least one bar"
    assert (df["high"] >= df["open"]).all(), "high < open on at least one bar"
    assert (df["high"] >= df["close"]).all(), "high < close on at least one bar"
    assert (df["high"] > df["low"]).all(), "high == low (zero-width bar)"
    # open should differ from close so high/low-dependent strategies can fire
    assert (df["open"] != df["close"]).any(), "open == close for every bar (cosmetic OHLC)"
