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


def test_timestamps_are_session_closes_not_midnight():
    df = SyntheticProvider(seed=1).get_bars(["AAA"], START, END, "1d")
    # daily bars carry the session-close time (16:00 ET -> 20:00/21:00 UTC), never naive midnight
    assert (df.index.hour != 0).all()
    assert str(df.index.tz) == "UTC"
