import pandas as pd
import pytest

from algua.data.schema import BAR_COLUMNS, to_bar_schema, validate_bars


def _good() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2024-07-01", "2024-07-01", "2024-07-02"], utc=True), name="timestamp"
    )
    return pd.DataFrame(
        {"symbol": ["AAA", "BBB", "AAA"], "open": [1.0, 2.0, 1.1], "high": [1.0, 2.0, 1.1],
         "low": [1.0, 2.0, 1.1], "close": [1.0, 2.0, 1.1], "adj_close": [1.0, 2.0, 1.1],
         "volume": [10.0, 20.0, 11.0]},
        index=idx,
    )[BAR_COLUMNS]


def test_validate_accepts_conformant_frame():
    df = _good()
    assert validate_bars(df) is df


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError):
        validate_bars(_good().drop(columns=["adj_close"]))


def test_validate_rejects_tz_naive_index():
    df = _good()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_wrong_index_name():
    df = _good()
    df.index = df.index.rename("ts")
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_nan_ohlc():
    df = _good()
    df.iloc[0, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_unsorted():
    with pytest.raises(ValueError):
        validate_bars(_good().iloc[::-1])  # reversed -> not sorted by (timestamp, symbol)


def test_to_bar_schema_reshapes_ts_column_frame():
    raw = pd.DataFrame(
        {"adj_close": [1.0, 2.0], "close": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
         "open": [1.0, 2.0], "symbol": ["BBB", "AAA"], "ts": ["2024-07-01", "2024-07-01"],
         "volume": [20.0, 10.0]}
    )
    out = to_bar_schema(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.name == "timestamp"
    assert str(out.index.tz) == "UTC"
    assert list(out["symbol"]) == ["AAA", "BBB"]  # sorted by (timestamp, symbol)
    validate_bars(out)
