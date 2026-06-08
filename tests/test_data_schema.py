import pandas as pd
import pytest

from algua.data.schema import BAR_COLUMNS, empty_bars, to_bar_schema, validate_bars


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


def test_validate_rejects_non_float_numeric_dtype():
    df = _good()
    df["volume"] = df["volume"].astype("int64")
    with pytest.raises(ValueError, match="float64"):
        validate_bars(df)


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
         "open": [1.0, 2.0], "symbol": ["BBB", "AAA"],
         "ts": ["2024-07-01T00:00:00+00:00", "2024-07-01T00:00:00+00:00"],
         "volume": [20.0, 10.0]}
    )
    out = to_bar_schema(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.name == "timestamp"
    assert str(out.index.tz) == "UTC"
    assert list(out["symbol"]) == ["AAA", "BBB"]  # sorted by (timestamp, symbol)
    assert all(str(out[col].dtype) == "float64" for col in BAR_COLUMNS if col != "symbol")
    validate_bars(out)


def test_to_bar_schema_rejects_non_numeric_values():
    raw = pd.DataFrame(
        {"adj_close": [1.0], "close": ["bad"], "high": [1.0], "low": [1.0],
         "open": [1.0], "symbol": ["AAA"], "ts": ["2024-07-01T00:00:00+00:00"], "volume": [10.0]}
    )
    with pytest.raises(ValueError):
        to_bar_schema(raw)


def test_to_bar_schema_requires_adj_close():
    raw = pd.DataFrame({
        "ts": ["2024-07-01T00:00:00+00:00", "2024-07-02T00:00:00+00:00"], "symbol": ["AAA", "AAA"],
        "open": [10.0, 11.0], "high": [10.0, 11.0], "low": [10.0, 11.0],
        "close": [10.0, 11.0], "volume": [1.0, 1.0],
    })
    with pytest.raises(ValueError):
        to_bar_schema(raw)


def test_to_bar_schema_rejects_naive_timestamps():
    raw = pd.DataFrame(
        {"adj_close": [1.0], "close": [1.0], "high": [1.0], "low": [1.0],
         "open": [1.0], "symbol": ["AAA"], "ts": ["2024-07-01"], "volume": [10.0]}
    )
    with pytest.raises(ValueError, match="tz-aware"):
        to_bar_schema(raw)


def test_to_bar_schema_preserves_non_utc_offset_as_utc_instant():
    # A tz-aware non-UTC timestamp is accepted and converted to UTC (same instant), not shifted.
    raw = pd.DataFrame(
        {"adj_close": [1.0], "close": [1.0], "high": [1.0], "low": [1.0],
         "open": [1.0], "symbol": ["AAA"], "ts": ["2024-07-01T05:00:00+05:00"], "volume": [10.0]}
    )
    out = to_bar_schema(raw)
    assert out.index[0] == pd.Timestamp("2024-07-01T00:00:00", tz="UTC")


def test_bar_column_lists_are_consistent():
    from algua.data.schema import FLOAT_COLUMNS, NON_NULL_COLUMNS

    assert NON_NULL_COLUMNS == FLOAT_COLUMNS
    assert BAR_COLUMNS == ["symbol", *FLOAT_COLUMNS]


def test_validate_rejects_nan_volume():
    df = _good()
    df.iloc[0, df.columns.get_loc("volume")] = float("nan")
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_duplicate_keys():
    df = _good()
    dup = pd.concat([df.iloc[[0]], df])  # duplicate first (timestamp, symbol)
    dup = dup.sort_index()
    with pytest.raises(ValueError):
        validate_bars(dup)


def test_empty_bars_is_contract_shaped():
    out = empty_bars()
    assert out.empty
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.name == "timestamp"
    assert str(out.index.tz) == "UTC"
    validate_bars(out)  # must satisfy the frozen schema unconditionally
