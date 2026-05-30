from __future__ import annotations

import pandas as pd

BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]
_NON_NULL = ["open", "high", "low", "close", "adj_close", "volume"]


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Assert `df` matches the frozen bar schema; return it unchanged on success.

    Raises ValueError describing the first violation.
    """
    if df.index.name != "timestamp":
        raise ValueError(f"bars index must be named 'timestamp', got {df.index.name!r}")
    if (
        not isinstance(df.index, pd.DatetimeIndex)
        or df.index.tz is None
        or str(df.index.tz) != "UTC"
    ):
        raise ValueError("bars index must be a tz-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("bars index must be monotonic non-decreasing")
    if list(df.columns) != BAR_COLUMNS:
        raise ValueError(f"bars columns must be {BAR_COLUMNS}, got {list(df.columns)}")
    if df[_NON_NULL].isna().any().any():
        raise ValueError("bars values (OHLC/adj_close/volume) must not contain NaN")
    keys = df.reset_index()[["timestamp", "symbol"]]
    if keys.duplicated().any():
        raise ValueError("bars must not contain duplicate (timestamp, symbol) rows")
    if not keys.equals(keys.sort_values(["timestamp", "symbol"]).reset_index(drop=True)):
        raise ValueError("bars must be sorted by (timestamp, symbol)")
    return df


def to_bar_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape a stored bars frame (a `ts` column + OHLCV + symbol, any column order) into the
    bar schema: tz-aware UTC `timestamp` index, ordered columns, sorted, validated."""
    out = frame.copy()
    if "ts" in out.columns:
        out = out.rename(columns={"ts": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("frame must have a 'ts' or 'timestamp' column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"frame missing bar columns: {missing}")
    out = out[["timestamp", *BAR_COLUMNS]]
    out = out.sort_values(["timestamp", "symbol"]).set_index("timestamp")
    return validate_bars(out)
