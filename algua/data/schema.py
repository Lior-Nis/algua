from __future__ import annotations

import pandas as pd

from algua.data.timeframes import DAILY

# Single source of truth for the numeric (float64, non-null) bar columns. `BAR_COLUMNS` (the full
# stored/serving column order) prepends the string `symbol` column. `NON_NULL_COLUMNS` and
# `FLOAT_COLUMNS` are the same set under two names the validator reads, derived — not duplicated.
FLOAT_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
NON_NULL_COLUMNS = FLOAT_COLUMNS
BAR_COLUMNS = ["symbol", *FLOAT_COLUMNS]

# Re-exported so algua.data.files can serialize the numeric bar columns for the logical content
# hash without importing store (schema imports only pandas, so this stays cycle-free).
BARS_FILE_HASH_COLUMNS = FLOAT_COLUMNS


def validate_bars(df: pd.DataFrame, *, timeframe: str | None = None) -> pd.DataFrame:
    """Assert `df` matches the frozen bar schema; return it unchanged on success.

    When `timeframe == "1d"`, the frozen contract pins each daily timestamp to the session date at
    UTC midnight; this asserts it (the invariant the FirstRate/Databento importers already enforce,
    now enforced uniformly on the single rail every ingest/serve path crosses — issue #262). Any
    other (or `None`) `timeframe` skips the daily-anchor check: intraday bars are NOT clock-aligned
    by contract, and `None` exists only for `empty_bars()` (a 0-row frame, vacuous regardless).

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
    if not all(isinstance(symbol, str) for symbol in df["symbol"]):
        raise ValueError("bars symbol values must be strings")
    bad_dtypes = [col for col in FLOAT_COLUMNS if str(df[col].dtype) != "float64"]
    if bad_dtypes:
        raise ValueError(f"bars numeric columns must be float64: {bad_dtypes}")
    if df[NON_NULL_COLUMNS].isna().any().any():
        raise ValueError("bars values (OHLC/adj_close/volume) must not contain NaN")
    keys = df.reset_index()[["timestamp", "symbol"]]
    if keys.duplicated().any():
        raise ValueError("bars must not contain duplicate (timestamp, symbol) rows")
    if not keys.equals(keys.sort_values(["timestamp", "symbol"]).reset_index(drop=True)):
        raise ValueError("bars must be sorted by (timestamp, symbol)")
    if timeframe == DAILY and len(df.index) and not df.index.equals(df.index.normalize()):
        bad = df.index[df.index != df.index.normalize()][0]
        raise ValueError(
            f"daily (1d) bars must be timestamped at UTC midnight (the session date); "
            f"found a non-midnight timestamp {bad}"
        )
    return df


def empty_bars() -> pd.DataFrame:
    """The contract's empty-but-typed bars frame: exact `BAR_COLUMNS`, float64 numeric dtypes, and
    a tz-aware empty UTC `timestamp` index. The read path returns this when a pushdown filter
    matches no rows (issue #130, GATE-1 MEDIUM #5) so consumers can rely on the schema even when a
    query is empty, instead of whatever an empty `to_pandas()` happens to produce."""
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    data = {"symbol": pd.Series([], dtype="object")}
    for col in FLOAT_COLUMNS:
        data[col] = pd.Series([], dtype="float64")
    return validate_bars(pd.DataFrame(data, index=index))


def to_bar_schema(frame: pd.DataFrame, *, timeframe: str | None) -> pd.DataFrame:
    """Reshape a stored bars frame (a `ts` column + OHLCV + symbol, any column order) into the
    bar schema: tz-aware UTC `timestamp` index, ordered columns, sorted, validated.

    `timeframe` is required (keyword-only) so no ingest/serve path can silently skip the daily
    UTC-midnight check (issue #262); pass the bars' timeframe, or `None` only when it is genuinely
    unknown (e.g. a legacy snapshot that recorded no timeframe), which skips that one check."""
    out = frame.copy()
    if "ts" in out.columns:
        out = out.rename(columns={"ts": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("frame must have a 'ts' or 'timestamp' column")
    # Providers deliver UTC. Parse without coercing tz so naive timestamps surface as NaT-free but
    # tz-naive, which we reject — never silently localize (that would shift intraday bars).
    parsed = pd.to_datetime(out["timestamp"], errors="raise")
    if parsed.dt.tz is None:
        raise ValueError(
            "bars 'ts'/'timestamp' must be tz-aware (providers deliver UTC); "
            "naive timestamps are rejected, not localized"
        )
    out["timestamp"] = parsed.dt.tz_convert("UTC")
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"frame missing bar columns: {missing}")
    out = out[["timestamp", *BAR_COLUMNS]]
    out["symbol"] = out["symbol"].astype(str)
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    out = out.sort_values(["timestamp", "symbol"]).set_index("timestamp")
    return validate_bars(out, timeframe=timeframe)
