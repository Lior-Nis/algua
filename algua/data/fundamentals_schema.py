from __future__ import annotations

import hashlib
import struct
from datetime import date, datetime

import numpy as np
import pandas as pd

from algua.contracts.types import (
    FUNDAMENTALS_AS_OF_KEY,
    FUNDAMENTALS_COLUMNS,
    FUNDAMENTALS_KNOWABLE_AT,
)

COLUMNS = list(FUNDAMENTALS_COLUMNS)
KEY = list(FUNDAMENTALS_AS_OF_KEY)
STRING_COLUMNS = ["symbol", "metric", "source"]
_SORT = [*KEY, FUNDAMENTALS_KNOWABLE_AT]


def validate_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Assert `df` matches the tidy/bitemporal fundamentals schema; return it unchanged on success.
    Raises ValueError describing the first violation."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"fundamentals columns must be {COLUMNS}, got {list(df.columns)}")
    for col in STRING_COLUMNS:
        if df[col].isna().any() or not all(isinstance(v, str) for v in df[col]):
            raise ValueError(f"fundamentals {col!r} must be non-null strings")
    if str(df["value"].dtype) != "float64":
        raise ValueError("fundamentals 'value' must be float64 (NaN permitted)")
    fpe = df["fiscal_period_end"]
    if not all(isinstance(v, date) and not isinstance(v, datetime) for v in fpe):
        raise ValueError("fundamentals 'fiscal_period_end' must be datetime.date values")
    ka = df[FUNDAMENTALS_KNOWABLE_AT]
    if not isinstance(ka.dtype, pd.DatetimeTZDtype) or str(ka.dt.tz) != "UTC":
        raise ValueError("fundamentals 'knowable_at' must be tz-aware UTC datetimes")
    if ka.isna().any():
        raise ValueError("fundamentals 'knowable_at' must not be null")
    # PIT floor: knowable_at >= start-of-day UTC of fiscal_period_end (same-day filing is valid).
    floor = pd.to_datetime([d.isoformat() for d in fpe], utc=True)
    if (ka.to_numpy() < floor.to_numpy()).any():
        raise ValueError(
            "fundamentals 'knowable_at' must be >= fiscal_period_end (UTC midnight floor)"
        )
    keys = df[_SORT]
    if keys.duplicated().any():
        raise ValueError(
            "fundamentals must have unique (symbol, fiscal_period_end, metric, knowable_at)"
        )
    if df.duplicated().any():
        raise ValueError("fundamentals must not contain exact-duplicate rows")
    expected = df.sort_values(_SORT).reset_index(drop=True)
    if not df.reset_index(drop=True).equals(expected):
        raise ValueError(f"fundamentals must be sorted by {_SORT}")
    return df


def to_fundamentals_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape/normalize an incoming tidy frame into canonical fundamentals form and validate.
    Symbols are upper-cased to match the (normalized) strategy universe; `fiscal_period_end` becomes
    datetime.date; `knowable_at` becomes tz-aware UTC (naive rejected, never localized)."""
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing fundamentals columns: {missing}")
    out = frame[COLUMNS].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["metric"] = out["metric"].astype(str)
    out["source"] = out["source"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="raise").astype("float64")
    fpe = pd.to_datetime(out["fiscal_period_end"], errors="raise")
    if getattr(fpe.dt, "tz", None) is not None:
        fpe = fpe.dt.tz_convert("UTC").dt.tz_localize(None)
    out["fiscal_period_end"] = [ts.date() for ts in fpe]
    ka = pd.to_datetime(out["knowable_at"], errors="raise")
    if ka.dt.tz is None:
        raise ValueError(
            "fundamentals 'knowable_at' must be tz-aware (UTC); naive timestamps are rejected"
        )
    out["knowable_at"] = ka.dt.tz_convert("UTC")
    out = out.sort_values(_SORT).reset_index(drop=True)
    return validate_fundamentals(out)


def empty_fundamentals() -> pd.DataFrame:
    """Contract-shaped empty fundamentals frame (exact columns + dtypes)."""
    data = {
        "symbol": pd.Series([], dtype="object"),
        "fiscal_period_end": pd.Series([], dtype="object"),
        "metric": pd.Series([], dtype="object"),
        "value": pd.Series([], dtype="float64"),
        "knowable_at": pd.Series([], dtype="datetime64[ns, UTC]"),
        "source": pd.Series([], dtype="object"),
    }
    return validate_fundamentals(pd.DataFrame(data)[COLUMNS])


def logical_fundamentals_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash over the logical rows, independent of parquet layout/version —
    the snapshot identity (mirrors `logical_bars_hash`). Rows sorted canonically; strings
    length-prefixed UTF-8; dates as int64 ordinals; knowable_at as int64 ns UTC; value as float64
    with -0.0 -> +0.0 and a canonical NaN bit-pattern."""
    ordered = df.sort_values(_SORT, kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(ordered)))
    for col in STRING_COLUMNS:
        encoded = [s.encode("utf-8") for s in ordered[col].astype(str)]
        lengths = np.array([len(b) for b in encoded], dtype="<u8")
        digest.update(lengths.tobytes())
        digest.update(b"".join(encoded))
    fpe_ord = np.array([d.toordinal() for d in ordered["fiscal_period_end"]], dtype="<i8")
    digest.update(fpe_ord.tobytes())
    ka = ordered["knowable_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    ka_ns = ka.to_numpy(dtype="datetime64[ns]").view("int64").astype("<i8")
    digest.update(ka_ns.tobytes())
    vals = ordered["value"].to_numpy(dtype="<f8") + 0.0  # -0.0 -> +0.0
    vals = np.where(np.isnan(vals), np.float64("nan"), vals)  # canonical NaN
    digest.update(vals.astype("<f8").tobytes())
    return digest.hexdigest()
