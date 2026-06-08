from __future__ import annotations

from pathlib import Path

import pandas as pd

from algua.data.store import normalize_symbols

_FIRSTRATE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]


def symbol_from_filename(name: str) -> str:
    """Derive the symbol from a FirstRate filename like `AAPL_full_1day_UNADJUSTED.txt`.

    The symbol is the filename segment before the first underscore, canonicalized (upper-cased).
    """
    stem = Path(name).name.split("_", 1)[0]
    return normalize_symbols([stem])[0]


def parse_firstrate_file(path: Path) -> pd.DataFrame:
    """Parse one FirstRate daily file into a frame with columns
    `ts, open, high, low, close, volume`. `ts` is a tz-aware UTC-midnight timestamp.

    Handles a present-or-absent header (sniffed from the first non-empty line). Raises ValueError on
    a malformed file (wrong column count, unparseable dates/numbers).
    """
    first_line = ""
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                first_line = line.strip().lower()
                break
    has_header = first_line.startswith("datetime")
    frame = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else _FIRSTRATE_COLUMNS,
        encoding="utf-8-sig",
    )
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in _FIRSTRATE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: FirstRate file missing columns {missing}")
    out = frame[_FIRSTRATE_COLUMNS].rename(columns={"datetime": "ts"})
    # Daily `datetime` is a bare date → localize to UTC midnight (never silently shift).
    parsed = pd.to_datetime(out["ts"], errors="raise")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    out["ts"] = parsed
    for col in [*_PRICE_COLUMNS, "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out
