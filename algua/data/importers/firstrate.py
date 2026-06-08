from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from algua.data.contracts import ImportRequest, ProviderBars
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


def _discover(directory: Path) -> dict[str, Path]:
    """Map canonical symbol -> file path for every file in `directory`.

    Raises ValueError if two files canonicalize to the same symbol (alias collision — the route by
    which a global (timestamp, symbol) duplicate would sneak into the consolidated snapshot).
    """
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        symbol = symbol_from_filename(path.name)
        if symbol in mapping:
            raise ValueError(
                f"duplicate symbol {symbol!r} in {directory.name}: "
                f"{mapping[symbol].name} and {path.name}"
            )
        mapping[symbol] = path
    return mapping


class FirstRateImporter:
    name = "firstrate"

    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]:
        if request.timeframe != "1d":
            raise ValueError("intraday import not yet supported (1d only)")
        raw_map = _discover(request.raw_dir)
        adj_map = _discover(request.adjusted_dir)
        if set(raw_map) != set(adj_map):
            only_raw = sorted(set(raw_map) - set(adj_map))
            only_adj = sorted(set(adj_map) - set(raw_map))
            raise ValueError(
                f"raw/adjusted symbol sets differ; refusing partial import. "
                f"only in raw: {only_raw}; only in adjusted: {only_adj}"
            )
        symbols = sorted(raw_map)
        if request.symbols is not None:
            wanted = set(normalize_symbols(list(request.symbols)))
            missing = sorted(wanted - set(symbols))
            if missing:
                raise ValueError(f"requested symbols with no files: {missing}")
            symbols = [s for s in symbols if s in wanted]
        for symbol in symbols:
            yield self._merge_symbol(symbol, raw_map[symbol], adj_map[symbol])

    def _merge_symbol(self, symbol: str, raw_path: Path, adj_path: Path) -> ProviderBars:
        raw = parse_firstrate_file(raw_path)
        adj = parse_firstrate_file(adj_path)[["ts", "close"]].rename(columns={"close": "adj_close"})

        if raw["ts"].duplicated().any():
            dupes = sorted(
                str(d) for d in raw.loc[raw["ts"].duplicated(keep=False), "ts"].dt.date.unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in raw file: {dupes}")
        if adj["ts"].duplicated().any():
            dupes = sorted(
                str(d) for d in adj.loc[adj["ts"].duplicated(keep=False), "ts"].dt.date.unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in adjusted file: {dupes}")

        raw_keys = set(raw["ts"])
        adj_keys = set(adj["ts"])
        if raw_keys != adj_keys:
            unmatched = sorted(str(ts.date()) for ts in raw_keys.symmetric_difference(adj_keys))
            raise ValueError(
                f"{symbol}: raw and adjusted key sets differ; refusing partial merge. "
                f"unmatched dates: {unmatched}"
            )

        merged = raw.merge(adj, on="ts", how="inner")
        merged["symbol"] = symbol
        price_cols = ["open", "high", "low", "close", "adj_close"]
        if merged[price_cols].isna().to_numpy().any() or (merged[price_cols] <= 0).to_numpy().any():
            raise ValueError(f"{symbol}: NaN or nonpositive price(s) in raw/adjusted data")
        frame = merged[
            ["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
        ].sort_values("ts").reset_index(drop=True)
        return ProviderBars(
            frame=frame,
            source_metadata={
                "vendor": "firstratedata",
                "symbol": symbol,
                "raw_file": raw_path.name,
                "adjusted_file": adj_path.name,
            },
        )
