from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytz

from algua.data.contracts import FirstRateImportRequest, ImportRequest, ProviderBars
from algua.data.store import normalize_symbols
from algua.data.timeframes import is_intraday, validate_timeframe

_FIRSTRATE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]
# Delimiter-bounded so the role marker is matched as a token, not a bare substring: a raw file
# named `…notunadjusted…` doesn't count as unadjusted, and an adjusted file whose name merely
# contains the letters "unadjusted" mid-word isn't false-rejected.
_UNADJUSTED_TOKEN_RE = re.compile(r"(?:^|[_\-.])unadjusted(?:$|[_\-.])")
# FirstRate US-equity intraday timestamps are naive wall-clock in US/Eastern.
_EXCHANGE_TZ = "America/New_York"


def _looks_unadjusted(name: str) -> bool:
    """True if the filename carries FirstRate's `unadjusted` role token (case-insensitive)."""
    return _UNADJUSTED_TOKEN_RE.search(name.lower()) is not None


def symbol_from_filename(name: str) -> str:
    """Derive the symbol from a FirstRate filename like `AAPL_full_1day_UNADJUSTED.txt`.

    The symbol is the filename segment before the first underscore, canonicalized (upper-cased).
    """
    stem = Path(name).name.split("_", 1)[0]
    return normalize_symbols([stem])[0]


def _localize_timestamps(parsed: pd.Series, timeframe: str, fname: str) -> pd.Series:
    """Localize a parsed `datetime` series to tz-aware UTC for the given timeframe.

    Daily: a bare date (naive) -> UTC midnight; a tz-aware value -> converted to UTC; then every
    value MUST be UTC midnight (a non-midnight `1d` timestamp means an intraday file was imported
    as daily — fail closed, parity with the Databento importer).

    Intraday: the input MUST be naive ET wall-clock. A tz-aware column is rejected (its wall-clock
    tz is unknowable). A local-midnight value is rejected (a 00:00 bar is never a valid FirstRate
    US-equity intraday bar -> a date-only/daily file was misfed). Otherwise localize to
    `America/New_York` (DST-ambiguous/nonexistent -> ValueError naming the time) and convert to UTC.
    """
    # Mixed UTC offsets (e.g. one row `-04:00`, another `-05:00`) make pd.to_datetime return an
    # object-dtype Series with no `.dt` accessor. Reject it cleanly here (a ValueError that
    # `import-bars`' json_errors() catches) rather than letting `.dt` raise AttributeError below.
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError(
            f"{fname}: timestamps did not parse to a single datetime dtype (mixed timezone "
            "offsets?); FirstRate timestamps must be naive wall-clock US/Eastern"
        )
    if not is_intraday(timeframe):
        utc = parsed.dt.tz_localize("UTC") if parsed.dt.tz is None else parsed.dt.tz_convert("UTC")
        if len(utc) and not bool((utc == utc.dt.normalize()).all()):
            bad = utc[utc != utc.dt.normalize()].iloc[0]
            raise ValueError(
                f"{fname}: 1d file has a non-midnight timestamp ({bad}); "
                "looks like an intraday file imported as daily"
            )
        return utc
    if parsed.dt.tz is not None:
        raise ValueError(
            f"{fname}: FirstRate intraday timestamps must be naive (wall-clock US/Eastern); "
            "found a tz-aware column — strip timezone offsets before importing"
        )
    if len(parsed) and bool((parsed == parsed.dt.normalize()).any()):
        bad = parsed[parsed == parsed.dt.normalize()].iloc[0]
        raise ValueError(
            f"{fname}: intraday file has a local-midnight timestamp ({bad}); "
            "looks like a date-only/daily file imported as intraday"
        )
    try:
        local = parsed.dt.tz_localize(_EXCHANGE_TZ, ambiguous="raise", nonexistent="raise")
    except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError) as exc:
        raise ValueError(
            f"{fname}: DST-ambiguous or nonexistent local time in {_EXCHANGE_TZ}: {exc}"
        ) from exc
    return local.dt.tz_convert("UTC")


def parse_firstrate_file(path: Path, timeframe: str = "1d") -> pd.DataFrame:
    """Parse one FirstRate file into a frame with columns `ts, open, high, low, close, volume`.

    `ts` is a tz-aware UTC timestamp. For `timeframe="1d"` the source is a bare date localized to
    UTC midnight; for an intraday `timeframe` the source is naive US/Eastern wall-clock localized to
    `America/New_York` and converted to UTC (see `_localize_timestamps`).

    Handles a present-or-absent header (sniffed from the first non-empty line). Raises ValueError on
    a malformed file (wrong column count, unparseable dates/numbers) or a timeframe/timestamp
    mismatch (non-midnight daily, tz-aware or local-midnight intraday, DST-invalid local time).
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
    parsed = pd.to_datetime(out["ts"], errors="raise")
    out["ts"] = _localize_timestamps(parsed, timeframe, path.name)
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


def _check_directory_roles(raw_map: dict[str, Path], adj_map: dict[str, Path]) -> None:
    """Fail closed if the raw/adjusted dirs look swapped or mislabeled.

    FirstRate unadjusted files carry the case-insensitive token `unadjusted` in their name; adjusted
    files do not. So every raw file must look unadjusted and no adjusted file may — either violation
    raises ValueError. Together these also catch both dirs pointing at the same directory.
    """
    raw_not_unadjusted = sorted(
        path.name for path in raw_map.values() if not _looks_unadjusted(path.name)
    )
    if raw_not_unadjusted:
        raise ValueError(
            f"raw dir holds files that don't look unadjusted: {raw_not_unadjusted}; "
            f"--raw-dir/--adjusted-dir may be swapped"
        )
    adj_unadjusted = sorted(
        path.name for path in adj_map.values() if _looks_unadjusted(path.name)
    )
    if adj_unadjusted:
        raise ValueError(
            f"adjusted dir holds unadjusted-looking files: {adj_unadjusted}; "
            f"--raw-dir/--adjusted-dir may be swapped"
        )


class FirstRateImporter:
    name = "firstrate"
    vendor_label = "firstratedata"

    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]:
        if not isinstance(request, FirstRateImportRequest):
            raise ValueError("FirstRateImporter requires a FirstRateImportRequest")
        validate_timeframe(request.timeframe)
        raw_map = _discover(request.raw_dir)
        adj_map = _discover(request.adjusted_dir)
        _check_directory_roles(raw_map, adj_map)
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
            yield self._merge_symbol(symbol, raw_map[symbol], adj_map[symbol], request.timeframe)

    def _merge_symbol(
        self, symbol: str, raw_path: Path, adj_path: Path, timeframe: str
    ) -> ProviderBars:
        raw = parse_firstrate_file(raw_path, timeframe)
        adj = parse_firstrate_file(adj_path, timeframe)[["ts", "close"]].rename(
            columns={"close": "adj_close"}
        )

        if raw["ts"].duplicated().any():
            dupes = sorted(
                str(ts) for ts in raw.loc[raw["ts"].duplicated(keep=False), "ts"].unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in raw file: {dupes}")
        if adj["ts"].duplicated().any():
            dupes = sorted(
                str(ts) for ts in adj.loc[adj["ts"].duplicated(keep=False), "ts"].unique()
            )
            raise ValueError(f"{symbol}: duplicate timestamps in adjusted file: {dupes}")

        raw_keys = set(raw["ts"])
        adj_keys = set(adj["ts"])
        if raw_keys != adj_keys:
            unmatched = sorted(str(ts) for ts in raw_keys.symmetric_difference(adj_keys))
            raise ValueError(
                f"{symbol}: raw and adjusted key sets differ; refusing partial merge. "
                f"unmatched timestamps: {unmatched}"
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
