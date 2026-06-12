"""Databento canonical-parquet importer (#150).

Normalizes a canonical local form — per-symbol raw OHLC parquet + one corporate-action events
parquet — into the bar-schema, computing `adj_close` via the #149 CA engine. NOT a parser of
Databento's native binary format (int-scaled prices / instrument_id / ns ts); the operator conforms
an export to this schema. See docs/superpowers/specs/2026-06-11-databento-importer-issue-150-design.md.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from algua.data.contracts import DatabentoImportRequest, ImportRequest, ProviderBars
from algua.data.corpactions import CorporateAction, Dividend, Split, back_adjust
from algua.data.store import normalize_symbols

_RAW_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]


def _canon_symbol(value: str) -> str:
    return normalize_symbols([str(value)])[0]


def parse_databento_raw(path: Path) -> pd.DataFrame:
    """Parse one canonical per-symbol raw parquet into `[ts, open, high, low, close, volume]`.

    `ts` → tz-aware UTC midnight (naive localized; tz-aware non-UTC rejected; non-midnight rejected —
    this is the 1d importer). OHLCV finite; prices > 0; volume >= 0. Raises `ValueError` otherwise.
    """
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in _RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: raw file missing columns {missing}")
    out = frame[_RAW_COLUMNS].copy()
    ts = pd.to_datetime(out["ts"], errors="raise")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    elif str(ts.dt.tz) != "UTC":
        raise ValueError(
            f"{path.name}: raw ts is tz-aware non-UTC ({ts.dt.tz}); refusing to shift the session date"
        )
    if len(ts) and not bool((ts == ts.dt.normalize()).all()):
        raise ValueError(f"{path.name}: raw ts must be UTC midnight (1d); found a non-midnight value")
    out["ts"] = ts
    for col in [*_PRICE_COLUMNS, "volume"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    numeric = out[[*_PRICE_COLUMNS, "volume"]].to_numpy()
    if numeric.size and not np.all(np.isfinite(numeric)):
        raise ValueError(f"{path.name}: raw OHLCV must be finite (no NaN/inf)")
    if (out[_PRICE_COLUMNS] <= 0).to_numpy().any():
        raise ValueError(f"{path.name}: raw prices must be > 0")
    if (out["volume"] < 0).to_numpy().any():
        raise ValueError(f"{path.name}: raw volume must be >= 0")
    return out


_CA_REQUIRED = ["symbol", "ex_date", "kind", "value"]
_VALID_KINDS = {"split", "dividend"}


def _to_utc_midnight(value: object, fname: str, i: int) -> pd.Timestamp:
    ts = pd.Timestamp(value)  # raises on unparseable
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    elif str(ts.tz) != "UTC":
        raise ValueError(f"{fname} row {i}: ex_date is tz-aware non-UTC ({ts.tz})")
    if ts != ts.normalize():
        raise ValueError(f"{fname} row {i}: ex_date must be a date / UTC midnight, got {value!r}")
    return ts


def parse_databento_corp_actions(path: Path) -> dict[str, list[CorporateAction]]:
    """Parse the canonical CA-events parquet into `{symbol: [CorporateAction, ...]}`.

    Row-level validation (kind in {split,dividend}; value finite > 0; ex_date → UTC midnight) with
    messages naming the row, then source de-duplication: by `(symbol, event_id)` when the optional
    `event_id` column is present (all rows must then carry a non-blank id; same key + differing
    economics → raise), else by exact full row. Surviving events flow to `back_adjust`, which
    aggregates same-date ones.
    """
    frame = pd.read_parquet(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in _CA_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: corp-actions file missing columns {missing}")
    has_event_id = "event_id" in frame.columns

    # (symbol, ex_date, kind, value, event_id) per row, fully validated.
    parsed: list[tuple[str, pd.Timestamp, str, float, str | None]] = []
    for i, rec in enumerate(frame.to_dict("records")):
        if pd.isna(rec["symbol"]) or not str(rec["symbol"]).strip():
            raise ValueError(f"{path.name} row {i}: blank symbol")
        symbol = _canon_symbol(rec["symbol"])
        kind = str(rec["kind"]).strip().lower()
        if kind not in _VALID_KINDS:
            raise ValueError(f"{path.name} row {i}: unknown kind {rec['kind']!r} (expected split|dividend)")
        value = float(pd.to_numeric(rec["value"], errors="raise"))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{path.name} row {i} ({symbol} {kind}): value must be finite and > 0, got {rec['value']!r}"
            )
        ex_date = _to_utc_midnight(rec["ex_date"], path.name, i)
        event_id: str | None = None
        if has_event_id:
            if pd.isna(rec["event_id"]) or not str(rec["event_id"]).strip():
                raise ValueError(
                    f"{path.name} row {i}: event_id column present but this row has a blank/null id"
                )
            event_id = str(rec["event_id"]).strip()
        parsed.append((symbol, ex_date, kind, value, event_id))

    if has_event_id:
        econ_by_key: dict[tuple[str, str], tuple[pd.Timestamp, str, float]] = {}
        for symbol, ex_date, kind, value, event_id in parsed:
            key = (symbol, event_id)  # event_id is non-None here
            econ = (ex_date, kind, value)
            if key in econ_by_key:
                if econ_by_key[key] != econ:
                    raise ValueError(
                        f"{path.name}: event_id {event_id!r} for {symbol} has differing economics "
                        f"across rows: {econ_by_key[key]} vs {econ}"
                    )
            else:
                econ_by_key[key] = econ
        surviving = [(sym, *econ) for (sym, _eid), econ in econ_by_key.items()]
    else:
        seen: set[tuple[str, pd.Timestamp, str, float]] = set()
        surviving = []
        for symbol, ex_date, kind, value, _eid in parsed:
            key = (symbol, ex_date, kind, value)
            if key in seen:
                continue
            seen.add(key)
            surviving.append((symbol, ex_date, kind, value))

    events: dict[str, list[CorporateAction]] = {}
    for symbol, ex_date, kind, value in surviving:
        event: CorporateAction = (
            Split(ex_date=ex_date, ratio=value) if kind == "split" else Dividend(ex_date=ex_date, cash=value)
        )
        events.setdefault(symbol, []).append(event)
    return events


def _discover_raw(directory: Path) -> dict[str, Path]:
    """Map canonical symbol -> per-symbol `.parquet` path (stem = symbol). Dup-symbol → raise."""
    mapping: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() != ".parquet":
            continue
        symbol = _canon_symbol(path.stem)
        if symbol in mapping:
            raise ValueError(
                f"duplicate symbol {symbol!r} in {directory.name}: {mapping[symbol].name} and {path.name}"
            )
        mapping[symbol] = path
    return mapping


class DatabentoImporter:
    name = "databento"
    vendor_label = "databento"

    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]:
        if not isinstance(request, DatabentoImportRequest):
            raise ValueError("DatabentoImporter requires a DatabentoImportRequest")
        if request.timeframe != "1d":
            raise ValueError("intraday import not yet supported (1d only)")
        raw_map = _discover_raw(request.raw_dir)
        events_by_symbol = parse_databento_corp_actions(request.corp_actions_path)
        symbols = sorted(raw_map)
        if request.symbols is not None:
            wanted = set(normalize_symbols(list(request.symbols)))
            missing = sorted(wanted - set(symbols))
            if missing:
                raise ValueError(f"requested symbols with no files: {missing}")
            symbols = [s for s in symbols if s in wanted]
        for symbol in symbols:
            yield self._build_symbol(symbol, raw_map[symbol], events_by_symbol.get(symbol, []))

    def _build_symbol(
        self, symbol: str, raw_path: Path, events: list[CorporateAction]
    ) -> ProviderBars:
        raw = parse_databento_raw(raw_path)
        result = back_adjust(raw[["ts", "close"]], events)
        aligned = (
            len(result) == len(raw)
            and result["ts"].reset_index(drop=True).equals(raw["ts"].reset_index(drop=True))
        )
        if not aligned:
            raise ValueError(f"{symbol}: back_adjust output misaligned with raw bars")
        frame = raw.copy()
        frame["symbol"] = symbol
        frame["adj_close"] = result["adj_close"].to_numpy()
        frame = frame[["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
        return ProviderBars(
            frame=frame,
            source_metadata={"vendor": "databento", "symbol": symbol, "raw_file": raw_path.name},
        )
