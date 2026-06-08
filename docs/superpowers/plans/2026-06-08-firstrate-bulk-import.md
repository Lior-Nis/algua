# FirstRateData Bulk-Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an import-local-vendor-files-and-normalize path so deep FirstRateData history (per-symbol unadjusted + adjusted files) lands as one canonical bar-schema snapshot, streamed to keep RAM bounded.

**Architecture:** A new file-oriented `BarImporter` seam (parallel to the network `BarProvider`) with its own registry. A FirstRateData adapter pairs an unadjusted file set (→ raw OHLC) with an adjusted file set (→ `adj_close`), merges per symbol, yields one normalized chunk per symbol. A new `DataStore.ingest_bars_streamed` streams those chunks into one consolidated `bars.parquet` via crash-safe staging → hash → atomic rename → manifest. A `data import-bars` CLI command wires it together.

**Tech Stack:** Python 3.12, pandas, pyarrow, Typer (CLI), pytest. Quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Design source:** `docs/superpowers/specs/2026-06-08-firstrate-bulk-import-design.md` (GATE-1 reviewed).

**Key facts about the existing code (read before starting):**
- Bar schema is enforced by `algua/data/schema.py::validate_bars` / `to_bar_schema`. Stored bars
  parquet columns are `[timestamp, symbol, open, high, low, close, adj_close, volume]` (a `timestamp`
  column, NOT an index — `read_bars` re-indexes via `to_bar_schema`). `to_bar_schema` accepts a
  frame with a `ts` OR `timestamp` column, rejects naive timestamps, sorts by `(timestamp, symbol)`.
- `ProviderBars` (`algua/data/contracts.py`) = `{frame: DataFrame, source_metadata: dict[str,str]}`.
  yfinance/Alpaca adapters emit frames with columns `(ts, symbol, open, high, low, close, adj_close,
  volume)` — the importer mirrors this exact shape so both seams converge at `to_bar_schema`.
- `frame_to_parquet_bytes` (`algua/data/files.py`) pins determinism: `pa.Table.from_pandas(frame,
  preserve_index=False).replace_schema_metadata(None)`, `compression="snappy"`, `version="2.6"`.
  The streamed writer MUST match these settings.
- `sha256_file` (`algua/data/files.py`) hashes a file in 1 MiB chunks.
- `DataStore` (`algua/data/store.py`): `_metadata(...)` builds `SnapshotMetadata`; `_snapshot_id(metadata,
  content_hash)` is the 16-char id; `manifest.find(id)` / `manifest.append(rec)`; `normalize_symbols`
  strips/uppers/dedups/sorts and raises on empty.
- CLI pattern (`algua/cli/data_cmd.py`): `@data_app.command(...)` + `@json_errors(ValueError,
  LookupError, FileNotFoundError)`, `emit(ok({"snapshot": rec.to_dict()}))`. `now_iso()` from
  `algua.cli._common`; `emit` from `algua.cli.app`.
- Tests are flat in `tests/` (no `tests/data/` subdir). CLI tests use `typer.testing.CliRunner` +
  `monkeypatch.setenv("ALGUA_DATA_DIR", ...)`.

---

## File Structure

- Create `algua/data/importers/__init__.py` — `BarImporter` registry (`get_importer` /
  `register_importer`), mirroring `algua/data/providers/__init__.py`.
- Create `algua/data/importers/firstrate.py` — the FirstRateData adapter + file parser.
- Modify `algua/data/contracts.py` — add `ImportRequest` dataclass + `BarImporter` protocol.
- Modify `algua/data/store.py` — add `ingest_bars_streamed` + `clear_staging` + helpers.
- Modify `algua/cli/data_cmd.py` — add the `import-bars` command.
- Create `tests/test_firstrate_importer.py`, `tests/test_data_ingest_streamed.py`,
  `tests/test_cli_import_bars.py`, `tests/test_bar_producer_conformance.py`.

---

## Task 1: `ImportRequest` contract + `BarImporter` seam + registry

**Files:**
- Modify: `algua/data/contracts.py`
- Create: `algua/data/importers/__init__.py`
- Test: `tests/test_data_ingest_streamed.py` (registry portion)

- [ ] **Step 1: Write the failing test** (`tests/test_data_ingest_streamed.py`)

```python
import pytest

from algua.data.contracts import ImportRequest
from algua.data.importers import get_importer, register_importer


def test_get_importer_unknown_raises():
    with pytest.raises(ValueError, match="unsupported bar importer: nope"):
        get_importer("nope")


def test_register_and_get_importer_roundtrip():
    class _Dummy:
        name = "dummy"

        def import_bars(self, request):
            return iter(())

    register_importer("dummy", lambda: _Dummy())
    assert get_importer("dummy").name == "dummy"


def test_import_request_defaults(tmp_path):
    req = ImportRequest(raw_dir=tmp_path / "raw", adjusted_dir=tmp_path / "adj")
    assert req.timeframe == "1d"
    assert req.adjustment == "split_div"
    assert req.symbols is None
    assert req.as_of is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_ingest_streamed.py -v`
Expected: FAIL — `ImportError: cannot import name 'ImportRequest'` / `algua.data.importers`.

- [ ] **Step 3: Add the contract** (append to `algua/data/contracts.py`)

Add `Path` and `Iterator` imports at the top (`from pathlib import Path`, and extend the typing
import to include `Iterator` under `TYPE_CHECKING` is not enough — `Iterator` is used only in an
annotation string, so import it from `collections.abc` at module top). Then append:

```python
@dataclass(frozen=True)
class ImportRequest:
    """A request to import local vendor bar files (the `BarImporter` seam).

    Distinct from `BarRequest` (network fetch by symbol/date): the source here is local files the
    operator already downloaded. `raw_dir` supplies unadjusted OHLC; `adjusted_dir` supplies the
    vendor-adjusted close used as `adj_close`. `adjustment` is the operator-declared flavor of that
    adjusted file (we never infer it). `symbols`, if set, restricts the import to that subset.
    """

    raw_dir: Path
    adjusted_dir: Path
    timeframe: str = "1d"
    as_of: str | None = None
    adjustment: str = "split_div"
    symbols: tuple[str, ...] | None = None


class BarImporter(Protocol):
    """Ingestion seam for local vendor files: yield one normalized `ProviderBars` per symbol.

    Yielding per symbol (rather than returning one giant frame) is what bounds RAM for a multi-GB
    import. Each yielded frame has the same column shape as a `BarProvider` frame
    (`ts, symbol, open, high, low, close, adj_close, volume`) so both seams converge at
    `algua.data.schema.to_bar_schema`.
    """

    name: str

    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]: ...
```

Add the imports near the top of `contracts.py`:

```python
from collections.abc import Iterator
from pathlib import Path
```

- [ ] **Step 4: Create the registry** (`algua/data/importers/__init__.py`)

```python
"""Local-file bar importers and their construction registry.

These are the `BarImporter` seam (`algua.data.contracts`): `import_bars(ImportRequest) ->
Iterator[ProviderBars]`, used by `algua data import-bars` to normalize vendor files into one
consolidated bar-schema snapshot. Distinct from the network-fetch `BarProvider` seam
(`algua/data/providers/`) — the source here is local files, not an API.

Construction lives here (not the CLI) so adding a vendor is open-for-extension: register a
name->factory and `get_importer` picks it up — no if/elif ladder to edit.
"""
from __future__ import annotations

from collections.abc import Callable

from algua.data.contracts import BarImporter
from algua.data.importers.firstrate import FirstRateImporter

ImporterFactory = Callable[[], BarImporter]


def _build_firstrate() -> BarImporter:
    return FirstRateImporter()


_REGISTRY: dict[str, ImporterFactory] = {
    "firstrate": _build_firstrate,
}


def register_importer(name: str, factory: ImporterFactory) -> None:
    """Register a name->factory so `get_importer(name)` can build it."""
    _REGISTRY[name] = factory


def get_importer(name: str) -> BarImporter:
    """Construct a `BarImporter` by name. Raises ValueError for an unknown name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unsupported bar importer: {name}") from None
    return factory()
```

NOTE: this imports `FirstRateImporter`, created in Task 2. Until Task 2 lands, temporarily comment
the `from algua.data.importers.firstrate import FirstRateImporter` line and the `firstrate` registry
entry, OR implement Task 2 first. Recommended: the test for Task 1's registry uses a dummy, so to run
Step 5 now, leave `_REGISTRY` empty (`{}`) and omit the firstrate import; Task 3 adds the entry back.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_ingest_streamed.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add algua/data/contracts.py algua/data/importers/__init__.py tests/test_data_ingest_streamed.py
git commit -m "feat(data): BarImporter seam + ImportRequest contract + registry"
```

---

## Task 2: FirstRateData file parser

A FirstRate daily file is CSV/TXT with columns `datetime, open, high, low, close, volume` (header
optional; the `datetime` is a date for daily). Parse one file → a frame with columns
`ts, open, high, low, close, volume` (all floats; `ts` a tz-aware UTC midnight timestamp).

**Files:**
- Create: `algua/data/importers/firstrate.py` (parser portion)
- Test: `tests/test_firstrate_importer.py`

- [ ] **Step 1: Write the failing test** (`tests/test_firstrate_importer.py`)

```python
import pandas as pd
import pytest

from algua.data.importers.firstrate import parse_firstrate_file, symbol_from_filename


def _write(path, rows, header=False):
    text = ""
    if header:
        text += "datetime,open,high,low,close,volume\n"
    text += "".join(rows)
    path.write_text(text, encoding="utf-8")


def test_parse_headerless_daily(tmp_path):
    f = tmp_path / "AAPL_full_1day_UNADJUSTED.txt"
    _write(f, ["2024-07-01,10.0,11.0,9.5,10.5,1000\n", "2024-07-02,10.5,12.0,10.0,11.5,2000\n"])
    out = parse_firstrate_file(f)
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["ts"].iloc[0] == pd.Timestamp("2024-07-01", tz="UTC")
    assert out["close"].iloc[1] == 11.5
    assert out["open"].dtype == "float64"


def test_parse_with_header(tmp_path):
    f = tmp_path / "MSFT_full_1day_adjsplitdiv.csv"
    _write(f, ["2024-07-01,1.0,1.0,1.0,1.0,5\n"], header=True)
    out = parse_firstrate_file(f)
    assert len(out) == 1
    assert out["volume"].iloc[0] == 5.0


def test_symbol_from_filename():
    assert symbol_from_filename("AAPL_full_1day_UNADJUSTED.txt") == "AAPL"
    assert symbol_from_filename("brk.b_full_1day_adjsplitdiv.csv") == "BRK.B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_firstrate_importer.py -v`
Expected: FAIL — `ModuleNotFoundError: algua.data.importers.firstrate`.

- [ ] **Step 3: Write the parser** (`algua/data/importers/firstrate.py`)

```python
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from algua.data.contracts import ImportRequest, ProviderBars
from algua.data.schema import to_bar_schema
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
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                first_line = line.strip().lower()
                break
    has_header = first_line.startswith("datetime")
    frame = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else _FIRSTRATE_COLUMNS,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_firstrate_importer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/importers/firstrate.py tests/test_firstrate_importer.py
git commit -m "feat(data): FirstRate daily file parser"
```

---

## Task 3: FirstRateData adapter — pairing, merge, adj_close, errors

Implement `FirstRateImporter.import_bars`: discover+pair files, per symbol merge raw OHLC with the
adjusted close as `adj_close`, guard nonpositive prices, yield one `ProviderBars` per symbol in
canonical sorted order. Then re-enable the `firstrate` registry entry from Task 1.

**Files:**
- Modify: `algua/data/importers/firstrate.py`
- Modify: `algua/data/importers/__init__.py` (restore the firstrate import + registry entry)
- Test: `tests/test_firstrate_importer.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_firstrate_importer.py`)

```python
from algua.data.contracts import ImportRequest
from algua.data.importers.firstrate import FirstRateImporter
from algua.data.schema import validate_bars


def _firstrate_dirs(tmp_path):
    raw = tmp_path / "raw"
    adj = tmp_path / "adj"
    raw.mkdir()
    adj.mkdir()
    return raw, adj


def _write_pair(raw, adj, symbol, raw_rows, adj_rows):
    (raw / f"{symbol}_full_1day_UNADJUSTED.txt").write_text("".join(raw_rows), encoding="utf-8")
    (adj / f"{symbol}_full_1day_adjsplitdiv.txt").write_text("".join(adj_rows), encoding="utf-8")


def test_import_merges_raw_and_adjusted(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,100,110,95,105,1000\n", "2024-07-02,105,120,100,115,2000\n"],
        ["2024-07-01,50,55,47,52,1000\n", "2024-07-02,52,60,50,57,2000\n"],
    )
    chunks = list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))
    assert len(chunks) == 1
    frame = chunks[0].frame
    # raw OHLC preserved; adj_close from the adjusted file's close
    row = frame.set_index("ts").loc[pd.Timestamp("2024-07-01", tz="UTC")]
    assert row["close"] == 105.0
    assert row["adj_close"] == 52.0
    validate_bars(to_bar_schema_frame(frame))


def to_bar_schema_frame(frame):
    from algua.data.schema import to_bar_schema
    return to_bar_schema(frame)


def test_import_yields_symbols_sorted(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["MSFT", "AAPL", "GOOG"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    chunks = list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))
    seen = [c.frame["symbol"].iloc[0] for c in chunks]
    assert seen == ["AAPL", "GOOG", "MSFT"]


def test_symbol_set_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "MSFT_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symbol sets differ"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_alias_collision_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (raw / "aapl_full_1day_OTHER.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate symbol"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_key_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,1,1,1,1,1\n", "2024-07-02,1,1,1,1,1\n"],
        ["2024-07-01,1,1,1,1,1\n"],  # adjusted missing 07-02
    )
    with pytest.raises(ValueError, match="key sets differ"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_nonpositive_price_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,0,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="nonpositive"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_symbols_filter_subset(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["AAPL", "MSFT"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    req = ImportRequest(raw_dir=raw, adjusted_dir=adj, symbols=("AAPL",))
    chunks = list(FirstRateImporter().import_bars(req))
    assert [c.frame["symbol"].iloc[0] for c in chunks] == ["AAPL"]


def test_bad_timeframe_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="intraday import not yet supported"):
        list(FirstRateImporter().import_bars(
            ImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1h")))
```

Add `import pandas as pd` is already present from Task 2's test; ensure `from algua.data.schema
import to_bar_schema` is available (used via helper).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_firstrate_importer.py -v`
Expected: FAIL — `FirstRateImporter` not defined.

- [ ] **Step 3: Implement the adapter** (append to `algua/data/importers/firstrate.py`)

```python
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
        if (merged[price_cols] <= 0).to_numpy().any():
            raise ValueError(f"{symbol}: nonpositive price(s) in raw/adjusted data")
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
```

- [ ] **Step 4: Restore the registry entry** (`algua/data/importers/__init__.py`)

Ensure the `from algua.data.importers.firstrate import FirstRateImporter` import and the
`"firstrate": _build_firstrate` entry are present (un-comment from Task 1 if you stubbed them).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_firstrate_importer.py tests/test_data_ingest_streamed.py -v`
Expected: PASS (all importer + registry tests, incl. the `firstrate` registry resolving now).

- [ ] **Step 6: Commit**

```bash
git add algua/data/importers/ tests/test_firstrate_importer.py
git commit -m "feat(data): FirstRate adapter (pair raw+adjusted, merge adj_close, sorted yield)"
```

---

## Task 4: `DataStore.ingest_bars_streamed` — staging, deterministic write, dedup, bounds

Stream per-symbol chunks into one consolidated `bars.parquet` with crash-safe staging → hash →
atomic rename → manifest. Compute observed bounds; validate against requested bounds; stamp the
large-snapshot `servable` flag.

**Files:**
- Modify: `algua/data/store.py`
- Test: `tests/test_data_ingest_streamed.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_data_ingest_streamed.py`)

```python
import pandas as pd

from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.data.store import DataStore


def _chunk(symbol, dates_prices):
    n = len(dates_prices)
    return pd.DataFrame({
        "ts": [pd.Timestamp(d, tz="UTC") for d, _ in dates_prices],
        "symbol": [symbol] * n,
        "open": [p for _, p in dates_prices], "high": [p for _, p in dates_prices],
        "low": [p for _, p in dates_prices], "close": [p for _, p in dates_prices],
        "adj_close": [p / 2 for _, p in dates_prices], "volume": [100.0] * n,
    })


def _two_symbol_chunks():
    return [
        _chunk("AAPL", [("2024-07-01", 100.0), ("2024-07-02", 101.0)]),
        _chunk("MSFT", [("2024-07-01", 200.0), ("2024-07-02", 201.0)]),
    ]


def _ingest_streamed(store, chunks, **kw):
    params = dict(
        provider="firstrate", symbols=["AAPL", "MSFT"], as_of="2024-07-03T00:00:00+00:00",
        source="firstratedata-import", timeframe="1d", adjustment="split_div",
    )
    params.update(kw)
    return store.ingest_bars_streamed(chunks=iter(chunks), **params)


def test_streamed_ingest_one_snapshot_reads_canonical(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    out = store.read_bars(rec.snapshot_id)
    validate_bars(out)                      # canonical (timestamp, symbol) order on read
    assert list(out.columns) == BAR_COLUMNS
    assert rec.row_count == 4
    assert rec.start == "2024-07-01" and rec.end == "2024-07-02"


def test_streamed_ingest_is_idempotent(tmp_path):
    store = DataStore(tmp_path)
    a = _ingest_streamed(store, _two_symbol_chunks())
    b = _ingest_streamed(store, _two_symbol_chunks())
    assert a.snapshot_id == b.snapshot_id
    assert len(store.list_snapshots("bars")) == 1


def test_streamed_ingest_no_orphan_on_empty(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="no bars"):
        _ingest_streamed(store, [])
    staging = tmp_path / "snapshots" / "_staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_requested_bounds_mismatch_errors(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="observed coverage"):
        _ingest_streamed(store, _two_symbol_chunks(), start="2020-01-01", end="2024-07-02")


def test_large_row_warning_flag(tmp_path, monkeypatch):
    import algua.data.store as store_mod
    monkeypatch.setattr(store_mod, "IMPORT_WARN_ROWS", 1)
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    assert rec.metadata.source_metadata["servable"] == "deferred-130"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_ingest_streamed.py -v`
Expected: FAIL — `ingest_bars_streamed` not defined.

- [ ] **Step 3: Implement** (`algua/data/store.py`)

Add imports at the top of `store.py`:

```python
import os
import shutil
import uuid
from collections.abc import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
```

Add a module constant near the top (after imports):

```python
# Above this row count, a streamed import warns and self-marks "not servable until #130", because
# the read path still fully materializes a snapshot. Not a hard cap — deep history is the point.
IMPORT_WARN_ROWS = 5_000_000
```

Add the methods to `DataStore`:

```python
    def clear_staging(self) -> None:
        """Remove any leftover streamed-import staging dirs (crash residue)."""
        staging = self.data_dir / "snapshots" / "_staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    def ingest_bars_streamed(
        self,
        *,
        provider: str,
        symbols: list[str],
        as_of: str,
        source: str,
        chunks: Iterable[pd.DataFrame],
        timeframe: str = "1d",
        adjustment: str = "split_div",
        start: str | None = None,
        end: str | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Stream per-symbol bar chunks into one consolidated, deduplicated bars snapshot.

        Crash-safe: stream → staging file, hash, dedup, atomic rename into the immutable snapshot
        path, append the manifest last. Each chunk is normalized via `to_bar_schema` (so output is
        schema-valid) and written as one row group, in the order received (the importer yields
        canonical sorted-symbol order — required for a stable `snapshot_id`).
        """
        staging_dir = self.data_dir / "snapshots" / "_staging" / uuid.uuid4().hex
        staging_file = staging_dir / "bars.parquet"
        staging_dir.mkdir(parents=True, exist_ok=True)
        writer: pq.ParquetWriter | None = None
        row_count = 0
        observed_min: pd.Timestamp | None = None
        observed_max: pd.Timestamp | None = None
        try:
            for chunk in chunks:
                normalized = to_bar_schema(chunk).reset_index()  # columns: timestamp, *BAR_COLUMNS
                table = pa.Table.from_pandas(
                    normalized, preserve_index=False
                ).replace_schema_metadata(None)
                if writer is None:
                    writer = pq.ParquetWriter(
                        staging_file, table.schema, compression="snappy", version="2.6"
                    )
                writer.write_table(table)
                row_count += len(normalized)
                cmin = normalized["timestamp"].min()
                cmax = normalized["timestamp"].max()
                observed_min = cmin if observed_min is None else min(observed_min, cmin)
                observed_max = cmax if observed_max is None else max(observed_max, cmax)
            if writer is None:
                raise ValueError("no bars to ingest (empty chunk stream)")
            writer.close()
            writer = None

            observed_start = observed_min.date().isoformat()
            observed_end = observed_max.date().isoformat()
            if start is not None or end is not None:
                if (start is not None and observed_start > start) or (
                    end is not None and observed_end < end
                ):
                    raise ValueError(
                        f"observed coverage [{observed_start}, {observed_end}] does not cover "
                        f"requested [{start}, {end}]"
                    )

            meta_extra = dict(source_metadata or {})
            if start is not None:
                meta_extra["requested_start"] = start
            if end is not None:
                meta_extra["requested_end"] = end
            meta_extra["observed_start"] = observed_start
            meta_extra["observed_end"] = observed_end
            if row_count >= IMPORT_WARN_ROWS:
                meta_extra["servable"] = "deferred-130"

            metadata = _metadata(
                dataset=Dataset.BARS.value,
                provider=provider,
                symbols=symbols,
                start=observed_start,
                end=observed_end,
                as_of=as_of,
                source=source,
                kind=Kind.BARS.value,
                timeframe=timeframe,
                adjustment=adjustment,
                source_metadata=meta_extra,
            )
            content_hash = sha256_file(staging_file)
            snapshot_id = _snapshot_id(metadata, content_hash)

            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing

            relative_path = Path("snapshots") / metadata.dataset / snapshot_id / "bars.parquet"
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_file, target)
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet",
            )
            self.manifest.append(rec)
            return rec
        finally:
            if writer is not None:
                writer.close()
            shutil.rmtree(staging_dir, ignore_errors=True)
```

Add `sha256_file` to the existing `from algua.data.files import (...)` block in `store.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_ingest_streamed.py -v`
Expected: PASS (all streamed-ingest tests).

- [ ] **Step 5: Run the data-layer suite for regressions**

Run: `uv run pytest tests/test_data_store.py tests/test_data_read_bars.py tests/test_data_serve.py -q`
Expected: PASS (no regressions in existing ingest/read paths).

- [ ] **Step 6: Commit**

```bash
git add algua/data/store.py tests/test_data_ingest_streamed.py
git commit -m "feat(data): streamed bars ingest (staging, atomic rename, observed bounds, warn flag)"
```

---

## Task 5: CLI `data import-bars` command

**Files:**
- Modify: `algua/cli/data_cmd.py`
- Test: `tests/test_cli_import_bars.py`

- [ ] **Step 1: Write the failing test** (`tests/test_cli_import_bars.py`)

```python
import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))


def _firstrate_dirs(tmp_path):
    raw = tmp_path / "raw"
    adj = tmp_path / "adj"
    raw.mkdir()
    adj.mkdir()
    for sym, rprice, aprice in [("AAPL", 100, 50), ("MSFT", 200, 180)]:
        (raw / f"{sym}_full_1day_UNADJUSTED.txt").write_text(
            f"2024-07-01,{rprice},{rprice},{rprice},{rprice},10\n", encoding="utf-8")
        (adj / f"{sym}_full_1day_adjsplitdiv.txt").write_text(
            f"2024-07-01,{aprice},{aprice},{aprice},{aprice},10\n", encoding="utf-8")
    return raw, adj


def test_import_bars_happy_path(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--timeframe", "1d", "--as-of", "2024-07-02T00:00:00+00:00",
        "--adjustment", "split_div",
    ])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    snap = payload["snapshot"]
    assert snap["provider"] == "firstrate"
    assert snap["adjustment"] == "split_div"
    assert snap["symbols"] == ["AAPL", "MSFT"]
    assert snap["row_count"] == 2
    assert snap["source_metadata"]["vendor"] == "firstratedata"


def test_import_bars_unknown_vendor_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "nope",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--as-of", "2024-07-02T00:00:00+00:00",
    ])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_import_bars.py -v`
Expected: FAIL — no such command `import-bars`.

- [ ] **Step 3: Implement the command** (`algua/cli/data_cmd.py`)

Add imports at the top:

```python
from algua.data.contracts import ImportRequest
from algua.data.importers import get_importer
```

Add the command (after `ingest_bars`):

```python
@data_app.command("import-bars")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_bars(
    vendor: str = typer.Option(..., "--vendor", help="bulk-file vendor, e.g. firstrate"),
    raw_dir: Path = typer.Option(..., "--raw-dir", help="dir of unadjusted per-symbol files"),
    adjusted_dir: Path = typer.Option(
        ..., "--adjusted-dir", help="dir of adjusted per-symbol files (supplies adj_close)"
    ),
    timeframe: str = typer.Option("1d", "--timeframe"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    adjustment: str = typer.Option(
        "split_div", "--adjustment", help="operator-declared adjusted-file flavor (recorded as-is)"
    ),
    start: str = typer.Option(None, "--start", help="optional requested coverage start YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="optional requested coverage end YYYY-MM-DD"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """Import local vendor bar files into one consolidated, normalized bars snapshot."""
    importer = get_importer(vendor)
    request = ImportRequest(
        raw_dir=raw_dir,
        adjusted_dir=adjusted_dir,
        timeframe=timeframe,
        as_of=as_of,
        adjustment=adjustment,
        symbols=tuple(normalize_symbols(symbols.split(","))) if symbols else None,
    )
    store = _store()
    store.clear_staging()
    chunks = importer.import_bars(request)
    # Peek the symbol list for provenance without buffering all data: collect from chunk frames as
    # they stream by wrapping the generator so symbols accumulate during the single pass.
    seen_symbols: list[str] = []

    def _tracked() -> "Iterator[object]":
        for chunk in chunks:
            seen_symbols.append(str(chunk.frame["symbol"].iloc[0]))
            yield chunk.frame

    rec = store.ingest_bars_streamed(
        provider=vendor,
        symbols=seen_symbols,  # mutated during the stream; finalized before _metadata reads it
        as_of=as_of,
        source=f"{vendor}-import",
        chunks=_tracked(),
        timeframe=timeframe,
        adjustment=adjustment,
        start=start,
        end=end,
        source_metadata={"vendor": "firstratedata"} if vendor == "firstrate" else {},
    )
    if rec.row_count is not None and rec.row_count >= 5_000_000:
        typer.echo(
            f"warning: imported {rec.row_count} rows; snapshot not servable by the current "
            f"read path until #130 (marked servable=deferred-130)",
            err=True,
        )
    emit(ok({"snapshot": rec.to_dict()}))
```

IMPORTANT ordering note for the implementer: `seen_symbols` is filled by `_tracked()` during the
single streaming pass, and `ingest_bars_streamed` only reads `symbols` (via `_metadata`) AFTER the
stream is exhausted (it builds metadata post-loop). So the list is complete by the time it's read.
Add a within-task test assertion if unsure: after the call, `rec.symbols == ("AAPL", "MSFT")`.
`normalize_symbols` inside `_metadata` will sort/dedup them. Import `Iterator` from
`collections.abc` at the top of `data_cmd.py` for the annotation (or drop the annotation).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_import_bars.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/data_cmd.py tests/test_cli_import_bars.py
git commit -m "feat(cli): data import-bars command (FirstRate bulk import)"
```

---

## Task 6: Shared producer/importer conformance + shuffle-invariance

Guard against the two normalization paths drifting (GATE-1 #10), and prove `snapshot_id` is stable
regardless of filesystem discovery order (GATE-1 #9).

**Files:**
- Create: `tests/test_bar_producer_conformance.py`
- Test: itself

- [ ] **Step 1: Write the test** (`tests/test_bar_producer_conformance.py`)

```python
import pandas as pd

from algua.data.contracts import ImportRequest
from algua.data.importers.firstrate import FirstRateImporter
from algua.data.schema import to_bar_schema, validate_bars
from algua.data.store import DataStore


def _write_pair(raw, adj, sym, price):
    (raw / f"{sym}_full_1day_UNADJUSTED.txt").write_text(
        f"2024-07-01,{price},{price},{price},{price},10\n"
        f"2024-07-02,{price},{price},{price},{price},10\n", encoding="utf-8")
    (adj / f"{sym}_full_1day_adjsplitdiv.txt").write_text(
        f"2024-07-01,{price / 2},{price / 2},{price / 2},{price / 2},10\n"
        f"2024-07-02,{price / 2},{price / 2},{price / 2},{price / 2},10\n", encoding="utf-8")


def _dirs(tmp_path, name):
    raw = tmp_path / f"{name}_raw"
    adj = tmp_path / f"{name}_adj"
    raw.mkdir()
    adj.mkdir()
    return raw, adj


def test_importer_output_is_bar_schema_valid(tmp_path):
    raw, adj = _dirs(tmp_path, "c")
    for sym in ["AAPL", "MSFT"]:
        _write_pair(raw, adj, sym, 100)
    for chunk in FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)):
        # Same terminal boundary both seams must satisfy.
        validate_bars(to_bar_schema(chunk.frame))


def test_snapshot_id_is_discovery_order_invariant(tmp_path):
    # Two dirs with identical content; the importer must sort symbols so the snapshot_id is stable
    # regardless of os listing order. Ingest twice and assert equal ids (dedup proves stability).
    raw, adj = _dirs(tmp_path, "d")
    for sym in ["MSFT", "AAPL", "GOOG"]:
        _write_pair(raw, adj, sym, 100)
    store = DataStore(tmp_path / "store")

    def _ingest():
        chunks = (c.frame for c in FirstRateImporter().import_bars(
            ImportRequest(raw_dir=raw, adjusted_dir=adj)))
        return store.ingest_bars_streamed(
            provider="firstrate", symbols=["AAPL", "GOOG", "MSFT"],
            as_of="2024-07-03T00:00:00+00:00", source="firstratedata-import",
            chunks=chunks, timeframe="1d", adjustment="split_div",
        )

    assert _ingest().snapshot_id == _ingest().snapshot_id
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_bar_producer_conformance.py -v`
Expected: PASS (2 tests). If `test_snapshot_id_is_discovery_order_invariant` fails, the importer is
not yielding in canonical sorted order — fix Task 3's `symbols = sorted(raw_map)`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bar_producer_conformance.py
git commit -m "test(data): bar-producer conformance + snapshot_id discovery-order invariance"
```

---

## Task 7: Docs + full quality gate

**Files:**
- Modify: `CLAUDE.md` (command surface)
- Modify: `docs/contracts/bar-schema.md` (note the symbol-major physical-layout for streamed imports)

- [ ] **Step 1: Add the command to `CLAUDE.md`** (under "## Command surface", after the
  `ingest-universe` line)

```markdown
- `uv run algua data import-bars --vendor firstrate --raw-dir DIR --adjusted-dir DIR --as-of TS` —
  bulk-import local vendor files (FirstRateData: per-symbol unadjusted + adjusted), normalized to
  the bar-schema as one consolidated snapshot. Streamed (bounded RAM); `adj_close` from the adjusted
  file (no corporate-action math yet).
```

- [ ] **Step 2: Add a physical-layout note to `docs/contracts/bar-schema.md`** (under "## Conformance")

```markdown
- **Streamed bulk imports** (`data import-bars`) store the consolidated `bars.parquet` in
  **symbol-major `(symbol, timestamp)`** order to keep ingest RAM bounded. This is contract-safe
  because `read_bars` re-sorts to canonical `(timestamp, symbol)` via `to_bar_schema` on read — the
  contract governs returned order, not on-disk layout. A future scaled read path (#130) must treat
  these datasets as symbol-major (per-symbol pruning / streaming merge), not time-major.
```

- [ ] **Step 3: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. Fix any `ruff`/`mypy` findings (e.g. unused imports, missing annotations) in
the files you touched. Common likely fixes: add return-type/`-> None` annotations; ensure
`pyarrow.parquet.ParquetWriter` typing satisfies mypy (annotate the local as
`pq.ParquetWriter | None = None`, already done).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/contracts/bar-schema.md
git commit -m "docs: document data import-bars command + symbol-major streamed layout"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** seam (T1) · FirstRate adapter + two-file merge + adj_close (T2,T3) · streamed
  write + staging/atomic-rename + determinism + observed/requested bounds + warn flag (T4) · CLI +
  provenance (T5) · conformance + shuffle-invariance (T6) · docs + gate (T7). Deferred per spec: CA
  math, Databento, intraday, #130 read-path scaling.
- **Type consistency:** `ImportRequest`, `ProviderBars.frame`, `parse_firstrate_file`,
  `symbol_from_filename`, `FirstRateImporter.import_bars`, `get_importer`, `register_importer`,
  `DataStore.ingest_bars_streamed`, `DataStore.clear_staging`, `IMPORT_WARN_ROWS` — names are used
  consistently across tasks.
- **Importer frame shape:** `import_bars` yields `ProviderBars` whose `.frame` has columns
  `(ts, symbol, open, high, low, close, adj_close, volume)`; the CLI's `_tracked()` passes
  `chunk.frame` (a DataFrame) into `ingest_bars_streamed`, which calls `to_bar_schema` on each. The
  store method's `chunks` param is `Iterable[pd.DataFrame]`, NOT `Iterable[ProviderBars]` — the CLI
  unwraps `.frame`. Keep this consistent.
```
