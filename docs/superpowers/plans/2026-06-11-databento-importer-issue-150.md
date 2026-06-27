# Databento Importer (#150) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `DatabentoImporter` under the existing `BarImporter` seam that normalizes a canonical local parquet form (per-symbol raw OHLC + a single corporate-action events file) into the bar-schema, computing `adj_close` via the #149 CA engine.

**Architecture:** Split `ImportRequest` into a `kw_only` common base + per-vendor subtypes (`FirstRateImportRequest`, `DatabentoImportRequest`); importers narrow via `isinstance` inside `import_bars`. New `algua/data/importers/databento.py` with two pure parsers (raw OHLC, CA events) + the importer; registered in the importer registry; CLI gains `--corp-actions` + a per-vendor arm with `corp_actions_sha256` provenance. Full design: `docs/superpowers/specs/2026-06-11-databento-importer-issue-150-design.md`.

**Tech Stack:** Python, pandas, numpy, pyarrow (parquet), typer (CLI), pytest. Reuses `algua.data.corpactions.back_adjust` (#149) and `DataStore.ingest_bars_streamed`.

**Quality gate (run before each commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

### Task 1: `ImportRequest` → per-vendor request types (refactor; keep everything green)

This is a behavior-preserving refactor: existing FirstRate/ingest/conformance tests must stay green after a mechanical rename. No new test.

**Files:**
- Modify: `algua/data/contracts.py`
- Modify: `algua/data/importers/firstrate.py`
- Modify: `tests/test_firstrate_importer.py`, `tests/test_bar_producer_conformance.py`, `tests/test_data_ingest_streamed.py`

- [ ] **Step 1: Replace `ImportRequest` in `algua/data/contracts.py`**

Replace the existing `ImportRequest` dataclass with the base + two subtypes, and update the `BarImporter` Protocol docstring. (Leave `BarRequest`, `ProviderBars`, `BarProvider` unchanged.)

```python
@dataclass(frozen=True, kw_only=True)
class ImportRequest:
    """Common fields for a local-file bulk import (the `BarImporter` seam).

    Vendor-specific source inputs live on the per-vendor subclasses (`FirstRateImportRequest`,
    `DatabentoImportRequest`). `raw_dir` supplies unadjusted OHLC; `as_of` is the operator's PIT
    stamp; `adjustment` is the operator-declared flavor recorded as-is; `symbols`, if set, restricts
    the import. Keyword-only so a required subclass field can follow the base's defaulted ones.
    """

    raw_dir: Path
    timeframe: str = "1d"
    as_of: str | None = None
    adjustment: str = "split_div"
    symbols: tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class FirstRateImportRequest(ImportRequest):
    """FirstRate import: `adjusted_dir` supplies the vendor-adjusted close taken as `adj_close`."""

    adjusted_dir: Path


@dataclass(frozen=True, kw_only=True)
class DatabentoImportRequest(ImportRequest):
    """Databento import: `corp_actions_path` is one parquet of split/dividend events. There is no
    vendor adjusted column — `adj_close` is computed via the #149 CA engine."""

    corp_actions_path: Path
```

In the `BarImporter` Protocol docstring, append:

```
    Callers pair an importer with its matching `ImportRequest` subtype. The registry erases the
    subtype, so each importer narrows via `isinstance` inside `import_bars` and raises on a mismatch
    (the Protocol method keeps the base `ImportRequest` parameter — narrowing the signature would
    violate parameter contravariance).
```

- [ ] **Step 2: Narrow `FirstRateImporter.import_bars` in `algua/data/importers/firstrate.py`**

Update the import line and add the narrowing guard as the first statement of `import_bars`:

```python
from algua.data.contracts import FirstRateImportRequest, ImportRequest, ProviderBars
```
```python
    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]:
        if not isinstance(request, FirstRateImportRequest):
            raise ValueError("FirstRateImporter requires a FirstRateImportRequest")
        if request.timeframe != "1d":
            raise ValueError("intraday import not yet supported (1d only)")
        raw_map = _discover(request.raw_dir)
        adj_map = _discover(request.adjusted_dir)
        # ... rest unchanged ...
```

- [ ] **Step 3: Rename construction sites in the three test files**

In `tests/test_firstrate_importer.py`, `tests/test_bar_producer_conformance.py`, `tests/test_data_ingest_streamed.py`: change every `ImportRequest(` that passes `adjusted_dir=` to `FirstRateImportRequest(` (all are already keyword-style), and update each file's import from `from algua.data.contracts import ... ImportRequest ...` to import `FirstRateImportRequest` instead (or in addition, if `ImportRequest` is still referenced — it is not, after the rename).

Run: `cd <repo>; grep -rn "ImportRequest(" tests/` — every remaining hit must be `FirstRateImportRequest(` or `DatabentoImportRequest(`.

- [ ] **Step 4: Run the affected tests — must stay green**

Run: `uv run pytest tests/test_firstrate_importer.py tests/test_bar_producer_conformance.py tests/test_data_ingest_streamed.py -q`
Expected: PASS (behavior unchanged; only the request type name changed).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. (The CLI at `algua/cli/data_cmd.py:121` still constructs `ImportRequest(... adjusted_dir=...)` — it is updated in Task 5. **If the gate's mypy/pytest fails only on `data_cmd.py` here, that is expected**; to keep Task 1 self-contained and green, also apply Task 5's CLI change now, or accept that the full gate goes green at the end of Task 5. Recommended: do the minimal CLI fix now — change line 121's `ImportRequest(` to `FirstRateImportRequest(` and add the import — so Task 1 leaves the gate green; Task 5 then adds the databento arm.)

Minimal CLI fix for Task 1 (in `algua/cli/data_cmd.py`): add `FirstRateImportRequest` to the contracts import and change the `import_bars` body's `request = ImportRequest(` to `request = FirstRateImportRequest(` (it already passes `adjusted_dir=adjusted_dir`).

- [ ] **Step 6: Commit**

```bash
git add algua/data/contracts.py algua/data/importers/firstrate.py algua/cli/data_cmd.py tests/test_firstrate_importer.py tests/test_bar_producer_conformance.py tests/test_data_ingest_streamed.py
git commit -m "refactor(data): per-vendor ImportRequest types (kw_only base + FirstRate subtype) (#150)"
```

---

### Task 2: Databento raw-OHLC parser

**Files:**
- Create: `algua/data/importers/databento.py`
- Test: `tests/test_databento_importer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_databento_importer.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algua.data.importers.databento import parse_databento_raw


def _write_raw(path: Path, ts, closes, *, opens=None, highs=None, lows=None, vols=None) -> None:
    n = len(closes)
    opens = opens or [float(c) for c in closes]
    highs = highs or [float(c) + 1 for c in closes]
    lows = lows or [float(c) - 1 for c in closes]
    vols = vols or [100.0] * n
    pd.DataFrame(
        {"ts": ts, "open": opens, "high": highs, "low": lows, "close": [float(c) for c in closes],
         "volume": vols}
    ).to_parquet(path)


def test_parse_raw_utc_and_naive(tmp_path):
    # tz-aware UTC midnight passes; tz-naive midnight is localized to UTC.
    p1 = tmp_path / "AAPL.parquet"
    _write_raw(p1, pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), [100, 110, 120])
    out = parse_databento_raw(p1)
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dt.tz) == "UTC"

    p2 = tmp_path / "MSFT.parquet"
    _write_raw(p2, pd.date_range("2024-01-01", periods=3, freq="D"), [100, 110, 120])  # naive
    assert str(parse_databento_raw(p2)["ts"].dt.tz) == "UTC"


def test_parse_raw_rejects_non_utc_tz(tmp_path):
    p = tmp_path / "AAPL.parquet"
    _write_raw(p, pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern"), [100, 110])
    with pytest.raises(ValueError, match="non-UTC"):
        parse_databento_raw(p)


def test_parse_raw_rejects_non_midnight(tmp_path):
    p = tmp_path / "AAPL.parquet"
    ts = pd.to_datetime(["2024-01-01 16:00", "2024-01-02 16:00"])  # naive, non-midnight
    _write_raw(p, ts, [100, 110])
    with pytest.raises(ValueError, match="midnight"):
        parse_databento_raw(p)


def test_parse_raw_rejects_bad_ohlcv(tmp_path):
    p = tmp_path / "AAPL.parquet"
    _write_raw(p, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 110],
               highs=[float("inf"), 111])
    with pytest.raises(ValueError, match="finite"):
        parse_databento_raw(p)
    p2 = tmp_path / "MSFT.parquet"
    _write_raw(p2, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 0])  # close 0
    with pytest.raises(ValueError, match="> 0"):
        parse_databento_raw(p2)
    p3 = tmp_path / "TSLA.parquet"
    _write_raw(p3, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 110],
               vols=[100.0, -1.0])
    with pytest.raises(ValueError, match="volume"):
        parse_databento_raw(p3)


def test_parse_raw_missing_column(tmp_path):
    p = tmp_path / "AAPL.parquet"
    pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=1, tz="UTC"), "close": [100.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing columns"):
        parse_databento_raw(p)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_databento_importer.py -q`
Expected: FAIL — `ModuleNotFoundError: algua.data.importers.databento`.

- [ ] **Step 3: Create `algua/data/importers/databento.py` with the raw parser**

```python
"""Databento canonical-parquet importer (#150).

Normalizes a canonical local form — per-symbol raw OHLC parquet + one corporate-action events
parquet — into the bar-schema, computing `adj_close` via the #149 CA engine. NOT a parser of
Databento's native binary format (int-scaled prices / instrument_id / ns ts); the operator conforms
an export to this schema. See docs/superpowers/specs/2026-06-11-databento-importer-issue-150-design.md.
"""
from __future__ import annotations

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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_databento_importer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/importers/databento.py tests/test_databento_importer.py
git commit -m "feat(data): Databento canonical raw-OHLC parser (#150)"
```

---

### Task 3: Databento CA-events parser (typed events + dedup hygiene)

> **Superseded by #264 (2026-06-27):** the no-`event_id` exact-full-row-duplicate path below now
> **raises** instead of silently dropping (silent drop under-adjusted `adj_close`). The sample
> test (`test_ca_no_event_id_exact_dup_dropped`), parser docstring, and `continue` in the sample
> implementation in this historical plan reflect the original drop design — see the current code,
> the design spec, and `test_ca_no_event_id_exact_dup_raises` for the shipped raise behavior.

**Files:**
- Modify: `algua/data/importers/databento.py`
- Test: `tests/test_databento_importer.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_databento_importer.py
from algua.data.corpactions import Dividend, Split
from algua.data.importers.databento import parse_databento_corp_actions


def _write_ca(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path)


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_ca_parse_split_and_dividend(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
        {"symbol": "msft", "ex_date": "2024-01-10", "kind": "Dividend ", "value": 1.0},  # case/space
    ])
    ev = parse_databento_corp_actions(p)
    assert ev["AAPL"] == [Split(ex_date=_utc("2024-01-03"), ratio=2.0),
                          Dividend(ex_date=_utc("2024-02-01"), cash=0.5)] or \
           sorted([type(e).__name__ for e in ev["AAPL"]]) == ["Dividend", "Split"]
    assert ev["MSFT"] == [Dividend(ex_date=_utc("2024-01-10"), cash=1.0)]


def test_ca_unknown_kind_and_bad_value(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "merger", "value": 1.0}])
    with pytest.raises(ValueError, match="unknown kind"):
        parse_databento_corp_actions(p)
    p2 = tmp_path / "ca2.parquet"
    _write_ca(p2, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "dividend", "value": 0.0}])
    with pytest.raises(ValueError, match="value"):
        parse_databento_corp_actions(p2)


def test_ca_non_midnight_ex_date_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [{"symbol": "AAPL", "ex_date": "2024-01-03 12:00", "kind": "split", "value": 2.0}])
    with pytest.raises(ValueError, match="midnight|UTC"):
        parse_databento_corp_actions(p)


def test_ca_same_date_two_dividends_both_kept(tmp_path):
    # regular + special dividend on one ex-date with distinct event_id -> both kept (engine sums).
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "r1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "s1"},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 2


def test_ca_event_id_dedup_and_conflict(tmp_path):
    # duplicate (symbol, event_id) identical economics -> one; differing economics -> raise.
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 1

    p2 = tmp_path / "ca2.parquet"
    _write_ca(p2, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.9, "event_id": "d1"},
    ])
    with pytest.raises(ValueError, match="differing economics"):
        parse_databento_corp_actions(p2)


def test_ca_event_id_blank_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-03-01", "kind": "dividend", "value": 0.5, "event_id": None},
    ])
    with pytest.raises(ValueError, match="event_id"):
        parse_databento_corp_actions(p)


def test_ca_same_event_id_across_symbols_kept(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "x"},
        {"symbol": "MSFT", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "x"},
    ])
    ev = parse_databento_corp_actions(p)
    assert len(ev["AAPL"]) == 1 and len(ev["MSFT"]) == 1


def test_ca_no_event_id_exact_dup_dropped(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_databento_importer.py -k corp_action -q` (and the `test_ca_*`)
Expected: FAIL — `parse_databento_corp_actions` undefined.

- [ ] **Step 3: Add the CA parser to `algua/data/importers/databento.py`**

Add `import math` to the imports, then:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_databento_importer.py -q`
Expected: PASS (all Task-2 + Task-3 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/importers/databento.py tests/test_databento_importer.py
git commit -m "feat(data): Databento CA-events parser + event_id dedup hygiene (#150)"
```

---

### Task 4: `DatabentoImporter` + registry

**Files:**
- Modify: `algua/data/importers/databento.py`
- Modify: `algua/data/importers/__init__.py`
- Test: `tests/test_databento_importer.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_databento_importer.py
from algua.data.contracts import DatabentoImportRequest, FirstRateImportRequest
from algua.data.importers import get_importer
from algua.data.importers.databento import DatabentoImporter


def _run(raw_dir: Path, ca: Path, **kw):
    req = DatabentoImportRequest(raw_dir=raw_dir, corp_actions_path=ca, **kw)
    return list(DatabentoImporter().import_bars(req))


def test_importer_split_adj_close(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
               [100, 110, 50, 55])  # 2:1 split on bar index 2
    _write_ca(ca, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}])
    [pb] = _run(raw, ca)
    assert list(pb.frame.columns) == ["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    np.testing.assert_allclose(pb.frame["adj_close"].to_numpy(), [50, 55, 50, 55])
    assert pb.source_metadata["vendor"] == "databento"


def test_importer_same_date_regular_plus_special_dividend(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
               [100, 110, 120, 130])  # P_prev for ex 01-03 = close[1] = 110
    _write_ca(ca, [
        {"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "dividend", "value": 2.0, "event_id": "r"},
        {"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "dividend", "value": 3.0, "event_id": "s"},
    ])
    [pb] = _run(raw, ca)
    m = (110 - 5) / 110  # summed cash 5, NOT cross-term
    np.testing.assert_allclose(pb.frame["adj_close"].to_numpy()[0], 100 * m)


def test_importer_multi_symbol_and_no_events(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), [100, 110, 120])
    _write_raw(raw / "MSFT.parquet", pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), [10, 11, 12])
    _write_ca(ca, [{"symbol": "AAPL", "ex_date": "2024-01-02", "kind": "split", "value": 2.0}])
    pbs = {pb.frame["symbol"].iloc[0]: pb for pb in _run(raw, ca)}
    assert set(pbs) == {"AAPL", "MSFT"}
    np.testing.assert_allclose(pbs["MSFT"].frame["adj_close"].to_numpy(), [10, 11, 12])  # no events → identity
    # each symbol is exactly one chunk
    assert all(pb.frame["symbol"].nunique() == 1 for pb in pbs.values())


def test_importer_rejects_wrong_request_and_intraday(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"), [100])
    _write_ca(ca, [])
    with pytest.raises(ValueError, match="DatabentoImportRequest"):
        list(DatabentoImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=raw)))
    with pytest.raises(ValueError, match="1d only"):
        _run(raw, ca, timeframe="1h")


def test_importer_dup_symbol_files(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=1, tz="UTC"), [100])
    _write_raw(raw / "aapl.parquet", pd.date_range("2024-01-01", periods=1, tz="UTC"), [100])
    _write_ca(ca, [])
    with pytest.raises(ValueError, match="duplicate symbol"):
        _run(raw, ca)


def test_registry_has_databento():
    assert isinstance(get_importer("databento"), DatabentoImporter)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_databento_importer.py -q`
Expected: FAIL — `DatabentoImporter` / registry entry undefined.

- [ ] **Step 3: Add the discover helper + `DatabentoImporter` to `databento.py`**

```python
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
```

- [ ] **Step 4: Register in `algua/data/importers/__init__.py`**

```python
from algua.data.importers.databento import DatabentoImporter
from algua.data.importers.firstrate import FirstRateImporter
```
```python
def _build_databento() -> BarImporter:
    return DatabentoImporter()


_REGISTRY: dict[str, ImporterFactory] = {
    "firstrate": _build_firstrate,
    "databento": _build_databento,
}
```

- [ ] **Step 5: Run to verify pass + gate**

Run: `uv run pytest tests/test_databento_importer.py -q` → PASS.
Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` → all green (note `lint-imports`: `databento.py` imports only `algua.data.*` siblings).

- [ ] **Step 6: Commit**

```bash
git add algua/data/importers/databento.py algua/data/importers/__init__.py tests/test_databento_importer.py
git commit -m "feat(data): DatabentoImporter wiring back_adjust + registry entry (#150)"
```

---

### Task 5: CLI `--corp-actions` + per-vendor arm + provenance

**Files:**
- Modify: `algua/cli/data_cmd.py`
- Test: `tests/test_cli_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_cli_data.py  (follow this file's existing runner/fixture conventions —
# typer.testing.CliRunner on the data app, reading JSON from stdout; mirror the existing
# firstrate import-bars test if present)
import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from algua.cli.app import app

runner = CliRunner()


def _write_raw_p(path, ts, closes):
    pd.DataFrame({"ts": ts, "open": closes, "high": [c + 1 for c in closes],
                  "low": [c - 1 for c in closes], "close": closes,
                  "volume": [100.0] * len(closes)}).to_parquet(path)


def test_cli_databento_import(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "store"))  # match this repo's data-dir env
    raw = tmp_path / "raw"; raw.mkdir(); ca = tmp_path / "ca.parquet"
    _write_raw_p(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
                 [100.0, 110.0, 50.0, 55.0])
    pd.DataFrame([{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}]).to_parquet(ca)
    res = runner.invoke(app, ["data", "import-bars", "--vendor", "databento", "--raw-dir", str(raw),
                              "--corp-actions", str(ca), "--as-of", "2024-06-01T00:00:00Z"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    # provenance carries the CA hash
    snap = payload["data"]["snapshot"]
    assert "corp_actions_sha256" in json.dumps(snap)


def test_cli_databento_requires_corp_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "store"))
    raw = tmp_path / "raw"; raw.mkdir()
    _write_raw_p(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"), [100.0])
    res = runner.invoke(app, ["data", "import-bars", "--vendor", "databento", "--raw-dir", str(raw),
                              "--as-of", "2024-06-01T00:00:00Z"])
    assert res.exit_code != 0
    assert "corp-actions" in res.output
```

> **Implementer note:** before writing, open `tests/test_cli_data.py` and reuse its existing
> patterns (the CliRunner setup, the data-dir env var name, and the JSON-envelope keys `ok`/`data`).
> If the repo's env var or envelope differs from the guesses above, match the file — do not invent.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_data.py -k databento -q`
Expected: FAIL — `--corp-actions` is not a known option / databento arm missing.

- [ ] **Step 3: Update `algua/cli/data_cmd.py`**

Add `import hashlib` at the top. Ensure the contracts import includes both request types:
```python
from algua.data.contracts import (
    BarProvider, BarRequest, DatabentoImportRequest, FirstRateImportRequest, ImportRequest,
)
```
Add a helper near the other module-level helpers:
```python
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```
Replace the `import_bars` command's signature options and request/metadata construction. New options block (make `--adjusted-dir` optional, add `--corp-actions`):
```python
    adjusted_dir: Path = typer.Option(
        None, "--adjusted-dir", help="firstrate: dir of adjusted per-symbol files (adj_close)"
    ),
    corp_actions: Path = typer.Option(
        None, "--corp-actions", help="databento: parquet of split/dividend events (computes adj_close)"
    ),
```
Replace the body that built `request = ImportRequest(...)` and the inline `source_metadata={...}` with the per-vendor arm (built before streaming):
```python
    importer = get_importer(vendor)
    sym_tuple = tuple(normalize_symbols(symbols.split(","))) if symbols else None
    if vendor == "firstrate":
        if adjusted_dir is None:
            raise ValueError("firstrate import requires --adjusted-dir")
        request: ImportRequest = FirstRateImportRequest(
            raw_dir=raw_dir, adjusted_dir=adjusted_dir, timeframe=timeframe, as_of=as_of,
            adjustment=adjustment, symbols=sym_tuple,
        )
        source_metadata = {
            "vendor": importer.vendor_label, "raw_dir": raw_dir.name, "adjusted_dir": adjusted_dir.name,
        }
    elif vendor == "databento":
        if corp_actions is None:
            raise ValueError("databento import requires --corp-actions")
        request = DatabentoImportRequest(
            raw_dir=raw_dir, corp_actions_path=corp_actions, timeframe=timeframe, as_of=as_of,
            adjustment=adjustment, symbols=sym_tuple,
        )
        source_metadata = {
            "vendor": importer.vendor_label, "raw_dir": raw_dir.name,
            "corp_actions_file": corp_actions.name, "corp_actions_sha256": _sha256(corp_actions),
            "ca_schema_version": "1",
        }
    else:
        raise ValueError(f"vendor {vendor!r} has no import-bars flag wiring")
```
Then pass `source_metadata=source_metadata` into the existing `store.ingest_bars_streamed(...)` call (replace the inline dict). The streaming/`_tracked()`/`emit` portion is otherwise unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_data.py -q`
Expected: PASS (databento + existing firstrate CLI tests).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/data_cmd.py tests/test_cli_data.py
git commit -m "feat(cli): databento import-bars arm (--corp-actions) + CA-file provenance (#150)"
```

---

### Task 6: Full-suite verification

- [ ] **Step 1: Run the whole gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green; FirstRate + ingest + conformance tests still pass (refactor regression), new databento tests pass.

- [ ] **Step 2: Confirm boundaries**

Run: `grep -nE "^(from|import) " algua/data/importers/databento.py`
Expected: imports only stdlib + numpy + pandas + `algua.data.contracts` / `algua.data.corpactions` / `algua.data.store` — no cross-layer import. `lint-imports` already enforces this.

- [ ] **Step 3:** If anything failed, fix and re-run; otherwise the slice is complete.

---

## Self-review notes (author)

- **Spec coverage:** canonical raw schema + tz/OHLCV policy (T2) ✓; CA schema + event_id dedup + row-level validation + aggregate-via-engine (T3) ✓; per-vendor request types + Protocol narrowing + FirstRate regression (T1) ✓; importer wiring + alignment check + registry + multi-symbol/one-chunk + same-date dividends (T4) ✓; CLI arm + `corp_actions_sha256`/`ca_schema_version` provenance + fail-closed flags (T5) ✓; gate + boundaries (T6) ✓. Deferred items (adj_factor persistence, native Databento, intraday) are intentionally not tasks.
- **Type consistency:** `parse_databento_raw` → `[ts, open, high, low, close, volume]`; `parse_databento_corp_actions` → `dict[str, list[CorporateAction]]`; `DatabentoImporter.import_bars(request: ImportRequest)` narrows to `DatabentoImportRequest`; request types/field names match the spec and Task 1.
- **No placeholders:** all code/test steps are concrete. The two implementer notes (CLI test conventions; rename-all-call-sites) point at exact files to match, not vague gaps.
