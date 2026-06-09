# Non-tabular Data Fundamentals Seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a point-in-time fundamentals data seam with two structurally-separated access modes — a strict as-of signal lane that feeds `compute_weights`, and a full-hindsight analysis lane for the agent — so hindsight can never leak into a backtest.

**Architecture:** A tidy/long bitemporal fundamentals record (`symbol, fiscal_period_end, metric, value, knowable_at, source`) reuses the existing snapshot/manifest/content-hash machinery. The engine owns decision `t` and masks `knowable_at ≤ t` per bar (mirroring `_members_as_of`). The hindsight accessor lives in `algua.data`, walled from the engine/strategy lanes by import-linter. Strategies opt in via `needs_fundamentals` and receive the masked frame as a 3rd `compute_weights` arg; the per-bar loop is forced (no vectorized fast path yet).

**Tech Stack:** Python 3.12, pandas, pyarrow/parquet, typer (CLI), pydantic (StrategyConfig), import-linter, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-non-tabular-data-fundamentals-seam-design.md`

**Quality gate (run after every task):**
```
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

---

## File structure

| File | Responsibility | New/Modify |
|---|---|---|
| `algua/contracts/types.py` | `FundamentalsProvider` protocol + fundamentals column-name constants (engine-importable) | Modify |
| `algua/data/models.py` | `Dataset.FUNDAMENTALS`, `Kind.FUNDAMENTALS` | Modify |
| `algua/data/fundamentals_schema.py` | validator, normalizer, empty frame, logical hash | New |
| `algua/data/store.py` | `ingest_fundamentals`, `read_fundamentals` | Modify |
| `algua/data/serve.py` | `StoreBackedFundamentalsProvider` (as-of lane) | Modify |
| `algua/data/hindsight.py` | `query_fundamentals` (hindsight lane) | New |
| `algua/strategies/base.py` | `needs_fundamentals`, `ComputeFundamentalsWeightsFn`, `fundamentals_fn`, dispatch, `config_hash` | Modify |
| `algua/strategies/loader.py` | signature validation, bind `fundamentals_fn` | Modify |
| `algua/strategies/examples/fundamentals_earnings_tilt.py` | example fundamentals strategy | New |
| `algua/backtest/engine.py` | `_fundamentals_as_of`, thread provider, mask per bar, force loop, fail-closed | Modify |
| `algua/backtest/result.py` | `BacktestResult.fundamentals_snapshot` | Modify |
| `algua/cli/backtest_cmd.py` | `--fundamentals-snapshot` wiring on `backtest run` | Modify |
| `algua/cli/data_cmd.py` | `ingest-fundamentals`, `query-fundamentals` | Modify |
| `algua/cli/paper_cmd.py`, `algua/cli/live_cmd.py` | fail-closed guard at the trading load points | Modify |
| `algua/registry/promotion.py` | research-promote guard | Modify |
| `pyproject.toml` | import-linter wall contracts | Modify |
| `docs/contracts/fundamentals-schema.md` | contract doc | New |
| `tests/...` | per-task tests | New |

---

## Task 1: Contracts — `FundamentalsProvider` protocol + column constants

**Files:**
- Modify: `algua/contracts/types.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_contracts.py`)

```python
def test_fundamentals_provider_protocol_and_constants():
    from algua.contracts.types import (
        FUNDAMENTALS_AS_OF_KEY,
        FUNDAMENTALS_COLUMNS,
        FUNDAMENTALS_KNOWABLE_AT,
        FundamentalsProvider,
    )

    assert FUNDAMENTALS_COLUMNS == (
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    )
    assert FUNDAMENTALS_AS_OF_KEY == ("symbol", "fiscal_period_end", "metric")
    assert FUNDAMENTALS_KNOWABLE_AT == "knowable_at"

    class _P:
        snapshot_id = "x"
        def get_fundamentals(self, symbols, end):
            return None

    assert isinstance(_P(), FundamentalsProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py::test_fundamentals_provider_protocol_and_constants -v`
Expected: FAIL with ImportError (names not defined).

- [ ] **Step 3: Implement** — add to `algua/contracts/types.py` (after the `DataProvider` protocol, before `Broker`):

```python
# --- Non-tabular: fundamentals seam (issue #132) -----------------------------------------------
# Canonical column names live HERE (the base layer the engine may import) so the backtest engine's
# as-of mask and the data-layer validator share one source of truth without the engine importing
# algua.data (which the import wall forbids). Pure strings — no pandas needed.
FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
)
# The identity of a fact across revisions: a restatement is a new row sharing this key with a later
# knowable_at. The as-of mask keeps, per key, the row with the greatest knowable_at <= t.
FUNDAMENTALS_AS_OF_KEY: tuple[str, ...] = ("symbol", "fiscal_period_end", "metric")
FUNDAMENTALS_KNOWABLE_AT = "knowable_at"


@runtime_checkable
class FundamentalsProvider(Protocol):
    """As-of consumption seam for point-in-time fundamentals (issue #132). Returns the FULL
    bitemporal history for `symbols` with knowable_at < end — no lower time bound, since the first
    decision bar needs the latest prior report. The engine owns decision `t` and masks
    knowable_at <= t per bar; the provider never sees `t`."""

    snapshot_id: str

    def get_fundamentals(self, symbols: list[str], end: datetime) -> pd.DataFrame: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py::test_fundamentals_provider_protocol_and_constants -v`
Expected: PASS.

- [ ] **Step 5: Run the gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/contracts/types.py tests/test_contracts.py
git commit -m "feat(contracts): FundamentalsProvider seam + fundamentals column constants (#132)"
```

---

## Task 2: Models — fundamentals dataset/kind enums

**Files:**
- Modify: `algua/data/models.py`
- Test: `tests/test_data_models.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
def test_fundamentals_enum_values():
    from algua.data.models import Dataset, Kind

    assert Dataset.FUNDAMENTALS.value == "fundamentals"
    assert Kind.FUNDAMENTALS.value == "fundamentals"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_data_models.py::test_fundamentals_enum_values -v`
Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement** — in `algua/data/models.py`:

```python
class Dataset(StrEnum):
    """Dataset routing key — the manifest `dataset` field and snapshot path component."""

    BARS = "bars"
    UNIVERSES = "universes"
    FUNDAMENTALS = "fundamentals"


class Kind(StrEnum):
    """Snapshot `kind` — the provenance of a snapshot's payload."""

    BARS = "bars"
    UNIVERSE = "universe"
    FILE = "file"
    FUNDAMENTALS = "fundamentals"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_data_models.py::test_fundamentals_enum_values -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/models.py tests/test_data_models.py
git commit -m "feat(data): fundamentals Dataset/Kind enum values (#132)"
```

---

## Task 3: Fundamentals schema — validator, normalizer, empty, logical hash

**Files:**
- Create: `algua/data/fundamentals_schema.py`
- Test: `tests/test_fundamentals_schema.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from algua.data.fundamentals_schema import (
    empty_fundamentals,
    logical_fundamentals_hash,
    to_fundamentals_schema,
    validate_fundamentals,
)


def _raw(rows):
    return pd.DataFrame(rows, columns=[
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ])


def _ok_rows():
    return _raw([
        ["aapl", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
        ["AAPL", "2025-03-31", "eps", float("nan"), "2025-05-01T13:00:00Z", "vendorX"],
    ])


def test_to_schema_normalizes_and_validates():
    out = to_fundamentals_schema(_ok_rows())
    assert list(out.columns) == [
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ]
    assert (out["symbol"] == "AAPL").all()  # upper-cased
    assert isinstance(out["fiscal_period_end"].iloc[0], date)
    assert str(out["knowable_at"].dt.tz) == "UTC"
    assert str(out["value"].dtype) == "float64"  # NaN preserved
    validate_fundamentals(out)


def test_knowable_at_must_be_ge_fiscal_period_end():
    bad = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-03-30T00:00:00Z", "v"]])
    with pytest.raises(ValueError, match="knowable_at"):
        to_fundamentals_schema(bad)


def test_same_day_filing_is_valid():
    ok = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-03-31T09:00:00Z", "v"]])
    out = to_fundamentals_schema(ok)  # must NOT raise
    assert len(out) == 1


def test_naive_knowable_at_rejected():
    bad = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-05-01T13:00:00", "v"]])
    with pytest.raises(ValueError, match="tz-aware"):
        to_fundamentals_schema(bad)


def test_bitemporal_key_uniqueness_enforced():
    dup = _raw([
        ["AAPL", "2025-03-31", "revenue", 1.0, "2025-05-01T13:00:00Z", "v"],
        ["AAPL", "2025-03-31", "revenue", 2.0, "2025-05-01T13:00:00Z", "v"],
    ])
    with pytest.raises(ValueError, match="unique"):
        to_fundamentals_schema(dup)


def test_empty_is_contract_shaped():
    e = empty_fundamentals()
    assert list(e.columns) == [
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ]
    assert len(e) == 0


def test_logical_hash_is_order_independent():
    a = to_fundamentals_schema(_ok_rows())
    shuffled = _ok_rows().iloc[::-1].reset_index(drop=True)
    b = to_fundamentals_schema(shuffled)
    assert logical_fundamentals_hash(a) == logical_fundamentals_hash(b)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_fundamentals_schema.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `algua/data/fundamentals_schema.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fundamentals_schema.py -v`
Expected: PASS (all).

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/fundamentals_schema.py tests/test_fundamentals_schema.py
git commit -m "feat(data): fundamentals schema — validator, normalizer, empty, logical hash (#132)"
```

---

## Task 4: Store — `ingest_fundamentals` + `read_fundamentals`

**Files:**
- Modify: `algua/data/store.py`
- Test: `tests/test_data_fundamentals_store.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from algua.data.fundamentals_schema import validate_fundamentals
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "vendorX"],  # restate
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def _store(tmp_path: Path) -> DataStore:
    return DataStore(tmp_path)


def test_ingest_then_read_roundtrips_and_validates(tmp_path):
    store = _store(tmp_path)
    rec = store.ingest_fundamentals(
        provider="vendorX", symbols=["AAPL"], as_of="2025-09-01T00:00:00Z",
        source="vendorX", frame=_raw(),
    )
    assert rec.dataset == "fundamentals"
    back = store.read_fundamentals(rec.snapshot_id)
    validate_fundamentals(back)
    assert len(back) == 2


def test_ingest_is_idempotent(tmp_path):
    store = _store(tmp_path)
    a = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-09-01T00:00:00Z", source="vendorX", frame=_raw())
    b = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-09-01T00:00:00Z", source="vendorX", frame=_raw())
    assert a.snapshot_id == b.snapshot_id


def test_ingest_rejects_knowable_after_as_of(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="as_of"):
        store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-06-01T00:00:00Z", source="vendorX", frame=_raw())


def test_read_filters_symbols(tmp_path):
    store = _store(tmp_path)
    two = pd.concat([_raw(), pd.DataFrame(
        [["MSFT", "2025-03-31", "revenue", 50.0, "2025-05-01T13:00:00Z", "vendorX"]],
        columns=_raw().columns)], ignore_index=True)
    rec = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL", "MSFT"],
                                    as_of="2025-09-01T00:00:00Z", source="vendorX", frame=two)
    only = store.read_fundamentals(rec.snapshot_id, symbols=["AAPL"])
    assert set(only["symbol"]) == {"AAPL"}
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_data_fundamentals_store.py -v`
Expected: FAIL (`ingest_fundamentals` missing).

- [ ] **Step 3: Implement** — in `algua/data/store.py`:

Add imports at the top of the file (extend the existing `algua.data.files` and `algua.data.schema` imports):

```python
from algua.data.files import (
    copy_snapshot,
    count_tabular_rows,
    frame_to_parquet_bytes,
    logical_bars_hash,
    read_partitioned_bars,
    sha256_bytes,
    sha256_file,
    write_bytes_snapshot,
    write_partitioned_bars,
)
from algua.data.fundamentals_schema import (
    empty_fundamentals,
    logical_fundamentals_hash,
    to_fundamentals_schema,
)
```

Add these two methods to the `DataStore` class (place after `read_bars`):

```python
    def ingest_fundamentals(
        self,
        *,
        provider: str,
        symbols: list[str],
        as_of: str,
        source: str,
        frame: pd.DataFrame,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Validate + normalize a tidy fundamentals frame and persist one immutable snapshot.
        `start`/`end` are DERIVED from the data (knowable_at range); every knowable_at must be
        <= `as_of` (you cannot have fetched a record that becomes knowable after you fetched it)."""
        canon = to_fundamentals_schema(frame)
        as_of_ts = pd.Timestamp(as_of)
        as_of_ts = (
            as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
        )
        if (canon["knowable_at"] > as_of_ts).any():
            raise ValueError(
                "fundamentals knowable_at must be <= as_of "
                "(cannot ingest a record knowable after the fetch time)"
            )
        start = canon["knowable_at"].min().date().isoformat()
        end = canon["knowable_at"].max().date().isoformat()
        metadata = _metadata(
            dataset=Dataset.FUNDAMENTALS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.FUNDAMENTALS.value,
            source_metadata=source_metadata,
        )
        content_hash = logical_fundamentals_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = (
            Path("snapshots") / metadata.dataset / snapshot_id / "fundamentals.parquet"
        )
        write_bytes_snapshot(frame_to_parquet_bytes(canon), self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        self.manifest.append(rec)
        return rec

    def read_fundamentals(
        self, snapshot_id: str, *, symbols: list[str] | None = None
    ) -> pd.DataFrame:
        """Read a fundamentals snapshot as a validated tidy frame. `symbols` filters in-memory
        (fundamentals are far smaller than bars; partitioned pushdown is deferred). Re-normalizes on
        read so parquet dtype drift cannot escape the schema. Empty result => empty_fundamentals()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.FUNDAMENTALS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, "
                f"not {Dataset.FUNDAMENTALS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_fundamentals()
        return to_fundamentals_schema(raw)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_data_fundamentals_store.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/store.py tests/test_data_fundamentals_store.py
git commit -m "feat(data): ingest_fundamentals + read_fundamentals snapshots (#132)"
```

---

## Task 5: Serving (`StoreBackedFundamentalsProvider`) + hindsight (`query_fundamentals`)

**Files:**
- Modify: `algua/data/serve.py`
- Create: `algua/data/hindsight.py`
- Test: `tests/test_fundamentals_serve_hindsight.py`

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd

from algua.contracts.types import FundamentalsProvider
from algua.data.hindsight import query_fundamentals
from algua.data.serve import StoreBackedFundamentalsProvider
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "v"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def _seed(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_fundamentals(provider="v", symbols=["AAPL"],
                                    as_of="2025-09-01T00:00:00Z", source="v", frame=_raw())
    return store, rec.snapshot_id


def test_as_of_provider_satisfies_protocol_and_returns_full_history(tmp_path):
    store, sid = _seed(tmp_path)
    prov = StoreBackedFundamentalsProvider(store, sid)
    assert isinstance(prov, FundamentalsProvider)
    assert prov.snapshot_id == sid
    out = prov.get_fundamentals(["AAPL"], pd.Timestamp("2025-12-31", tz="UTC"))
    assert len(out) == 2  # provider returns full bitemporal history; engine masks per t


def test_get_fundamentals_excludes_at_or_after_end(tmp_path):
    store, sid = _seed(tmp_path)
    prov = StoreBackedFundamentalsProvider(store, sid)
    out = prov.get_fundamentals(["AAPL"], pd.Timestamp("2025-06-01", tz="UTC"))
    assert len(out) == 1  # only the 2025-05-01 row is knowable before end


def test_hindsight_returns_everything(tmp_path):
    store, sid = _seed(tmp_path)
    out = query_fundamentals(store, sid, symbols=["AAPL"])
    assert len(out) == 2
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_fundamentals_serve_hindsight.py -v`
Expected: FAIL (imports missing).

- [ ] **Step 3a: Implement the as-of provider** — append to `algua/data/serve.py`:

```python
class StoreBackedFundamentalsProvider:
    """Serves one fundamentals snapshot through the as-of `FundamentalsProvider` seam. Returns the
    FULL bitemporal history with knowable_at < end (no lower bound — the first decision bar needs
    the latest prior report). The engine applies the per-bar knowable_at <= t mask; this provider
    never sees `t`."""

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_fundamentals(self, symbols: list[str], end: datetime) -> pd.DataFrame:
        frame = self.store.read_fundamentals(self.snapshot_id, symbols=symbols)
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        return frame[frame["knowable_at"] < end_ts].reset_index(drop=True)
```

- [ ] **Step 3b: Implement the hindsight accessor** — create `algua/data/hindsight.py`:

```python
from __future__ import annotations

import pandas as pd

from algua.data.store import DataStore

# The HINDSIGHT lane (issue #132). Returns FULL history regardless of any decision time — for agent
# post-mortems and idea sourcing ONLY. It is never wired into the backtest engine, and the import
# wall (pyproject.toml) forbids algua.backtest / algua.features / algua.contracts / algua.strategies
# from importing this module: hindsight must be structurally unable to reach compute_weights.


def query_fundamentals(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Full-hindsight fundamentals read (no as-of masking). Stable canonical row order for
    reproducible agent diffs (read_fundamentals already returns the canonical sort)."""
    return store.read_fundamentals(snapshot_id, symbols=symbols)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fundamentals_serve_hindsight.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/serve.py algua/data/hindsight.py tests/test_fundamentals_serve_hindsight.py
git commit -m "feat(data): as-of fundamentals provider + hindsight query lane (#132)"
```

---

## Task 6: The import wall — pyproject contracts + wall tests

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_fundamentals_wall.py`

- [ ] **Step 1: Write failing tests**

```python
import ast
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _contracts():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    return data["tool"]["importlinter"]["contracts"]


def test_strategies_barred_from_data_lane():
    cs = _contracts()
    assert any(
        c.get("source_modules") == ["algua.strategies"]
        and "algua.data" in c.get("forbidden_modules", [])
        for c in cs
    ), "missing: algua.strategies forbidden from algua.data"


def test_contracts_barred_from_data_lane():
    cs = _contracts()
    contracts_rule = next(c for c in cs if c.get("source_modules") == ["algua.contracts"])
    assert "algua.data" in contracts_rule["forbidden_modules"]


def test_hindsight_module_walled():
    cs = _contracts()
    rule = next(
        (c for c in cs if c.get("forbidden_modules") == ["algua.data.hindsight"]), None
    )
    assert rule is not None, "missing dedicated algua.data.hindsight forbidden contract"
    for src in ["algua.backtest", "algua.features", "algua.contracts",
                "algua.strategies", "algua.live", "algua.execution"]:
        assert src in rule["source_modules"]


def test_no_static_data_import_in_pure_layers():
    """Defense beyond config: assert no module under algua/strategies or algua/contracts imports
    algua.data (the actual property the wall protects)."""
    offenders = []
    for pkg in ["algua/strategies", "algua/contracts"]:
        for path in (REPO / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("algua.data"):
                    offenders.append(str(path))
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.startswith("algua.data"):
                            offenders.append(str(path))
    assert not offenders, f"pure layers import algua.data: {offenders}"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_fundamentals_wall.py -v`
Expected: FAIL (contracts not present yet).

- [ ] **Step 3: Implement** — in `pyproject.toml`:

(a) Add `"algua.data"` to the existing `contracts layer is pure` contract's `forbidden_modules`:

```toml
[[tool.importlinter.contracts]]
name = "contracts layer is pure (imports no other algua module)"
type = "forbidden"
source_modules = ["algua.contracts"]
forbidden_modules = [
    "algua.cli",
    "algua.registry",
    "algua.config",
    "algua.calendar",
    "algua.data",
]
```

(b) Append two new contracts (anywhere in the `[[tool.importlinter.contracts]]` list):

```toml
[[tool.importlinter.contracts]]
# Strategy modules are pure authored functions — they receive ALL data (bars, fundamentals) via the
# engine, never by importing the data layer. Without this, a strategy could import the hindsight
# accessor straight into compute_weights (issue #132 — the worst leak in the platform).
name = "strategies layer stays off the data lane"
type = "forbidden"
source_modules = ["algua.strategies"]
forbidden_modules = ["algua.data"]

[[tool.importlinter.contracts]]
# The hindsight (full-future) fundamentals accessor must be structurally unreachable from any lane
# that can influence a trading decision. Stated directly so it survives a future relocation.
name = "hindsight fundamentals accessor is unreachable from decision lanes"
type = "forbidden"
source_modules = [
    "algua.backtest", "algua.features", "algua.contracts",
    "algua.strategies", "algua.live", "algua.execution",
]
forbidden_modules = ["algua.data.hindsight"]
```

- [ ] **Step 4: Run to verify pass + the real linter**

Run: `uv run pytest tests/test_fundamentals_wall.py -v && uv run lint-imports`
Expected: PASS, and `lint-imports` reports all contracts kept.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add pyproject.toml tests/test_fundamentals_wall.py
git commit -m "feat(wall): import-linter contracts isolating the hindsight lane (#132)"
```

---

## Task 7: Strategy contract — `needs_fundamentals`, dispatch, config_hash

**Files:**
- Modify: `algua/strategies/base.py`
- Test: `tests/test_strategies_base_fundamentals.py`

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _cfg(needs):
    return StrategyConfig(
        name="x", universe=["AAPL"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, needs_fundamentals=needs,
    )


def test_config_defaults_no_fundamentals():
    assert _cfg(False).needs_fundamentals is False


def test_dispatch_passes_fundamentals_when_declared():
    seen = {}

    def fund_fn(view, params, fundamentals):
        seen["f"] = fundamentals
        return pd.Series(dtype="float64")

    ls = LoadedStrategy(config=_cfg(True), fundamentals_fn=fund_fn)
    frame = pd.DataFrame({"k": [1]})
    ls.target_weights(pd.DataFrame(), frame)
    assert seen["f"] is frame


def test_dispatch_plain_when_not_declared():
    def plain(view, params):
        return pd.Series([1.0], index=["AAPL"])

    ls = LoadedStrategy(config=_cfg(False), fn=plain)
    out = ls.target_weights(pd.DataFrame())
    assert out["AAPL"] == 1.0


def test_post_init_requires_matching_fn():
    with pytest.raises(ValueError, match="needs_fundamentals"):
        LoadedStrategy(config=_cfg(True), fn=lambda v, p: None)  # missing fundamentals_fn


def test_config_hash_includes_needs_fundamentals():
    a = LoadedStrategy(config=_cfg(False), fn=lambda v, p: None)
    b = LoadedStrategy(config=_cfg(True), fundamentals_fn=lambda v, p, f: None)
    assert config_hash(a) != config_hash(b)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_strategies_base_fundamentals.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — in `algua/strategies/base.py`:

Add the new authored-fn type after `ComputeWeightsPanelFn`:

```python
# OPT-IN fundamentals signal (issue #132): a strategy that declares `needs_fundamentals=True` in
# CONFIG authors `compute_weights(view, params, fundamentals)` — the 3rd arg is the PIT-correct tidy
# fundamentals frame the engine materialized for decision bar t (knowable_at <= t). Distinct type so
# the 2-arg and 3-arg forms never silently overload.
ComputeFundamentalsWeightsFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]
```

Add the field to `StrategyConfig`:

```python
class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}
    # Opt into the as-of fundamentals lane (issue #132). When True the loader binds the 3-arg
    # compute_weights as `fundamentals_fn` and the engine injects the PIT-correct frame per bar.
    needs_fundamentals: bool = False
```

Replace the `LoadedStrategy` dataclass with:

```python
@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + the authored signal function(s) into an object satisfying the
    Strategy protocol. Exactly one of (`fn`, `fundamentals_fn`) is active, selected by
    `config.needs_fundamentals`. The adapter is the ONLY place the protocol-level `target_weights`
    exists — it injects params (and, for the fundamentals lane, the masked frame)."""

    config: StrategyConfig
    fn: ComputeWeightsFn | None = None
    fundamentals_fn: ComputeFundamentalsWeightsFn | None = None
    panel_fn: ComputeWeightsPanelFn | None = None

    def __post_init__(self) -> None:
        if self.config.needs_fundamentals:
            if self.fundamentals_fn is None:
                raise ValueError(
                    "needs_fundamentals=True requires a 3-arg compute_weights (fundamentals_fn)"
                )
        elif self.fn is None:
            raise ValueError("needs_fundamentals=False requires a 2-arg compute_weights (fn)")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> list[str]:
        return self.config.universe

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution

    @property
    def params(self) -> dict[str, Any]:
        return self.config.params

    def target_weights(
        self, features: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.Series:
        if self.config.needs_fundamentals:
            assert self.fundamentals_fn is not None  # __post_init__ guarantees this
            return self.fundamentals_fn(features, self.config.params, fundamentals)
        assert self.fn is not None
        return self.fn(features, self.config.params)
```

Extend `config_hash`'s payload to include the flag:

```python
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": asdict(strategy.execution),
            "needs_fundamentals": strategy.config.needs_fundamentals,
        },
        sort_keys=True,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_strategies_base_fundamentals.py -v`
Expected: PASS.

- [ ] **Step 5: Update the Strategy protocol** — in `algua/contracts/types.py`, widen `target_weights` so the adapter matches the protocol:

```python
@runtime_checkable
class Strategy(Protocol):
    name: str
    execution: ExecutionContract

    def target_weights(
        self, features: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.Series: ...
```

- [ ] **Step 6: Gate + commit**

Note: `config_hash` now changes for ALL strategies (one-time, intentional — spec §5). Existing tests that pin a literal hash must be updated to the new value if any fail; re-run and adjust.

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/base.py algua/contracts/types.py tests/test_strategies_base_fundamentals.py
git commit -m "feat(strategies): needs_fundamentals opt-in + 3-arg compute_weights dispatch (#132)"
```

---

## Task 8: Loader signature validation + example fundamentals strategy

**Files:**
- Modify: `algua/strategies/loader.py`
- Create: `algua/strategies/examples/fundamentals_earnings_tilt.py`
- Test: `tests/test_loader_fundamentals.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from algua.strategies.loader import load_strategy


def test_loads_fundamentals_example():
    ls = load_strategy("fundamentals_earnings_tilt")
    assert ls.config.needs_fundamentals is True
    assert ls.fundamentals_fn is not None
    assert ls.fn is None


def test_loads_plain_example_unchanged():
    ls = load_strategy("cross_sectional_momentum")
    assert ls.config.needs_fundamentals is False
    assert ls.fn is not None
    assert ls.fundamentals_fn is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_loader_fundamentals.py -v`
Expected: FAIL (example missing / loader doesn't bind fundamentals_fn).

- [ ] **Step 3a: Create the example** — `algua/strategies/examples/fundamentals_earnings_tilt.py`:

```python
"""Earnings-yield tilt: among the universe, hold (equal-weight) the names whose latest KNOWN
diluted EPS is positive. A minimal demonstration of the as-of fundamentals lane (issue #132)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="fundamentals_earnings_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"metric": "eps_diluted"},
    needs_fundamentals=True,
)


def compute_weights(
    view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame
) -> pd.Series:
    metric = str(params["metric"])
    rows = fundamentals[fundamentals["metric"] == metric]
    if rows.empty:
        return pd.Series(dtype="float64")
    # latest known value per symbol (frame is already as-of-masked + canonically sorted by the
    # engine, so the last row per symbol is the most-recently-knowable)
    latest = rows.groupby("symbol")["value"].last()
    winners = latest[latest > 0.0].index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=list(winners))
```

- [ ] **Step 3b: Implement loader changes** — replace `load_strategy` in `algua/strategies/loader.py`:

```python
import importlib
import inspect
import pkgutil

from algua.strategies import examples
from algua.strategies.base import LoadedStrategy


class StrategyNotFound(LookupError):
    pass


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy module by name; it must expose CONFIG + compute_weights."""
    try:
        module = importlib.import_module(f"algua.strategies.examples.{name}")
    except ModuleNotFoundError as exc:
        raise StrategyNotFound(name) from exc
    if not hasattr(module, "CONFIG") or not hasattr(module, "compute_weights"):
        raise StrategyNotFound(f"{name} is missing CONFIG or compute_weights")

    panel_fn = getattr(module, "compute_weights_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.compute_weights_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(module.CONFIG, "needs_fundamentals", False))
    n_params = len(inspect.signature(module.compute_weights).parameters)
    if needs_fundamentals:
        # The fundamentals lane forces the per-bar loop (no vectorized fast path yet) and needs a
        # 3-arg signature. Reject the panel hook + a wrong arity, loudly, at load time.
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: compute_weights_panel is not supported with needs_fundamentals "
                f"(no vectorized fundamentals fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_fundamentals=True requires compute_weights(view, params, "
                f"fundamentals); got {n_params} params"
            )
        return LoadedStrategy(config=module.CONFIG, fundamentals_fn=module.compute_weights)

    if n_params != 2:
        raise StrategyNotFound(
            f"{name}: compute_weights must take (view, params); got {n_params} params"
        )
    return LoadedStrategy(config=module.CONFIG, fn=module.compute_weights, panel_fn=panel_fn)


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_loader_fundamentals.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/loader.py algua/strategies/examples/fundamentals_earnings_tilt.py tests/test_loader_fundamentals.py
git commit -m "feat(strategies): loader signature validation + example fundamentals strategy (#132)"
```

---

## Task 9: Engine — as-of mask, thread provider, force loop, fail-closed, result stamp

**Files:**
- Modify: `algua/backtest/engine.py`
- Modify: `algua/backtest/result.py`
- Test: `tests/test_engine_fundamentals.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import datetime

import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, _fundamentals_as_of, run
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy


def _funds():
    return pd.DataFrame(
        [
            # original report, then a restatement that flips the sign later
            ["AAPL", "2025-03-31", "eps_diluted", 1.0, "2025-05-01T13:00:00Z", "v"],
            ["AAPL", "2025-03-31", "eps_diluted", -1.0, "2025-08-01T13:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def test_as_of_mask_picks_latest_knowable_and_restatement_flips():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    before = _fundamentals_as_of(frame, pd.Timestamp("2025-06-01", tz="UTC"))
    assert before["value"].iloc[0] == 1.0  # original, restatement not yet knowable
    after = _fundamentals_as_of(frame, pd.Timestamp("2025-09-01", tz="UTC"))
    assert after["value"].iloc[0] == -1.0  # restated value now knowable


def test_as_of_mask_rejects_naive_t():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    with pytest.raises(BacktestError, match="tz-aware"):
        _fundamentals_as_of(frame, pd.Timestamp("2025-06-01"))


def test_as_of_mask_empty_before_any_knowable():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    out = _fundamentals_as_of(frame, pd.Timestamp("2025-01-01", tz="UTC"))
    assert len(out) == 0


def test_run_fails_closed_when_fundamentals_strategy_lacks_provider(tmp_path):
    # bars snapshot for the universe
    store = DataStore(tmp_path)
    bars = _toy_bars()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    strat = load_strategy("fundamentals_earnings_tilt")
    with pytest.raises(BacktestError, match="fundamentals"):
        run(strat, StoreBackedProvider(store, brec.snapshot_id),
            datetime(2025, 1, 1), datetime(2025, 1, 10))


def _toy_bars():
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = []
    for s in ["AAPL", "MSFT", "NVDA"]:
        for t in idx:
            rows.append([t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0])
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close",
                                     "adj_close", "volume"]).set_index("timestamp")
    return df


def test_run_with_fundamentals_stamps_snapshot(tmp_path):
    store = DataStore(tmp_path)
    bars = _toy_bars()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    funds = pd.DataFrame(
        [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "MSFT", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    strat = load_strategy("fundamentals_earnings_tilt")
    result = run(strat, StoreBackedProvider(store, brec.snapshot_id),
                 datetime(2025, 1, 1), datetime(2025, 1, 10),
                 fundamentals_provider=StoreBackedFundamentalsProvider(store, frec.snapshot_id))
    assert result.fundamentals_snapshot == frec.snapshot_id
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_engine_fundamentals.py -v`
Expected: FAIL.

- [ ] **Step 3a: Result field** — in `algua/backtest/result.py`, add to `BacktestResult` (after `universe_snapshots`):

```python
    # Fundamentals snapshot used by a needs_fundamentals strategy (issue #132); None otherwise.
    fundamentals_snapshot: str | None = None
```

- [ ] **Step 3b: Engine** — in `algua/backtest/engine.py`:

Extend imports:

```python
from algua.contracts.types import (
    FUNDAMENTALS_AS_OF_KEY,
    FUNDAMENTALS_COLUMNS,
    FUNDAMENTALS_KNOWABLE_AT,
    DataProvider,
    FundamentalsProvider,
)
```

Add the as-of mask + a light shape assertion (engine-importable; full validation lives at ingest +
in the provider — the engine cannot import `algua.data`):

```python
def _assert_fundamentals_shape(frame: pd.DataFrame) -> None:
    """Cheap structural defense at the engine seam (no algua.data import): the provider must hand
    back the contract columns with a tz-aware knowable_at. This is the no-branded-types insurance
    against an accidental wrong-frame swap (spec §2.1)."""
    missing = [c for c in FUNDAMENTALS_COLUMNS if c not in frame.columns]
    if missing:
        raise BacktestError(f"fundamentals frame missing columns {missing}")
    ka = frame[FUNDAMENTALS_KNOWABLE_AT]
    if not isinstance(ka.dtype, pd.DatetimeTZDtype):
        raise BacktestError("fundamentals 'knowable_at' must be tz-aware")


def _fundamentals_as_of(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """As-of-t fundamentals: of the rows with knowable_at <= t, keep for each
    (symbol, fiscal_period_end, metric) the row with the greatest knowable_at (latest revision
    knowable by t). knowable_at is unique per key within a snapshot, so the pick is deterministic.
    Uses only knowable_at <= t -> no look-ahead. Empty in/empty out (returns a 0-row slice, never a
    view into future rows)."""
    if t.tz is None:
        raise BacktestError("fundamentals as-of mask requires a tz-aware (UTC) timestamp t")
    visible = frame[frame[FUNDAMENTALS_KNOWABLE_AT] <= t]
    if visible.empty:
        return frame.iloc[0:0].copy()
    ordered = visible.sort_values(FUNDAMENTALS_KNOWABLE_AT, kind="stable")
    latest = ordered.drop_duplicates(subset=list(FUNDAMENTALS_AS_OF_KEY), keep="last")
    return latest.reset_index(drop=True)
```

Thread the provider through `_decision_weights`. Replace its signature + the per-bar call:

```python
def _decision_weights(
    strategy: LoadedStrategy,
    bars: pd.DataFrame,
    adj: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

Inside the loop, after the universe masking of `view` (just before `w = strategy.target_weights(view)`), replace the single call with:

```python
        if fundamentals is not None:
            f_asof = _fundamentals_as_of(fundamentals, t)
            if universe_by_date is not None:
                f_asof = f_asof[f_asof["symbol"].isin(members)]
            w = strategy.target_weights(view, f_asof)
        else:
            w = strategy.target_weights(view)
```

Thread `fundamentals` through the selector `_decision_weights_fast_or_loop` (force the loop when a
frame is present):

```python
def _decision_weights_fast_or_loop(
    strategy: LoadedStrategy,
    bars: pd.DataFrame,
    adj: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if strategy.panel_fn is None or universe_by_date is not None or fundamentals is not None:
        return _decision_weights(
            strategy, bars, adj, universe_by_date=universe_by_date, fundamentals=fundamentals
        )
    return _decision_weights_fast(strategy, bars, adj)
```

In `simulate`, add the param + materialize/validate/fail-closed, and pass it down. Update the
signature and the body after bars are fetched:

```python
def simulate(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    ...
    # (existing bars fetch + adj pivot stay unchanged)
    fundamentals: pd.DataFrame | None = None
    if strategy.config.needs_fundamentals:
        if fundamentals_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_fundamentals but no "
                f"fundamentals_provider was supplied (fail closed)"
            )
        fundamentals = fundamentals_provider.get_fundamentals(
            _fetch_symbols(strategy, universe_by_date), end
        )
        _assert_fundamentals_shape(fundamentals)

    weights = _decision_weights_fast_or_loop(
        strategy, bars, adj, universe_by_date=universe_by_date, fundamentals=fundamentals
    )
    ...  # (lag shift + vbt.Portfolio.from_orders unchanged)
```

In `run`, add the param, pass it to `simulate`, and stamp the snapshot id:

```python
def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
) -> BacktestResult:
    pf, weights_eff = simulate(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, fundamentals_provider=fundamentals_provider,
    )
    metrics = portfolio_metrics(pf, weights_eff)
    stamps = runtime_stamps()
    prov = provenance(provider, seed)
    return BacktestResult(
        strategy=strategy.name,
        metrics=metrics,
        config_hash=config_hash(strategy),
        timeframe="1d",
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        universe_name=universe_name,
        universe_snapshots=universe_snapshots,
        fundamentals_snapshot=getattr(fundamentals_provider, "snapshot_id", None),
        **prov,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_engine_fundamentals.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/backtest/engine.py algua/backtest/result.py tests/test_engine_fundamentals.py
git commit -m "feat(engine): as-of fundamentals masking + provider threading + result stamp (#132)"
```

---

## Task 10: Backtest CLI — `--fundamentals-snapshot`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_backtest_fundamentals.py`

- [ ] **Step 1: Write failing test**

```python
import json

from typer.testing import CliRunner

from algua.cli.app import app
from algua.config.settings import get_settings
from algua.data.store import DataStore
# reuse the toy-bars + funds helpers
import pandas as pd

runner = CliRunner()


def _seed(data_dir):
    store = DataStore(data_dir)
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "MSFT", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"]).set_index("timestamp")
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    funds = pd.DataFrame(
        [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "MSFT", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    return brec.snapshot_id, frec.snapshot_id


def test_backtest_run_with_fundamentals_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()  # if settings is cached; otherwise harmless
    bid, fid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "run", "fundamentals_earnings_tilt",
                              "--snapshot", bid, "--fundamentals-snapshot", fid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["fundamentals_snapshot"] == fid
```

(Adjust the settings/env fixture to match the repo's existing CLI-test convention in
`tests/test_cli_data.py` — use the same `data_dir` override mechanism those tests use.)

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_cli_backtest_fundamentals.py -v`
Expected: FAIL (`--fundamentals-snapshot` unknown).

- [ ] **Step 3: Implement** — in `algua/cli/backtest_cmd.py`:

Add the import:

```python
from algua.data.serve import StoreBackedFundamentalsProvider
from algua.data.store import DataStore
```

Add the option + provider construction in `run` (the `backtest run` command):

```python
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
```

```python
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot
        else None
    )
    result = run_backtest(
        strategy, provider, start_dt, end_dt,
        universe_by_date=universe_by_date,
        universe_name=universe, universe_snapshots=universe_prov,
        fundamentals_provider=fundamentals_provider,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_backtest_fundamentals.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/backtest_cmd.py tests/test_cli_backtest_fundamentals.py
git commit -m "feat(cli): backtest run --fundamentals-snapshot wiring (#132)"
```

---

## Task 11: Data CLI — `ingest-fundamentals` + `query-fundamentals`

**Files:**
- Modify: `algua/cli/data_cmd.py`
- Test: `tests/test_cli_data_fundamentals.py`

- [ ] **Step 1: Write failing test**

```python
import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.app import app

runner = CliRunner()


def test_ingest_then_query_fundamentals_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))  # match tests/test_cli_data.py convention
    src = tmp_path / "funds.csv"
    pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "vendorX"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    ).to_csv(src, index=False)

    ing = runner.invoke(app, ["data", "ingest-fundamentals", "--from-file", str(src),
                              "--provider", "vendorX", "--symbols", "AAPL",
                              "--as-of", "2025-09-01T00:00:00Z", "--source", "vendorX"])
    assert ing.exit_code == 0, ing.output
    sid = json.loads(ing.output)["snapshot"]["snapshot_id"]

    q = runner.invoke(app, ["data", "query-fundamentals", "--snapshot-id", sid, "--symbols", "AAPL"])
    assert q.exit_code == 0, q.output
    rows = json.loads(q.output)
    assert len(rows) == 2  # full hindsight
    assert rows[0]["symbol"] == "AAPL"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_cli_data_fundamentals.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — in `algua/cli/data_cmd.py`:

Add imports:

```python
import pandas as pd

from algua.data.hindsight import query_fundamentals
```

Add two commands (after `ingest_universe`):

```python
@data_app.command("ingest-fundamentals")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_fundamentals(
    provider: str = typer.Option(..., "--provider"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option(..., "--source", help="source/provenance label"),
    from_file: Path = FROM_FILE_OPTION,
) -> None:
    """Ingest a local tidy fundamentals file (CSV/parquet) as one validated snapshot."""
    path = from_file.expanduser()
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    rec = _store().ingest_fundamentals(
        provider=provider,
        symbols=normalize_symbols(symbols.split(",")),
        as_of=as_of,
        source=source,
        frame=raw,
    )
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("query-fundamentals")
@json_errors(ValueError, LookupError, FileNotFoundError)
def query_fundamentals_cmd(
    snapshot_id: str = typer.Option(..., "--snapshot-id"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """HINDSIGHT fundamentals read (full history) — the agent's post-mortem/analysis surface."""
    syms = normalize_symbols(symbols.split(",")) if symbols else None
    frame = query_fundamentals(_store(), snapshot_id, symbols=syms)
    records = [
        {
            "symbol": row.symbol,
            "fiscal_period_end": row.fiscal_period_end.isoformat(),
            "metric": row.metric,
            "value": None if pd.isna(row.value) else float(row.value),
            "knowable_at": row.knowable_at.isoformat(),
            "source": row.source,
        }
        for row in frame.itertuples(index=False)
    ]
    emit(records)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_data_fundamentals.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/data_cmd.py tests/test_cli_data_fundamentals.py
git commit -m "feat(cli): data ingest-fundamentals + query-fundamentals (hindsight) (#132)"
```

---

## Task 12: Fail-closed guards (paper/live run entry + research promote)

**Files:**
- Modify: `algua/strategies/base.py` (shared guard helper)
- Modify: `algua/cli/paper_cmd.py`, `algua/cli/live_cmd.py`
- Modify: `algua/registry/promotion.py`
- Test: `tests/test_fundamentals_guards.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

from algua.strategies.base import assert_tradable_without_fundamentals
from algua.strategies.loader import load_strategy


def test_helper_blocks_fundamentals_strategy():
    strat = load_strategy("fundamentals_earnings_tilt")
    with pytest.raises(ValueError, match="fundamentals"):
        assert_tradable_without_fundamentals(strat)


def test_helper_allows_plain_strategy():
    assert_tradable_without_fundamentals(load_strategy("cross_sectional_momentum"))  # no raise


def test_promotion_preflight_blocks_fundamentals(tmp_path):
    # build a registry with a fundamentals strategy at stage backtested, then preflight must refuse
    from algua.contracts.lifecycle import Actor
    from algua.registry.promotion import promotion_preflight
    from algua.registry.store import SqliteStrategyRepository
    import sqlite3

    conn = sqlite3.connect(":memory:")
    repo = SqliteStrategyRepository(conn)  # adjust to the repo's actual constructor/migration call
    repo.add("fundamentals_earnings_tilt")
    # advance to backtested via the repo's transition API (match existing test helpers)
    from algua.contracts.lifecycle import Stage
    from algua.registry.transitions import transition_strategy
    transition_strategy(repo, "fundamentals_earnings_tilt", Stage.BACKTESTED, Actor.AGENT, "seed")
    with pytest.raises(ValueError, match="fundamentals"):
        promotion_preflight(repo, "fundamentals_earnings_tilt", actor=Actor.AGENT,
                            declared_combos=None, allow_holdout_reuse=False, allow_non_pit=False)
```

(Match `SqliteStrategyRepository` construction / migration to the pattern in the existing registry
tests, e.g. `tests/test_registry_*`.)

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_fundamentals_guards.py -v`
Expected: FAIL.

- [ ] **Step 3a: Shared helper** — add to `algua/strategies/base.py`:

```python
def assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_fundamentals strategy must NOT run paper/live yet — the as-of
    fundamentals lane is wired only into the backtest engine (issue #132). Called at every trading
    load point so no actor (agent promote OR human raw transition) can run it blind."""
    if strategy.config.needs_fundamentals:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_fundamentals; paper/live fundamentals "
            f"wiring is not built yet (#132 follow-up) — refusing to trade it blind"
        )
```

- [ ] **Step 3b: Paper guard** — in `algua/cli/paper_cmd.py`, inside `_load_gated_strategy`, after
`strategy = load_strategy(name)` (line ~65):

```python
    from algua.strategies.base import assert_tradable_without_fundamentals
    assert_tradable_without_fundamentals(strategy)
```

- [ ] **Step 3c: Live guard** — in `algua/cli/live_cmd.py`, inside `_run_strategy_tick`, after
`strategy = load_strategy(name)` (line ~108):

```python
    from algua.strategies.base import assert_tradable_without_fundamentals
    assert_tradable_without_fundamentals(strategy)
```

- [ ] **Step 3d: Promote guard** — in `algua/registry/promotion.py`, inside `promotion_preflight`,
right after the stage check (`if rec.stage is not Stage.BACKTESTED: ...`), add:

```python
    # Fundamentals strategies cannot be promoted past backtested until the paper/live fundamentals
    # lane exists (#132): block the agent's only path to shortlisted early, with a clear message.
    from algua.strategies.loader import load_strategy

    if load_strategy(name).config.needs_fundamentals:
        raise ValueError(
            f"strategy {name!r} declares needs_fundamentals; it cannot be promoted past "
            f"backtested until the paper/live fundamentals lane is built (#132)"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fundamentals_guards.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/base.py algua/cli/paper_cmd.py algua/cli/live_cmd.py algua/registry/promotion.py tests/test_fundamentals_guards.py
git commit -m "feat(guards): fail-closed block of fundamentals strategies in paper/live/promote (#132)"
```

---

## Task 13: Contract doc + PIT-universe leak test + final sweep

**Files:**
- Create: `docs/contracts/fundamentals-schema.md`
- Test: `tests/test_engine_fundamentals_pit_universe.py`

- [ ] **Step 1: PIT-universe × fundamentals leak test** (the GATE-1 CRITICAL — a future
constituent's known fundamentals must not influence a current member's weight):

```python
from datetime import datetime

import pandas as pd

from algua.backtest.engine import run
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy


def test_future_member_fundamentals_do_not_leak(tmp_path):
    """NVDA joins the universe only AFTER the decision window; even though its EPS is positive and
    knowable, the as-of-member mask must keep it out of the weights while it is not a member."""
    store = DataStore(tmp_path)
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"]).set_index("timestamp")
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    # universe: only AAPL is a member during the window; NVDA becomes a member far later.
    store.ingest_universe(universe="u", symbols=["AAPL"], effective_date="2024-12-01",
                          as_of="2025-01-01T00:00:00Z", source="t")
    funds = pd.DataFrame(
        [
            ["AAPL", "2024-12-31", "eps_diluted", 1.0, "2024-12-31T00:00:00Z", "v"],
            ["NVDA", "2024-12-31", "eps_diluted", 1.0, "2024-12-31T00:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    from algua.cli._common import resolve_universe_inputs  # reuse the wiring helper
    ubd, _ = resolve_universe_inputs("u", datetime(2025, 1, 1), datetime(2025, 1, 10))
    result = run(load_strategy("fundamentals_earnings_tilt"),
                 StoreBackedProvider(store, brec.snapshot_id),
                 datetime(2025, 1, 1), datetime(2025, 1, 10),
                 universe_by_date=ubd, universe_name="u",
                 fundamentals_provider=StoreBackedFundamentalsProvider(store, frec.snapshot_id))
    # the run completes without a non-member-weight BacktestError => NVDA never got weight
    assert result.strategy == "fundamentals_earnings_tilt"
```

- [ ] **Step 2: Run to verify it passes** (this validates the Task-9 universe masking).

Run: `uv run pytest tests/test_engine_fundamentals_pit_universe.py -v`
Expected: PASS. If it FAILS with a non-member-weight `BacktestError`, the fundamentals universe
masking in `_decision_weights` (Task 9 Step 3b) is wrong — fix there.

- [ ] **Step 3: Write the contract doc** — create `docs/contracts/fundamentals-schema.md`:

```markdown
# Fundamentals schema (the non-tabular PIT seam)

Tidy/long, bitemporal. One row = one metric value for one issuer/period, stamped with when it
became knowable.

| column | type | meaning |
|---|---|---|
| `symbol` | str (upper-cased, non-null) | issuer ticker |
| `fiscal_period_end` | date (non-null) | the period the figure describes |
| `metric` | str (non-null) | metric name, e.g. `revenue`, `eps_diluted` |
| `value` | float64 (NaN allowed = reported-but-unavailable) | the figure |
| `knowable_at` | tz-aware UTC datetime (non-null) | report availability = filing + lag; the PIT key |
| `source` | str (non-null) | provenance label |

## Point-in-time rule
A record is visible at decision `t` iff `knowable_at <= t`. The backtest engine owns `t` (the bar
timestamp) and masks per bar — the strategy never chooses `t`. Because daily bars are midnight-UTC,
an intraday filing on day D is first visible at the decision for D+1 (conservative; never leaks).

## Restatements
A restatement is a NEW row sharing `(symbol, fiscal_period_end, metric)` with a later `knowable_at`.
The as-of mask keeps, per that key, the row with the greatest `knowable_at <= t` — originally-reported
before the restatement is knowable, restated after.

## Validation floor
`knowable_at >= fiscal_period_end` (UTC midnight) — a sanity floor, not a precise availability model.
Bitemporal key `(symbol, fiscal_period_end, metric, knowable_at)` is unique within a snapshot.

## Two access modes
- **As-of (signal):** `FundamentalsProvider.get_fundamentals` → engine mask → `compute_weights`.
- **Hindsight (analysis):** `algua data query-fundamentals` (full history) — agent post-mortems only,
  structurally walled from the engine (import-linter).
```

- [ ] **Step 4: Full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add docs/contracts/fundamentals-schema.md tests/test_engine_fundamentals_pit_universe.py
git commit -m "docs+test: fundamentals contract doc + PIT-universe leak test (#132)"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** typed contract+validator (T3), consumption seam (T1/T5/T9), PIT enforcement +
  engine-owned t (T9), restatement (T9 test), PIT-universe interaction (T13), sanitization/dedup
  (T3/T4), storage substrate reuse (T4), dual access modes + wall (T5/T6), strategy seam (T7/T8),
  promotion/paper-live guards (T12), CLI both lanes (T10/T11), config_hash + result stamp (T7/T9),
  contract doc (T13). All spec sections map to a task.
- **Validation-location clarification (vs spec §2.1):** the engine cannot import `algua.data`, so
  full `validate_fundamentals` runs at ingest (T4) and inside the provider's read (T5); the engine
  does a light column/tz assertion `_assert_fundamentals_shape` (T9). Same intent, respects the wall.
- **Wall-test clarification (vs spec §2.3):** implemented as config-presence + an AST "no algua.data
  import in pure layers" test (T6) rather than a gate-breaking must-fail fixture.
- **Type consistency:** `FUNDAMENTALS_COLUMNS/_AS_OF_KEY/_KNOWABLE_AT` (T1) used identically in
  schema (T3) and engine (T9); `ComputeFundamentalsWeightsFn`/`fundamentals_fn` (T7) bound by loader
  (T8) and dispatched by adapter (T7); `get_fundamentals(symbols, end)` signature identical across
  protocol (T1), provider (T5), engine call (T9).
- **Note for executor:** CLI tests must use the repo's existing `data_dir`/settings override
  convention (see `tests/test_cli_data.py`); the `ALGUA_DATA_DIR` env shown is a placeholder to
  align with that pattern.
```
