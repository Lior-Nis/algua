# Scale the Bars Read Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve bars snapshots from a hive-partitioned (by `symbol`) parquet dataset with pyarrow predicate pushdown, so a narrow query no longer loads the whole snapshot into RAM.

**Architecture:** A bars snapshot becomes a directory `snapshots/bars/<id>/symbol=AAPL/part-0.parquet`. Reproducible identity comes from a *logical* content hash over the canonical rows (not physical file bytes). `read_bars` pushes `symbols` + half-open `[start, end)` filters down to a `pyarrow.dataset`, scanning only matching fragments. `StoreBackedProvider.get_bars` delegates to it. As-of reads and incremental composition are out of scope.

**Tech Stack:** Python 3.12, pandas, pyarrow (`pyarrow.dataset`), pytest. Driven through `uv run`.

**Spec:** `docs/superpowers/specs/2026-06-08-scale-bars-read-path-design.md`

**Gate (run after each task's commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File structure

- `algua/data/files.py` — **Modify.** Add the partitioned-bars I/O + logical hash primitives: `logical_bars_hash`, `write_partitioned_bars`, `read_partitioned_bars`, plus the shared `BARS_FILE_SCHEMA` / hive partitioning + `_ts_scalar`. Keeps all pyarrow-dataset mechanics in one place; `store.py` stays orchestration.
- `algua/data/schema.py` — **Modify.** Add `empty_bars()` — the contract's empty-but-typed frame.
- `algua/data/store.py` — **Modify.** `ingest_bars` writes the partitioned layout + logical hash; `read_bars` gains `symbols/start/end` pushdown filters, a legacy-layout guard, and the empty-frame path. Remove the now-unused `_normalize_bar_frame`.
- `algua/data/serve.py` — **Modify.** `StoreBackedProvider.get_bars` delegates filtering to `read_bars`.
- `tests/test_data_files_partitioned.py` — **Create.** Unit tests for the new `files.py` primitives.
- `tests/test_data_schema.py` — **Modify.** Add an `empty_bars` test.
- `tests/test_data_store.py` — **Modify.** Update the two single-file-layout assertions; replace the physical-bytes content-hash test with a logical-hash test.
- `tests/test_data_read_bars.py` — **Modify.** Add pushdown / legacy-guard / empty-result tests.
- `tests/test_data_serve.py` — **Modify.** Add a pushdown-delegation + boundary test (existing tests should still pass unchanged).
- `tests/test_cli_data.py` — **Modify.** Update the `storage_format == "parquet"` assertion to `"parquet_dataset"`.

---

### Task 1: `logical_bars_hash` — version-independent content identity

**Files:**
- Modify: `algua/data/files.py`
- Test: `tests/test_data_files_partitioned.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_files_partitioned.py
import hashlib

import pandas as pd

from algua.data.files import logical_bars_hash


def _canon(rows):
    # rows: list of (ts_iso, symbol, o, h, l, c, adj, vol)
    df = pd.DataFrame(
        rows, columns=["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_logical_hash_is_order_invariant_and_deterministic():
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 2.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 3.0),
    ]
    h1 = logical_bars_hash(_canon(rows))
    h2 = logical_bars_hash(_canon(list(reversed(rows))))  # shuffled input
    assert h1 == h2  # same logical rows => same hash, regardless of row order
    assert len(h1) == len(hashlib.sha256().hexdigest())


def test_logical_hash_changes_on_value_change():
    base = _canon([("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0)])
    changed = _canon([("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 2.0)])
    assert logical_bars_hash(base) != logical_bars_hash(changed)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_data_files_partitioned.py -q`
Expected: FAIL — `ImportError: cannot import name 'logical_bars_hash'`.

- [ ] **Step 3: Implement `logical_bars_hash`**

Add to `algua/data/files.py`. Add `import struct` to the existing imports and import the float column list from schema:

```python
import struct

from algua.data.schema import BARS_FILE_HASH_COLUMNS
```

> Note: `BARS_FILE_HASH_COLUMNS` is added in Step 3b below (it is just `FLOAT_COLUMNS`, re-exported under a name `files.py` can depend on without a cycle — `schema.py` imports only pandas).

```python
def logical_bars_hash(canon: pd.DataFrame) -> str:
    """Content hash over the *logical* bar rows — independent of physical parquet layout/version.

    `canon` carries a tz-aware UTC `ts` column, a `symbol` column, and the six float columns. Rows
    are sorted by (ts, symbol); each column is serialized as fixed-width little-endian bytes (ts as
    int64 nanoseconds-since-epoch UTC, floats as IEEE-754 float64) with a NUL-joined symbol blob.
    Identical logical bars => identical digest regardless of write threading, file splitting, or
    pyarrow version. This is the snapshot identity for the partitioned bars layout (issue #130,
    GATE-1 HIGH #1/#2), replacing the single-file physical-bytes hash.
    """
    ordered = canon.sort_values(["ts", "symbol"], kind="stable")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(ordered)))
    ts_utc = ordered["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    ts_ns = ts_utc.to_numpy(dtype="datetime64[ns]").view("int64").astype("<i8")
    digest.update(ts_ns.tobytes())
    digest.update("\x00".join(ordered["symbol"].astype(str)).encode("utf-8"))
    digest.update(b"\x00")
    for col in BARS_FILE_HASH_COLUMNS:
        digest.update(ordered[col].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()
```

- [ ] **Step 3b: Re-export the hash column list from `schema.py`**

In `algua/data/schema.py`, immediately after `FLOAT_COLUMNS` is defined, add:

```python
# Re-exported so algua.data.files can serialize the numeric bar columns for the logical content
# hash without importing store (schema imports only pandas, so this stays cycle-free).
BARS_FILE_HASH_COLUMNS = FLOAT_COLUMNS
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_data_files_partitioned.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/data/files.py algua/data/schema.py tests/test_data_files_partitioned.py
git commit -m "feat(data): logical content hash for partitioned bars identity (#130)"
```

---

### Task 2: `write_partitioned_bars` + `read_partitioned_bars` round-trip & pushdown

**Files:**
- Modify: `algua/data/files.py`
- Test: `tests/test_data_files_partitioned.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_files_partitioned.py`:

```python
from datetime import datetime, UTC

from algua.data.files import read_partitioned_bars, write_partitioned_bars


def _canon_sorted(rows):
    return _canon(rows).sort_values(["symbol", "ts"])


def test_write_then_read_full_round_trips(tmp_path):
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 2.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 3.0),
    ]
    dest = tmp_path / "snap"
    file_count = write_partitioned_bars(_canon_sorted(rows), dest)
    assert file_count == 2  # one file per symbol (AAA, BBB)
    assert (dest / "symbol=AAA").is_dir() and (dest / "symbol=BBB").is_dir()

    out = read_partitioned_bars(dest)
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert len(out) == 3
    assert list(out.columns) == ["ts", "symbol", "open", "high", "low", "close",
                                 "adj_close", "volume"]
    assert all(isinstance(s, str) for s in out["symbol"])  # not categorical/dict


def test_read_pushes_down_symbol_and_half_open_window(tmp_path):
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 2.0),
        ("2024-07-03T00:00:00+00:00", "AAA", 12.0, 12.0, 12.0, 12.0, 12.0, 3.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 4.0),
    ]
    dest = tmp_path / "snap"
    write_partitioned_bars(_canon_sorted(rows), dest)

    out = read_partitioned_bars(
        dest, symbols=["AAA"],
        start=datetime(2024, 7, 1, tzinfo=UTC), end=datetime(2024, 7, 3, tzinfo=UTC),
    )
    assert set(out["symbol"]) == {"AAA"}                 # symbol pruning
    assert out["ts"].min() == pd.Timestamp("2024-07-01", tz="UTC")   # start inclusive
    assert out["ts"].max() == pd.Timestamp("2024-07-02", tz="UTC")   # end exclusive (07-03 dropped)


def test_read_empty_window_returns_empty_frame(tmp_path):
    rows = [("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0)]
    dest = tmp_path / "snap"
    write_partitioned_bars(_canon_sorted(rows), dest)
    out = read_partitioned_bars(
        dest, symbols=["AAA"],
        start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert out.empty
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_data_files_partitioned.py -q`
Expected: FAIL — `ImportError: cannot import name 'write_partitioned_bars'`.

- [ ] **Step 3: Implement the writer/reader**

Add to `algua/data/files.py`. Extend the pyarrow imports at the top:

```python
import functools

import pyarrow.dataset as pads
```

(`import pyarrow as pa` and `import pyarrow.parquet as pq` already exist.)

Then add the schema, partitioning, and functions:

```python
BARS_FILE_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adj_close", pa.float64()),
        ("volume", pa.float64()),
    ]
)
_BARS_PARTITIONING = pads.partitioning(pa.schema([("symbol", pa.string())]), flavor="hive")


def write_partitioned_bars(canon: pd.DataFrame, dest_dir: Path) -> int:
    """Write `canon` (a tz-aware-UTC `ts` column + `symbol` + OHLCV, pre-sorted by symbol then ts)
    as a hive-partitioned-by-symbol parquet dataset under `dest_dir`. Returns the parquet file
    count. The snapshot identity is the caller's `logical_bars_hash`, NOT these bytes, so write
    threading / file splitting are free to vary (issue #130)."""
    table = pa.Table.from_pandas(
        canon[["ts", "symbol", *BARS_FILE_HASH_COLUMNS]],
        schema=BARS_FILE_SCHEMA,
        preserve_index=False,
    )
    pads.write_dataset(
        table,
        dest_dir,
        format="parquet",
        partitioning=_BARS_PARTITIONING,
        basename_template="part-{i}.parquet",
    )
    return sum(1 for _ in dest_dir.rglob("*.parquet"))


def read_partitioned_bars(
    dest_dir: Path,
    *,
    symbols: list[str] | None = None,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    """Read a hive-partitioned bars dataset with predicate pushdown. `symbols` prunes partitions at
    the directory level (no other symbol's footer is read); `start`/`end` push a half-open
    `[start, end)` filter on `ts` down to the scanner. Any of the three may be None (unbounded).
    Returns a raw frame (`ts` column + `symbol` + OHLCV); the caller reshapes to bar-schema. Only
    matching fragments are scanned (issue #130)."""
    dataset = pads.dataset(dest_dir, format="parquet", partitioning=_BARS_PARTITIONING)
    conds = []
    if symbols is not None:
        conds.append(pads.field("symbol").isin(list(symbols)))
    if start is not None:
        conds.append(pads.field("ts") >= _ts_scalar(start))
    if end is not None:
        conds.append(pads.field("ts") < _ts_scalar(end))
    filt = functools.reduce(lambda a, b: a & b, conds) if conds else None
    table = dataset.to_table(columns=["ts", "symbol", *BARS_FILE_HASH_COLUMNS], filter=filt)
    return table.to_pandas()


def _ts_scalar(value: object) -> pa.Scalar:
    """Build a `timestamp[ns, tz=UTC]` pyarrow scalar from a datetime/Timestamp, normalizing naive
    inputs to UTC. Constructing the literal in the column's exact type avoids tz/precision-mismatch
    boundary bugs in the pushed-down filter (GATE-1 MEDIUM #4)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return pa.scalar(ts.value, type=pa.timestamp("ns", tz="UTC"))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_data_files_partitioned.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/data/files.py tests/test_data_files_partitioned.py
git commit -m "feat(data): partitioned bars writer + predicate-pushdown reader (#130)"
```

---

### Task 3: `empty_bars()` — the contract's empty-but-typed frame

**Files:**
- Modify: `algua/data/schema.py`
- Test: `tests/test_data_schema.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_schema.py`:

```python
from algua.data.schema import BAR_COLUMNS, empty_bars, validate_bars


def test_empty_bars_is_contract_shaped():
    out = empty_bars()
    assert out.empty
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.name == "timestamp"
    assert str(out.index.tz) == "UTC"
    validate_bars(out)  # must satisfy the frozen schema unconditionally
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_data_schema.py::test_empty_bars_is_contract_shaped -q`
Expected: FAIL — `ImportError: cannot import name 'empty_bars'`.

- [ ] **Step 3: Implement `empty_bars`**

Add to `algua/data/schema.py` (after `validate_bars`):

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_data_schema.py::test_empty_bars_is_contract_shaped -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/schema.py tests/test_data_schema.py
git commit -m "feat(data): empty_bars() contract-shaped empty frame (#130)"
```

---

### Task 4: `ingest_bars` writes partitioned; `read_bars` pushes down + guards legacy

**Files:**
- Modify: `algua/data/store.py`
- Test: `tests/test_data_store.py`, `tests/test_data_read_bars.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_data_store.py`, **update** the two single-file assertions and **replace** the physical-bytes hash test.

Change `test_ingest_bars_writes_parquet_snapshot_with_provenance`:
- replace `assert rec.storage_format == "parquet"` with `assert rec.storage_format == "parquet_dataset"`
- the `saved = pd.read_parquet(tmp_path / "data" / rec.data_path)` line still works (`pd.read_parquet` reads the partitioned directory); keep `assert list(saved["close"]) == [100.0, 101.0]`.

Replace `test_content_hash_is_canonical_parquet_bytes` entirely with:

```python
def test_content_hash_is_logical_and_layout_independent(tmp_path):
    # #130: identity is a logical hash over canonical rows, so the same logical bars in a different
    # input row order dedup to the same snapshot (independent of physical file layout/order).
    store = DataStore(tmp_path / "data")
    rows = {
        "ts": ["2026-01-03T00:00:00+00:00", "2026-01-02T00:00:00+00:00"],
        "symbol": ["AAPL", "AAPL"],
        "open": [100.0, 99.0], "high": [102.0, 101.0], "low": [99.0, 98.0],
        "close": [101.0, 100.0], "adj_close": [100.5, 99.5], "volume": [1100.0, 1000.0],
    }
    kwargs = dict(
        provider="fixture", symbols=["AAPL"], start="2026-01-02", end="2026-01-03",
        as_of="2026-01-04T00:00:00+00:00", source="fixture",
    )
    first = store.ingest_bars(frame=pd.DataFrame(rows), **kwargs)
    shuffled = {k: list(reversed(v)) for k, v in rows.items()}
    second = store.ingest_bars(frame=pd.DataFrame(shuffled), **kwargs)
    assert second.snapshot_id == first.snapshot_id
    assert len(store.list_snapshots()) == 1
```

In `tests/test_data_read_bars.py`, append pushdown + guard + empty tests:

```python
from datetime import UTC, datetime


def test_read_bars_pushes_down_symbol_and_window(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)  # AAA/BBB on 2024-07-01 and 2024-07-02
    out = store.read_bars(
        rec.snapshot_id, symbols=["AAA"],
        start=datetime(2024, 7, 1, tzinfo=UTC), end=datetime(2024, 7, 2, tzinfo=UTC),
    )
    validate_bars(out)
    assert set(out["symbol"]) == {"AAA"}
    assert out.index.max() == pd.Timestamp("2024-07-01", tz="UTC")  # 07-02 == end, excluded


def test_read_bars_empty_window_returns_typed_empty(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    out = store.read_bars(
        rec.snapshot_id, symbols=["AAA"],
        start=datetime(2030, 1, 1, tzinfo=UTC), end=datetime(2030, 1, 2, tzinfo=UTC),
    )
    validate_bars(out)
    assert out.empty
    assert list(out.columns) == BAR_COLUMNS


def test_read_bars_rejects_legacy_single_file_snapshot(tmp_path):
    from algua.data.manifest import SnapshotManifest
    from algua.data.models import SnapshotMetadata, SnapshotRecord
    from pathlib import Path

    store = DataStore(tmp_path)
    rec = _ingest(store)
    # Forge a manifest record claiming the old single-file layout for a bars dataset.
    legacy = SnapshotRecord(
        snapshot_id="legacyid00000000",
        metadata=SnapshotMetadata(
            dataset="bars", provider="p", symbols=("AAA",), start="2024-07-01",
            end="2024-07-01", as_of="2024-07-02T00:00:00+00:00", source="s", kind="bars",
            timeframe="1d", adjustment="none",
        ),
        row_count=1, content_hash="h",
        data_path=Path("snapshots/bars/legacyid00000000/bars.parquet"),
        created_at="2024-07-02T00:00:00+00:00", storage_format="parquet",
    )
    SnapshotManifest(tmp_path / "manifest.jsonl").append(legacy)
    with pytest.raises(ValueError, match="legacy"):
        store.read_bars("legacyid00000000")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data_store.py tests/test_data_read_bars.py -q`
Expected: FAIL — new tests error (e.g. `read_bars() got an unexpected keyword argument 'symbols'`) and the updated provenance test fails on `storage_format`.

- [ ] **Step 3: Rewire `ingest_bars` and `read_bars`**

In `algua/data/store.py`:

(a) Update imports — add the new helpers and `empty_bars`:

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
from algua.data.schema import empty_bars, to_bar_schema
```

(b) Replace the body of `ingest_bars` so it builds the canonical frame, hashes it logically, dedups, and writes the partitioned layout (replace the `return self._ingest_parquet(...)` tail):

```python
        metadata = _metadata(
            dataset=Dataset.BARS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.BARS.value,
            timeframe=timeframe,
            adjustment=adjustment,
            source_metadata=source_metadata,
        )
        canon = to_bar_schema(frame).reset_index().rename(columns={"timestamp": "ts"})
        content_hash = logical_bars_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id
        write_partitioned_bars(
            canon.sort_values(["symbol", "ts"]), self.data_dir / relative_path
        )
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet_dataset",
        )
        self.manifest.append(rec)
        return rec
```

(c) Replace `read_bars` with the filtered, guarded version:

```python
    def read_bars(
        self,
        snapshot_id: str,
        *,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read a bars snapshot as a bar-schema DataFrame, pushing `symbols` + half-open
        `[start, end)` filters down to the partitioned parquet dataset (issue #130). Any filter left
        as None is unbounded. Empty result => the contract's empty-but-typed frame."""
        rec = self.get_snapshot(snapshot_id)  # raises SnapshotNotFound
        if rec.dataset != Dataset.BARS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not {Dataset.BARS.value!r}"
            )
        if rec.storage_format != "parquet_dataset":
            raise ValueError(
                f"snapshot {snapshot_id} is a legacy single-file bars snapshot "
                f"({rec.storage_format!r}); re-ingest under the partitioned layout"
            )
        raw = read_partitioned_bars(
            self.data_dir / rec.data_path, symbols=symbols, start=start, end=end
        )
        if raw.empty:
            return empty_bars()
        return to_bar_schema(raw)
```

(d) Delete the now-unused `_normalize_bar_frame` function at the bottom of the file (its only caller was `ingest_bars`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data_store.py tests/test_data_read_bars.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/data/store.py tests/test_data_store.py tests/test_data_read_bars.py
git commit -m "feat(data): partitioned ingest_bars + pushdown read_bars with legacy guard (#130)"
```

---

### Task 5: `StoreBackedProvider.get_bars` delegates with pushdown

**Files:**
- Modify: `algua/data/serve.py`
- Test: `tests/test_data_serve.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_serve.py`:

```python
def test_get_bars_prunes_to_requested_symbol(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)  # AAA/BBB across 2024-07-01..07-03
    provider = StoreBackedProvider(store, rec.snapshot_id)
    out = provider.get_bars(
        ["BBB"], datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 3, tzinfo=UTC), "1d"
    )
    validate_bars(out)
    assert set(out["symbol"]) == {"BBB"}
    assert out.index.max() == pd.Timestamp("2024-07-02", tz="UTC")  # half-open end


def test_get_bars_naive_datetimes_are_treated_as_utc(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    provider = StoreBackedProvider(store, rec.snapshot_id)
    out = provider.get_bars(
        ["AAA"], datetime(2024, 7, 1), datetime(2024, 7, 2), "1d"  # naive
    )
    validate_bars(out)
    assert out.index.max() == pd.Timestamp("2024-07-01", tz="UTC")
```

- [ ] **Step 2: Run the tests to verify they fail or pass**

Run: `uv run pytest tests/test_data_serve.py -q`
Expected: the two new tests FAIL only if delegation is wrong; if the old `get_bars` body still materializes everything they may pass by accident. Proceed to Step 3 to make delegation explicit, then both must pass alongside the existing serve tests.

- [ ] **Step 3: Rewrite `StoreBackedProvider.get_bars` to delegate**

Replace the body of `get_bars` in `algua/data/serve.py` (keep the class docstring and `__init__`):

```python
    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        rec = self.store.get_snapshot(self.snapshot_id)
        if timeframe != rec.metadata.timeframe:
            raise ValueError(
                f"snapshot {self.snapshot_id} is timeframe {rec.metadata.timeframe!r}, "
                f"not {timeframe!r}"
            )
        # Filtering (symbol pruning + half-open [start, end) on ts, with naive->UTC normalization)
        # is pushed down to the partitioned dataset in read_bars — no full-snapshot materialization.
        return self.store.read_bars(self.snapshot_id, symbols=symbols, start=start, end=end)
```

If `import pandas as pd` becomes unused after this edit, leave it — it is still referenced by the `-> pd.DataFrame` return annotation. Ruff will confirm in the gate.

- [ ] **Step 4: Run the serve tests + the full data suite**

Run: `uv run pytest tests/test_data_serve.py tests/test_data_read_bars.py tests/test_data_store.py -q`
Expected: PASS (all, including the pre-existing boundary/timeframe tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/serve.py tests/test_data_serve.py
git commit -m "feat(data): StoreBackedProvider.get_bars delegates to pushdown read (#130)"
```

---

### Task 6: Update CLI assertion + full gate

**Files:**
- Modify: `tests/test_cli_data.py`

- [ ] **Step 1: Update the storage-format assertion**

In `tests/test_cli_data.py`, in `test_data_ingest_bars_with_provider`, change:

```python
    assert out["snapshot"]["storage_format"] == "parquet"
```
to:
```python
    assert out["snapshot"]["storage_format"] == "parquet_dataset"
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS — all tests green. If any other test ingested bars and asserted the single-file `storage_format` or read `data_path` as a single file, fix it the same way (point `pd.read_parquet` at the directory; expect `parquet_dataset`). Investigate each failure rather than weakening an assertion.

- [ ] **Step 3: Run the complete gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: tests pass, ruff clean, mypy clean, lint-imports "0 broken".

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_data.py
git commit -m "test(data): partitioned bars storage_format assertion (#130)"
```

---

## Self-review notes (author)

- **Spec coverage:** partitioned-by-symbol layout (Task 2/4), logical content hash (Task 1), predicate-pushdown read + symbol pruning + half-open boundary (Task 2/4/5), empty-but-typed frame (Task 3/4), explicit Arrow UTC typing in the filter literal (`_ts_scalar`, Task 2), legacy-layout guard (Task 4), `DataProvider` signature unchanged (Task 5), engine boundary intact (pyarrow stays in `algua.data`; no import-linter change needed — verified in the gate). As-of and incremental composition are out of scope, as specced.
- **Scale honesty:** `read_partitioned_bars` returns a materialized pandas frame bounded by the *filtered* result — the spec's stated contract; constant-memory serving stays deferred.
- **`file_count`:** not persisted as a manifest field (avoids a model schema change / unused-field cruft); `storage_format="parquet_dataset"` is the operator-visible layout marker, and the count is derivable via `rglob` if ever needed. Minor refinement of the spec's "record file_count" note.
- **Type consistency:** `logical_bars_hash`, `write_partitioned_bars`, `read_partitioned_bars`, `_ts_scalar`, `empty_bars`, `BARS_FILE_HASH_COLUMNS`, `storage_format="parquet_dataset"` are named identically everywhere they appear.
```
