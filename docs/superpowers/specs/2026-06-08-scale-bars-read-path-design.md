# Scale the bars read path — partitioned dataset + predicate-pushdown serving

**Status:** Approved (design). **Date:** 2026-06-08. **Issue:** #130.
**Gate:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Problem

The serving path loads exactly ONE snapshot fully into RAM. `StoreBackedProvider` is pinned to a
single `snapshot_id`; `get_bars` does `pd.read_parquet(<one bars.parquet>)` then filters in pandas
(`algua/data/serve.py`). There is no partitioning, no predicate pushdown, and no lazy read. A
narrow query (a handful of symbols over a sub-window) still materializes the entire snapshot, so
26 years of bars — or the many per-symbol/per-year files issue #129 will produce — cannot be
served. **Scale (RAM/read-amplification) is the live driver.**

## Scope

In scope (the scale fix only):
- A **hive-partitioned, directory-shaped bars snapshot** partitioned **by `symbol`**.
- A **predicate-pushdown read path** (pyarrow dataset) that pushes `symbols` + half-open
  `[start, end)` down to the reader, scanning only matching fragments.

Explicitly **deferred** (do not build here):
- **As-of reads** (the `as_of` param the bar-schema contract reserves) — a separable contract
  change with no current data-revision driver. Tracked for its own issue.
- **Incremental multi-snapshot composition** (growing a served identity by appending files over
  time) — the atomic directory-snapshot below covers today's need.
- **Constant-memory / chunked serving** — see "Scale: the honest contract" below.

## Invariants preserved (must not break)

1. **Bar-schema conformance** on the read path — `validate_bars` still passes; the frozen contract
   (`docs/contracts/bar-schema.md`) and `DataProvider.get_bars` signature are **unchanged** (no
   `as_of` here).
2. **Half-open `[start, end)`** look-ahead-safe boundary — a bar timestamped exactly at `end` is
   never returned; the `start` bar is.
3. **Reproducibility** — a served snapshot still pins to one hashed, manifested `snapshot_id`.
4. **Determinism** — an identical input frame yields an identical `snapshot_id` (the store dedups
   on it).
5. **Engine boundary** — the engine depends only on the `DataProvider` protocol; `pyarrow` stays
   inside `algua.data`. import-linter unaffected.

## Design

### 1. Storage layout — atomic directory-snapshot, partitioned by symbol

A bars snapshot stops being a single `bars.parquet` and becomes a directory:

```
<data_dir>/snapshots/bars/<snapshot_id>/
  symbol=AAPL/part-0.parquet
  symbol=MSFT/part-0.parquet
  ...
```

`symbol` is a hive partition key in the path; each parquet file holds only
`ts, open, high, low, close, adj_close, volume` and is **sorted by `ts`** so its row-group
statistics prune time sub-windows. Partition keys: **`symbol` only** — backtests fetch a small
universe over a long window, so `symbol` is the dominant filter; directory-level symbol pruning
reaches exactly the N requested files without reading any other symbol's footer. (Adding a `year`
sub-partition was rejected: for daily bars it produces ~26× more, trivially-tiny files — the
small-files anti-pattern — for little gain, since the time axis is handled well enough by
row-group statistics within each symbol file.)

The snapshot is **atomic**: one ingest registers the whole directory as one snapshot under one
`snapshot_id`. There is no incremental append (deferred).

### 2. Content identity — a *logical* hash (not physical bytes)

`content_hash` is computed over the **canonicalized logical rows**, independent of physical file
layout and pyarrow version:

- Sort rows by `(timestamp, symbol)`.
- For each row, emit a fixed byte encoding: `timestamp` as int64 **nanoseconds since epoch, UTC**;
  `symbol` as UTF-8 (length-prefixed); each of `open, high, low, close, adj_close, volume` as its
  IEEE-754 `float64` little-endian bytes.
- `content_hash = sha256(concatenation)`.

`_snapshot_id` folds `content_hash` in exactly as today (formula unchanged), so dedup and the
`snapshot_id` identity keep working.

**Why logical, not a file-bytes merkle (GATE-1 HIGH #1/#2):** `pyarrow.dataset.write_dataset` is
threaded (row order within a file is non-deterministic without `preserve_order`) and parquet
footers carry writer-version/encoding metadata that shifts across the allowed `pyarrow>=15,<24`
range. Hashing physical bytes would make `snapshot_id` non-reproducible and break dedup. A logical
hash is version- and layout-independent and future-proofs the deferred compaction/incremental
work. (The full Iceberg-style logical/layout/manifest 3-hash split is deliberately **not** adopted
— one logical identity hash suffices for the atomic-snapshot model.)

### 3. Write path

- `ingest_bars` sorts the frame by `(symbol, ts)`, computes the logical `content_hash`, derives
  `snapshot_id`, dedups against the manifest, then writes via a new shared helper
  `write_partitioned_bars(frame, dest_dir)`.
- `write_partitioned_bars` calls `pyarrow.dataset.write_dataset` with an **explicit Arrow schema**
  (`ts: timestamp[ns, tz=UTC]`, `symbol: string`, OHLCV `float64`), `partitioning=["symbol"]`
  (hive), and a fixed `basename_template`. Writer threading/order does not affect identity (the
  hash is logical), but rows are pre-sorted by `ts` so each file's row-group stats are monotonic.
- #129's bulk-ingest adapter reuses `write_partitioned_bars`, so the on-disk layout has one owner.
- `ingest_universe` is untouched (still one small parquet).
- Manifest accounting (GATE-1 LOW #9): `data_path` now points at the snapshot **directory**;
  `storage_format = "parquet_dataset"` (distinct from the single-file `"parquet"`); `row_count`
  comes from `len(frame)` at write time; record `file_count`.

### 4. Read path — predicate pushdown + symbol pruning

`store.read_bars(snapshot_id, *, symbols, start, end)` (filters now part of the signature):

1. Resolve the record; reject a bars record lacking the new `parquet_dataset` layout marker with a
   clear "legacy single-file bars snapshot — re-ingest under the partitioned layout" error
   (GATE-1 LOW #10).
2. Open `pyarrow.dataset.dataset(dir, format="parquet", partitioning=<hive: symbol: string>)` with
   an **explicit partition schema** so `symbol` returns as `string`, never dictionary/categorical
   (GATE-1 MEDIUM #6).
3. Normalize `start`/`end` to UTC; build the pushdown filter from pyarrow scalars typed
   `timestamp[ns, tz=UTC]` (GATE-1 MEDIUM #4):
   `symbol ∈ symbols ∧ ts >= start ∧ ts < end`.
   No year-bound derivation is needed (symbol-only partitioning). The `ts` predicate is the sole
   authority for the half-open boundary; symbol pruning happens at the directory level.
4. `to_table(columns=[ts, symbol, open, high, low, close, adj_close, volume])` → pandas, then
   `to_bar_schema` rebuilds the UTC `timestamp` index, orders columns, sorts, and `validate_bars`.
5. **Empty result** (GATE-1 MEDIUM #5): a named requirement — if the scan returns zero rows,
   `read_bars` returns the contract's empty-but-typed frame built from `BAR_COLUMNS` and
   `DatetimeIndex([], tz="UTC", name="timestamp")`, not whatever an empty `to_pandas()` yields.

`StoreBackedProvider.get_bars` keeps its exact signature, still pins to one `snapshot_id`, still
validates `timeframe` against `rec.metadata.timeframe`, but delegates the filtering to the
pushed-down `read_bars` instead of read-all-then-filter-in-pandas.

### 5. Scale: the honest contract (GATE-1 HIGH #3)

This slice bounds reads to **the filtered result set, not constant memory**. It eliminates *read
amplification* — a narrow query no longer loads the whole snapshot. But the frozen
`DataProvider.get_bars` returns a materialized pandas `DataFrame`, so a query that selects the full
universe over full history still materializes it. Constant-memory / batched serving requires a
`DataProvider` contract change and a chunked backtest backend — that is the spec's open
`vectorbt operating envelope` question and is **out of scope here**. The spec/docs state this
precisely rather than claiming constant memory.

## Decomposition / build order

A safe single slice — it changes both write and read paths together so the layout is exercised
end-to-end by round-trip tests (no half-landed contract). The atomic directory-snapshot does not
corner the deferred work: as-of and incremental append layer on top of the manifest, and the
logical `content_hash` already tolerates physical re-layout/compaction without identity churn.

## Testing

- **Round-trip:** ingest a multi-symbol, multi-year frame → filtered read equals the expected
  rows; result passes `validate_bars`.
- **Half-open boundary:** the `start` bar is included; a bar exactly at `end` is excluded;
  nanosecond-precision edge.
- **Symbol pruning correctness:** a sub-universe query returns only requested symbols.
- **Determinism / dedup:** the same input frame yields the same `snapshot_id`; re-ingest dedups.
- **Logical-hash stability:** the hash is invariant to physical write order (e.g. shuffled input
  rows that canonicalize identically produce the same `content_hash`).
- **Empty result:** out-of-window / unknown-symbol query returns the exact empty-but-typed frame.
- **Symbol dtype:** reconstructed `symbol` is `str` (not categorical); `year` never appears (it is
  not a partition key here) and no extra columns leak past `algua.data`.
- **`timeframe` mismatch** still raises.
- **Legacy guard:** a record marked single-file `parquet` raises the clear re-ingest error.
- **`DataProvider` protocol conformance** of `StoreBackedProvider` unchanged.

## Out of scope (tracked)

- As-of reads (`as_of` param + `validate_bars`/contract/consumers update) — own issue.
- Incremental multi-snapshot composition.
- Constant-memory / chunked serving (DataProvider contract change; vectorbt-envelope open
  question).
