# Make streamed bars ingest servable: partitioned write + commit protocol (#153)

**Status:** design approved (after GATE-1 design review: Codex + Gemini)
**Issue:** #153 (#129 × #130 interaction)
**Date:** 2026-06-08

## Problem

`DataStore.ingest_bars_streamed` (the FirstRate bulk-import path) writes one consolidated
`bars.parquet` with `storage_format="parquet"`. `read_bars` (post-#130) only serves the
hive-partitioned-by-symbol `parquet_dataset` and **raises** for single-file `parquet`. So streamed
snapshots are unreadable, and `test_streamed_ingest_one_snapshot_reads_canonical` is red on `main`.

## Approach (option a): write the partitioned layout from the stream

The partition key is `symbol` and the stream delivers **exactly one symbol per chunk** (cross-chunk
symbol uniqueness already enforced), so each chunk maps 1:1 to a `symbol=<SYM>/` partition. Write
each chunk's partition into the UUID staging dir (one chunk in memory at a time → bounded RAM
preserved), then commit the staging *directory* into the immutable snapshot path. `storage_format`
becomes `parquet_dataset`; `read_bars` serves it unchanged with pushdown. Rejected option (b)
(teach `read_bars` to serve single-file) loses partition-by-symbol pruning for the largest snapshots.

## Decisions from the GATE-1 review (Codex + Gemini)

### 1. Commit protocol — adopt-on-target-exists (was CRITICAL)
`os.replace` of a *directory* onto an existing non-empty target **fails** (unlike single-file rename,
which overwrites). So a kill between rename and manifest-append would brick idempotent re-ingest, and
two concurrent ingests of the same id would race. Because the target dir name **is** the content hash
(content-addressed), a pre-existing target holds identical content. Protocol:
- compute `snapshot_id`; `existing = manifest.find(id)`; if found → discard staging, return existing.
- else attempt the commit; if the target dir already exists (orphan from a prior crash, or a
  concurrent winner) → re-check the manifest; if a record now exists return it, else **adopt** the
  existing dir by appending the manifest record. Discard staging either way.
- manifest appended **last** (unchanged ordering). This makes ingest self-healing — no separate GC.

### 2. Deterministic, streamable identity (was HIGH)
Compose per-symbol leaf hashes **sorted by symbol** (NOT arrival order — the store enforces
symbol-uniqueness, not order, so arrival-order composition would be non-deterministic). Each leaf =
`logical_bars_hash` of that symbol's chunk (layout-independent). Parent digest is domain-separated
and versioned: update with a version tag, leaf count, then per leaf `(len-prefixed symbol, row_count,
32 leaf digest bytes)` in symbol order. Streamable: only `(symbol, row_count, leaf_digest)` per
symbol is held (tiny).

### 3. Self-describing identity (was HIGH)
Record a `content_hash_algorithm` marker (e.g. `bars-symbol-merkle-v1`) in `source_metadata` so the
hash's meaning is unambiguous across the three routes (single-file bytes / global logical / this
composed form). This resolves the reviewers' "unify identity" concern **without** changing the
working in-memory `ingest_bars` route.

### 4. Per-chunk partition write mechanics (was MEDIUM)
Write each one-symbol chunk to its own partition deterministically (assert one symbol per chunk
before writing). Bound row-group size so `[start,end)` pushdown prunes within a symbol; chunks are
already ts-sorted per symbol.

### 5. Cleanup (was LOW)
Drop the now-obsolete `servable=deferred-130` flag, the `_ingest`-time warn branch, and the CLI
warning in `data_cmd.py`; update the test that asserted the flag. Add a `BRK.B`-style symbol
partition round-trip test (hive path safety).

## Declined / deferred (with rationale)
- **Migration of existing single-file snapshots** — VERIFIED there are no persisted single-file bars
  snapshots on disk (only ephemeral test `tmp_path` dirs); the break is vacuous. Re-ingest is the
  remedy. No migration command (YAGNI).
- **Cross-route identity unification** — cross-route dedup never fires in practice (streamed =
  vendor files, in-memory = a fetch → different bytes anyway); unifying would change the working
  `ingest_bars` ids for a theoretical benefit. The `content_hash_algorithm` marker (3) covers the
  real concern.
- **fsync / power-loss durability** — the store is SIGKILL-safe, not power-loss-safe (the manifest
  append isn't fsync'd either); staying consistent. Out of scope.
- **Universe snapshots** keep single-file `parquet` (read via `read_universe`) — untouched.

## Testing (TDD)
- `test_streamed_ingest_one_snapshot_reads_canonical` (existing red) → green: streamed snapshot is a
  `parquet_dataset` and reads back canonically.
- read-back with `symbols=`/`[start,end)` pushdown on a streamed snapshot.
- idempotent re-ingest → same id, one manifest record.
- **adopt-on-orphan:** simulate a committed target dir with no manifest record → re-ingest adopts it
  (one record, readable), does not raise.
- deterministic id regardless of chunk arrival order (feed symbols in two orders → same id).
- `BRK.B` symbol round-trips through the hive partition.
- `servable` flag gone; CLI no longer warns.

Gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
