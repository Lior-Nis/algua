# External bulk-parquet ingestion — FirstRateData → bar-schema (Issue #129, slice 1)

**Status:** Design (pending GATE-1 review + user sign-off). **Date:** 2026-06-08.
**Issue:** #129 — External bulk-parquet ingestion adapter. **Relates to:** #130 (read-path scaling).

## Problem

The data layer can fetch bars over the network (`BarProvider`: yfinance, Alpaca) but has **no
import-external-file-and-normalize path**. `ingest_bars(frame=...)` requires a whole bar-schema
frame in RAM; `ingest_file` stores an opaque blob that `read_bars` then mis-reads as a single
bar-schema parquet. Deep history from vendors like FirstRateData ships as per-symbol local files
that must be mapped to the canonical bar-schema (`docs/contracts/bar-schema.md`) and imported
without loading everything into RAM at once.

## Scope of THIS slice

Import + normalize + provenance + chunked write for **FirstRateData**, `1d` only. Decisions made
during brainstorming:

1. **Defer corporate-action math.** `adj_close` is sourced from the vendor's already-adjusted
   data, not computed by us. Split/dividend computation is a tracked follow-up.
2. **First vendor: FirstRateData.** It sells separate adjustment variants as per-symbol files.
3. **`adj_close` via two-file merge.** Import takes an unadjusted file set (→ raw OHLC) AND an
   adjusted file set (→ `adj_close`), joined on `(timestamp, symbol)`. Mirrors the existing
   Alpaca raw+adjusted merge. A real `adj_close`, no CA math by us.
4. **New `data import-bars` command + new `BarImporter` seam**, parallel to the network-fetch
   `BarProvider`. Own registry mirroring `algua/data/providers/`.
5. **One consolidated snapshot.** All symbols stream into one tidy/long `bars.parquet` =
   one `snapshot_id`, one manifest row.
6. **The operator declares the adjustment flavor** (`--adjustment split_div|split`, default
   `split_div`). We cannot *prove* what FirstRate's adjusted file did, so we never hardcode/assume
   it — the declared value is stamped into provenance verbatim and is what downstream consumers see.
   (GATE-1 #1: mislabeling `adj_close` semantics is a silent-backtest risk.)
7. **On-disk layout is an explicit contract:** symbol-major `(symbol, timestamp)` (see Physical
   layout). Not an accident of streaming — a deliberate, documented choice that constrains #130.

### Explicit non-goals (deferred)
- No split/dividend **computation** (`adj_close` comes from the vendor's adjusted file).
- No Databento adapter (it ships raw OHLC + separate CA dataset → needs the deferred math).
- No intraday timeframes (`1d` only this slice).
- No read-path scaling for multi-GB consolidated snapshots — that is **#130**. This slice is the
  **write side**; `read_bars` continues to read+re-sort the whole file in RAM until #130 lands.

## Architecture

### 1. `BarImporter` seam — `algua/data/importers/`

A file-oriented protocol distinct from `BarProvider` (which has no file path in `BarRequest`):

```python
@dataclass(frozen=True)
class ImportRequest:
    raw_dir: Path
    adjusted_dir: Path
    timeframe: str = "1d"
    as_of: str | None = None
    symbols: tuple[str, ...] | None = None   # optional filter; default = all paired symbols

class BarImporter(Protocol):
    name: str
    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]: ...
```

- `import_bars` **yields one `ProviderBars` per symbol** (existing `ProviderBars(frame,
  source_metadata)` value object reused) — yielding rather than returning one giant frame is what
  bounds RAM for a multi-GB import.
- Registry: `algua/data/importers/__init__.py` with `get_importer(name)` /
  `register_importer(name, factory)`, mirroring `providers/__init__.py`. First entry: `firstrate`.

### 2. FirstRateData adapter — `algua/data/importers/firstrate.py`

- **File discovery / pairing.** Parse the symbol from each filename in `raw_dir` and `adjusted_dir`
  (FirstRate names like `AAPL_full_1day_*.txt`/`.csv`). **Canonicalize each parsed symbol**
  (`normalize_symbols` — strip/upper) *before* building the `{symbol: path}` map.
  - Error if two filenames in one dir canonicalize to the **same** symbol (alias collision —
    GATE-1 #6: this is how a global `(timestamp,symbol)` duplicate would sneak in).
  - Error if the two symbol sets disagree (a symbol present in one dir but not the other).
  - If `request.symbols` is set, restrict to that subset (error on a requested symbol with no file).
  - **Iterate symbols in canonical sorted order** when yielding chunks (GATE-1 #9: directory
    listing order is filesystem-dependent; a fixed order is required for a stable `snapshot_id`).
- **Per-symbol normalization** (`1d`):
  - Read the raw file → `open, high, low, close, volume` (raw, non-null).
  - Read the adjusted file → take its `close` as **`adj_close`** (split+div adjusted).
  - **Join on `(timestamp, symbol)`**; error if the key sets disagree (mirrors Alpaca's
    key-agreement check). Adjusted file's open/high/low/volume are ignored.
  - Daily timestamp (a bare date) → **UTC midnight** (`2024-07-01 00:00:00+00:00`), matching the
    yfinance daily convention. `to_bar_schema` rejects naive timestamps, so the adapter localizes
    explicitly — never silently.
  - **Cheap sanity guard** (GATE-1 #7, partial): reject nonpositive `open/high/low/close/adj_close`
    (≤ 0) — garbled vendor data. Full economic-plausibility checks (constant adj/raw ratio over
    no-corporate-action spans, discontinuity flags) need the **deferred** CA dataset and are a
    tracked follow-up, not this slice.
  - Hand each per-symbol chunk through `to_bar_schema` (per-symbol schema validation: dtypes,
    non-null, tz, within-symbol uniqueness/sort).
- **FirstRate file format** (documented assumption, encoded in a small parser): headerless or
  header CSV with columns `datetime, open, high, low, close, volume`; daily `datetime` is a date.
  The parser is isolated so format quirks (header presence, delimiter) live in one place.

### 3. Streaming store write — `DataStore.ingest_bars_streamed`

New method alongside `ingest_bars`:

```python
def ingest_bars_streamed(
    self, *, provider, symbols, start, end, as_of, source,
    chunks: Iterable[pd.DataFrame], timeframe="1d", adjustment="split_div",
    source_metadata=None,
) -> SnapshotRecord: ...
```

- **Crash-safe staging** (GATE-1 #5). The streamed path can't know `snapshot_id` until the file is
  closed and hashed, so:
  1. Stream chunks into a **staging path** under `snapshots/_staging/<run-tag>/bars.parquet` via
     `pyarrow.ParquetWriter`.
  2. Close the writer, `sha256_file` the finished file → `content_hash` → `snapshot_id`.
  3. `manifest.find(snapshot_id)`: if it exists, **discard staging** and return the existing record
     (idempotent — no orphan in the live tree).
  4. Otherwise **atomically rename** staging → `snapshots/bars/<snapshot_id>/bars.parquet`
     (same-filesystem `os.replace`), then append the manifest row **last**.
  - The CLI cleans **stale `_staging/` dirs** on entry (a crash between write and rename leaves a
    staging dir, never an unmanifested file in the live snapshot tree).
- **Deterministic write** (GATE-1 #2, #9). The `ParquetWriter` is configured to match
  `frame_to_parquet_bytes` (snappy, version `2.6`, stripped schema metadata, `preserve_index=False`)
  **and** pins a fixed `row_group_size` and writes chunks in **canonical sorted-symbol order**, so
  identical inputs yield byte-identical files (hence a stable `snapshot_id`) within a pinned
  pyarrow/compression environment — the same reproducibility assumption the existing
  `frame_to_parquet_bytes` already relies on. *Out of scope (declined):* replacing the
  physical-bytes `content_hash` with a logical-row hash. That would diverge from the existing
  `ingest_bars` content-hash model; a logical-identity key is a separate cross-cutting change.
- After staging, hash + dedup + manifest as above; `row_count` accumulated across chunks.
- `start`/`end`: both **requested** (optional CLI bounds) and **observed** (min/max session date)
  are recorded; see Data flow / provenance.
- `ingest_bars(frame=...)` is unchanged — the in-memory path stays for provider ingestion.

### 3a. Physical layout contract (GATE-1 #3)

The consolidated `bars.parquet` is stored **symbol-major: `(symbol, timestamp)`**, not the
canonical serving order `(timestamp, symbol)`. This is a **deliberate, documented physical-layout
contract**, not an incidental side effect of streaming:

- Writing canonical `(timestamp, symbol)` order would require a **global sort across all symbols**,
  which is unbounded RAM — defeating the streaming/bounded-RAM goal. Symbol-major is the layout that
  a bounded-RAM streamed write *can* produce.
- It is contract-safe **today** because `read_bars` re-sorts via `to_bar_schema` on read; the
  bar-schema governs what `read_bars` *returns*, not on-disk layout.
- **Constraint on #130** (recorded here so it isn't a latent trap): the scaled read path must treat
  this dataset as **symbol-major** — prune via per-symbol row-group statistics and/or a streaming
  k-way merge to produce time-ordered output. It must **not** assume time-major row groups, and it
  must preserve the `validate_bars` contract without full materialization.

### 4. CLI + provenance — `algua/cli/data_cmd.py`

```
algua data import-bars --vendor firstrate \
    --raw-dir PATH --adjusted-dir PATH \
    --timeframe 1d --as-of TS --adjustment split_div \
    [--start D --end D] [--symbols AAPL,MSFT]
```

- Resolves the importer via `get_importer(vendor)`, builds `ImportRequest`, passes the chunk
  iterator to `DataStore.ingest_bars_streamed`, emits the snapshot record as JSON (matching the
  other `data` commands' output shape).
- **Provenance** stamped into `SnapshotMetadata` / `source_metadata`:
  - `provider="firstrate"`, `kind="bars"`, `dataset="bars"`.
  - `adjustment` = the **operator-declared** flavor (`--adjustment`, default `split_div`). We never
    hardcode/assume it — the declared value is recorded verbatim (GATE-1 #1).
  - `as_of` = user-provided (required for reproducibility; CLI errors if absent).
  - `source` = a stable label (e.g. `"firstratedata-import"`).
  - `source_metadata`: `{vendor, raw_dir basename, adjusted_dir basename, symbol count, total
    row count, adjustment, requested_start/end (if given), observed_start/end}` — enough to audit
    the import without the files.
- **Large-snapshot guardrail** (GATE-1 #4, #11). Because the read path still fully materializes a
  snapshot (until #130), an import whose `row_count` exceeds a threshold (e.g. configurable
  `IMPORT_WARN_ROWS`, default ~5M) emits a **stderr warning** and stamps
  `source_metadata["servable"]="deferred-130"` so the snapshot self-documents that it is not safely
  servable by the current read path. **Not a hard cap** — deep bulk history is the point; the
  operator is warned, not blocked.

## Data flow

```
data import-bars --vendor firstrate --raw-dir R --adjusted-dir A --as-of T
  → get_importer("firstrate")
  → FirstRateImporter.import_bars(ImportRequest(R, A, "1d", T, symbols?))
      pair files by symbol across R and A  (validate symbol-set agreement)
      for each symbol:
        raw = parse(R/<sym>)      → ts, open, high, low, close, volume
        adj = parse(A/<sym>)      → ts, close(→adj_close)
        merge on (ts, symbol), validate key agreement
        ts: date → UTC midnight
        yield ProviderBars(to_bar_schema(chunk).reset_index(), source_metadata)
  → DataStore.ingest_bars_streamed(chunks=…, provider="firstrate", adjustment=<declared>, as_of=T,
        requested_start/end=<optional>, source_metadata=…)
      stream chunks (sorted-symbol order) → snapshots/_staging/<run-tag>/bars.parquet
      sha256_file → content_hash → snapshot_id
        → exists? discard staging, return existing record (idempotent)
        → else os.replace(staging → snapshots/bars/<snapshot_id>/bars.parquet) → manifest.append
  → JSON snapshot record on stdout
```

**`start`/`end` provenance** (GATE-1 #8). The authoritative metadata `start`/`end` are the
**observed** min/max session dates (what actually parsed). When the operator passes `--start/--end`
(**requested** bounds), they are recorded in `source_metadata` and the **observed** range is
validated against them — a mismatch (observed coverage narrower than requested) **errors** unless
explicitly overridden, so a truncated vendor file can't masquerade as complete. Recording both
means observed-only imports stay easy while explicit imports get a coverage check.

The half-open `[start, end)` window is **not** re-clipped at import — consistent with the existing
convention that adapters don't re-clip and the serving read path enforces the boundary.

## Error handling

- Symbol-set disagreement between `raw_dir`/`adjusted_dir` → `ValueError` naming the offending
  symbols (don't silently drop).
- Two filenames in one dir canonicalizing to the same symbol → `ValueError` (alias collision).
- `(timestamp, symbol)` key disagreement between a symbol's raw and adjusted files → `ValueError`.
- Nonpositive OHLC/`adj_close` in a row → `ValueError` (garbled data).
- Naive/unparseable timestamps → surfaced by `to_bar_schema` (rejected, never localized silently).
- `timeframe != "1d"` → `ValueError` ("intraday import not yet supported").
- Missing `--as-of` → CLI usage error.
- Requested `--start/--end` not covered by observed data (without override) → `ValueError`.
- Empty dir / no paired symbols → `ValueError` (no empty snapshot written; staging cleaned up).
- Unknown vendor → `ValueError` from `get_importer` (mirrors `get_provider`).

## Testing

- **FirstRate adapter unit tests** (`tests/data/test_firstrate_importer.py`): pairing logic;
  raw+adjusted merge produces correct `adj_close`; date → UTC-midnight; key-disagreement error;
  symbol-set-disagreement error; alias-collision error; nonpositive-price rejection; `--symbols`
  subset filter; bad timeframe error. Fixtures are small synthetic FirstRate-format files under
  `tmp_path`.
- **Streaming store tests** (`tests/data/test_ingest_streamed.py`): multi-symbol chunk iterator →
  one snapshot; `read_bars` round-trips to canonical bar-schema order (validates the symbol-major
  on-disk layout end-to-end); idempotency (same inputs → same `snapshot_id`); **shuffle-invariance**
  (shuffled filesystem discovery order → same `snapshot_id`, GATE-1 #9); staging cleanup +
  atomic-rename leaves no orphan on a simulated mid-write failure (GATE-1 #5); `row_count` /
  observed `start`/`end` correctness; requested-vs-observed bounds mismatch error (GATE-1 #8).
- **Shared producer conformance** (`tests/data/test_bar_producer_conformance.py`, GATE-1 #10): a
  parametrized suite asserting **both** a `BarProvider` (synthetic) and the `BarImporter` terminate
  at `to_bar_schema`-valid output with the same provenance vocabulary — guards against the two
  normalization paths drifting.
- **CLI test** (`tests/cli/test_cli_import_bars.py`): `data import-bars` happy path emits a valid
  JSON snapshot record; provenance fields present (incl. declared `adjustment`); large-row warning +
  `servable` flag stamped; errors surface as non-zero exit with a message. Uses `CliRunner` +
  `ALGUA_DATA_DIR` tmp pattern (existing convention).
- **Serving smoke**: ingest via the streamed path, then `StoreBackedProvider.get_bars` returns
  bar-schema-valid frames with the half-open window honored.

## Boundaries / import-linter

`algua/data/importers/*` stays within the data layer (may import `algua.data.contracts`,
`algua.data.schema`, pandas, pyarrow). No imports from `features`/`strategies`/`backtest`. The
bar-schema contract is unchanged — no `validate_bars` edits, so no coordinated contract change.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## Appendix — GATE-1 design-review triage (Codex, 2026-06-08)

11 findings; triaged on merit. **Folded in:** #1 (operator-declared `adjustment`, never hardcoded),
#3 (explicit symbol-major physical-layout contract + #130 constraint), #4/#11 (large-snapshot
stderr warning + `servable` metadata flag, no hard cap), #5 (staging → hash → atomic-rename →
manifest-last + stale-staging cleanup), #6 (canonical symbol dedup → global uniqueness), #8
(record requested + observed bounds, validate on mismatch), #9 (sorted-symbol write order for
stable `snapshot_id`), #10 (shared producer-conformance test). **Partially accepted:** #2 — pinned
deterministic write accepted; the logical-row-hash *redesign* **declined** (would diverge from the
existing physical-bytes `content_hash` model used by `ingest_bars`; a logical-identity key is a
separate cross-cutting change, not this slice). #7 — cheap nonpositive-price guard accepted; full
economic-plausibility checks (constant adj/raw ratio over no-action spans) **deferred** with the CA
math, since they need the corporate-action dataset this slice explicitly omits.

**Tracked follow-ups (deferred, not dropped):** corporate-action math (compute `adj_close` from
splits/dividends); economic-plausibility validation of `adj_close`; #130 read-path scaling honoring
the symbol-major layout above.
