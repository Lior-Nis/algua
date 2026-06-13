# Power-loss durability for snapshot payloads — issue #184 (design)

Status: draft for review (GATE 1)
Date: 2026-06-13
Issue: #184 (deferred from #158 / PR #177)

## Problem

#158 made the snapshot store safe under the **process-crash** threat model: payloads
publish staged + atomic-rename *before* the manifest append, and the manifest append is
itself `fsync`'d. Page cache and completed renames survive process death, so a crashed
process never leaves the manifest naming a half-written payload.

What #158 deliberately did NOT cover is **power loss / kernel crash**. Three classes of
bytes are left in page cache, un-`fsync`'d, at the moment the manifest commits:

1. **Payload file contents** — the temp file in `write_bytes_snapshot`, the `shutil.copy2`
   staging copy in `ingest_file`, and the parquet part-files pyarrow writes into a
   partitioned staging tree.
2. **Staging-dir trees** — the `symbol=<SYM>/` subdirectories and their entries for a
   partitioned bars dataset.
3. **Parent-directory rename entries** — after `os.replace`, the new name in the target's
   parent directory is a dirty dir-entry; the manifest's own parent dir on first creation
   and after repair-by-rename has the same gap.

After a power cut, the manifest (which IS durable) can name a snapshot whose payload bytes
or directory entry never reached stable storage.

### Threat model (unchanged from #158)

Single Linux host, cooperative CPython writers, local filesystem. We defend against
**truncation / missing-file / torn-footer** failure modes — exactly what an unclean
power loss produces (page-cache contents that were never written back). We do **not**
defend against silent mid-page bit-flips (media rot); those are out of the power-loss
threat model and would require a full content-hash recompute on every read, which is
overengineering for a single-host, fully re-ingestable research store.

### Urgency

Low: single-host research workloads, all data re-ingestable from vendors. Becomes
load-bearing if the data dir ever holds non-reproducible data (the `ingest_file`
registration path is the one such case today).

## Part 1 — Write-side fsync discipline

### New primitives (`algua/data/files.py`)

```python
def fsync_file(path: Path) -> None:
    """fsync a file's data to stable storage (open O_RDONLY, fsync, close)."""

def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation entry within it is durable."""

def fsync_tree(root: Path) -> None:
    """Bottom-up fsync of every regular file, then every subdir, then `root` itself.
    For partitioned staging trees whose part-files pyarrow wrote without exposing a
    handle we can fsync directly."""
```

`fsync_file`/`fsync_dir` both `os.open(path, os.O_RDONLY)` → `os.fsync(fd)` → close. On
Linux, `fsync` on a read-only descriptor still flushes the inode's dirty pages (files) or
directory entries (dirs). `fsync_tree` walks `root` with `os.walk(topdown=False)`,
fsyncing files then directories so child durability precedes parent.

### The invariant

At every publish site: **payload bytes durable → `os.replace` → parent-dir entry durable**,
all completed *before* the manifest append (which remains the single commit point). A power
cut between payload-durable and manifest-commit leaves an orphan payload — harmless, and
already handled by #158's re-adopt (dirs) / atomic-replace (files) on the next same-id
ingest.

### Publish-site changes

| Site | File | Change |
|---|---|---|
| `write_bytes_snapshot` (universe / fundamentals / news single-file) | `files.py` | `fh.flush()` + `os.fsync(fh.fileno())` on the temp fd before `os.replace`; `fsync_dir(target_path.parent)` after |
| `_commit_bars_dir` (partitioned bars — both `ingest_bars` and `ingest_bars_streamed`) | `store.py` | `fsync_tree(staging_dir)` before `os.replace`; `fsync_dir(target.parent)` after the rename. The adoption branch (target already exists, `ENOTEMPTY`) renames nothing and needs no fsync. |
| `ingest_file` (`shutil.copy2` → `os.replace`) | `store.py` | `fsync_file(staged)` before `os.replace` (copy2 does not fsync); `fsync_dir(target.parent)` after |
| `SnapshotManifest._repair` (repair-by-rename) | `manifest.py` | `fsync_dir(self.path.parent)` after `os.replace` (the temp is already fsync'd) |
| `SnapshotManifest.append_if_absent` | `manifest.py` | `fsync_dir(self.path.parent)` **only when the manifest file did not previously exist** (first creation makes a new dir-entry; plain appends to an existing inode do not change the parent dir, and the content fsync already covers them). The pre-read already tells us whether the file existed. |

No new ordering dependencies: payload publish already precedes the manifest append in
every ingest method, so the durability fsyncs slot into the existing sequence.

### Why `append_if_absent`'s parent-dir fsync is conditional

`fsync(file)` already makes the appended bytes and the inode's size durable. Appending to
an *existing* manifest does not change its parent directory entry, so no parent-dir fsync
is needed there. Only the **first** creation of `manifest.jsonl` adds a dir-entry that must
be flushed. Gating on "did the file exist before this call" keeps the per-commit hot path
at exactly one fsync (the content fsync) while still closing the creation gap.

## Part 2 — Read backstop: `algua data verify`

A deliberate, operator/agent-invoked command — **not** an inline hot-path check (inline
full validation would read every footer on every read and defeat `read_bars`' symbol/time
pushdown). Hot reads keep pushdown and already raise naturally if a *touched* partition's
footer is torn; single-file reads read the whole file and already raise in pyarrow on a
torn file. `data verify` is the explicit backstop to run after an unclean shutdown or on
demand.

### CLI

```
uv run algua data verify [--snapshot-id ID]
```

- `--snapshot-id ID` → verify that one snapshot. Omitted → verify **all** committed
  snapshots.
- Emits JSON on stdout (the algua data-command contract):
  ```json
  {"verified": 12, "failed": 1,
   "snapshots": [{"snapshot_id": "…", "dataset": "…", "storage_format": "parquet_dataset",
                  "ok": false, "error": "adoption validation failed …"}]}
  ```
- **Fails closed:** non-zero exit if any snapshot fails (a damaged payload must not be
  reported as healthy).

### Validation dispatch (metadata-only, by `storage_format`)

| `storage_format` | Check | Catches |
|---|---|---|
| `parquet_dataset` (partitioned bars) | `validate_partitioned_bars_dir(target, expected_row_count=rec.row_count or 0, expected_symbols=set(rec.metadata.symbols))` — the existing #158 helper: every file matches `symbol=<SYM>/part-*.parquet`, every footer parses, summed rows == expected, symbol set == expected | missing part-file, torn footer, foreign/partial dir |
| `parquet` (single-file: universe / fundamentals / news, and `.parquet` via `ingest_file`) | new `validate_parquet_file(path, expected_row_count)`: `pq.ParquetFile(path).metadata.num_rows` (raises on a torn footer) must equal `rec.row_count` | truncated / torn single-file parquet |
| anything else (`ingest_file` csv / generic `file`) | `sha256_file(path) == rec.content_hash` — exact, and the way `ingest_file` computed `content_hash` in the first place; this is the one non-reproducible path so the full-file read is justified | any truncation / tampering of a registered file |

`validate_partitioned_bars_dir` and `sha256_file` already exist. `validate_parquet_file`
is a new small helper in `files.py`. All three are metadata-or-single-pass; none recompute
a logical content hash for parquet (footer + row-count is the right power-loss detector and
keeps the bars case from reading data pages).

Note on symbol-set expectation: `expected_symbols` is taken from `rec.metadata.symbols`.
If a snapshot's recorded symbol list proves not to round-trip the partition set exactly,
the row-count check still catches truncation; the implementation must confirm
`metadata.symbols` holds the full partition set for both `ingest_bars` and
`ingest_bars_streamed` (and pass it, or drop the symbol-set arg, accordingly).

## Out of scope (deferred, with rationale)

- **Full content-hash recompute on read** — catches silent media bit-flips, outside the
  power-loss threat model; expensive (full read of every snapshot). Parquet's own page
  checksums already guard the hot read for the touched data.
- **Inline-on-every-read validation** — defeats bars pushdown; the explicit command is the
  backstop.
- **fsync of the manifest sidecar lock file** — the lock conveys no committed state; losing
  it after power loss is benign (recreated on next acquire).
- **Orphan-payload GC** — unchanged from #158 (re-adopt / atomic-replace on next same-id
  ingest; `clear_staging` sweeps staging residue).
- **NFS / remote-FS** — threat model is local FS (doc note from #158 stands).

## Tests

### Write-side (placement + ordering; power loss itself cannot be unit-tested without a
fault injector — we assert the fsyncs happen at the right place/order, and that existing
atomicity/idempotence behavior is unchanged)

1. **Helper correctness** — `fsync_file` / `fsync_dir` / `fsync_tree` run without error on
   real paths; `fsync_tree` visits nested `symbol=<SYM>/part-*.parquet` (files fsync'd
   before their parent dir) — assert via a spy recording `(path, is_dir)` in walk order.
2. **`write_bytes_snapshot`** — spy on `os.fsync` + `os.replace`: the temp file is fsync'd
   *before* the replace and the parent dir *after*; no temp residue (existing test still
   green).
3. **`_commit_bars_dir`** — `fsync_tree(staging_dir)` completes before `os.replace`, and
   `fsync_dir(target.parent)` after; the adoption branch performs no rename-fsync.
4. **`ingest_file`** — the staged copy is fsync'd before `os.replace`, parent dir after.
5. **manifest `_repair`** — parent dir fsync'd after the repair replace; committed records
   intact (extends the existing repair test).
6. **manifest `append_if_absent`** — parent dir fsync'd on first manifest creation; **not**
   fsync'd on a subsequent append to the existing file (assert the create-vs-append gating).

### Read-side (`data verify` — real behavioral tests)

7. Healthy partitioned-bars snapshot → `ok: true`, exit 0.
8. Torn part-file (truncate a `part-*.parquet`) → `ok: false`, non-zero exit, error names
   the dir.
9. Missing part-file (delete one) → row-count mismatch → fails closed.
10. Healthy single-file parquet (fundamentals / news / universe) → ok.
11. Truncated single-file parquet (cut the footer) → fails closed.
12. `ingest_file` generic file (csv) tampered after ingest → `sha256_file` mismatch → fails.
13. `verify` with no `--snapshot-id` over a mix of healthy + one damaged snapshot →
    aggregate counts correct, exit code reflects the failure.
14. `verify --snapshot-id` on an unknown id → `SnapshotNotFound` surfaced as an error
    (non-zero exit), not a crash.

## Files touched

- `algua/data/files.py` — `fsync_file`, `fsync_dir`, `fsync_tree`, `validate_parquet_file`;
  fsyncs inside `write_bytes_snapshot`.
- `algua/data/store.py` — `_commit_bars_dir`, `ingest_file` fsyncs; `verify_snapshot(s)`
  read-side method dispatching by `storage_format`.
- `algua/data/manifest.py` — parent-dir fsyncs in `_repair` and (conditional)
  `append_if_absent`.
- `algua/cli/data_cmd.py` — `data verify` subcommand.
- Tests under `tests/` (extend `test_data_store_publish.py`; new verify tests).

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
(`algua/data` stays I/O-bound but imports no research/contracts internals — boundary
unchanged).
