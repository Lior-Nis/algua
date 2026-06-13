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
**truncation / missing-file / missing-dir-entry / torn-or-unreadable payload** failure
modes — exactly what an unclean power loss produces (page-cache contents and dirty
directory entries that were never written back, in arbitrary writeback order). We do
**not** defend against silent mid-page bit-flips on already-durable bytes (media rot);
those are out of the power-loss threat model. Note writeback order is **not** guaranteed:
a parquet file's footer can become durable while interior data pages are lost, so a
footer-only check is insufficient (see Part 2 — verify does a full read-back).

`data_dir` (the configured store root) is assumed to pre-exist and be durable; it is the
fsync boundary — durability fsyncs walk *up to and including* `data_dir` but no further.

### Urgency

Low: single-host research workloads, all data re-ingestable from vendors. Becomes
load-bearing if the data dir ever holds non-reproducible data (the `ingest_file`
registration path is the one such case today).

## Part 1 — Write-side fsync discipline

### New primitives (`algua/data/files.py`)

```python
def fsync_file(path: Path) -> None:
    """fsync a file's data to stable storage. Linux-only: open(O_RDONLY|O_CLOEXEC),
    fsync, close — a read-only fd still flushes the inode's dirty data pages."""

def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation entry within it is durable. Linux-only:
    open(O_RDONLY|O_DIRECTORY|O_CLOEXEC) — O_DIRECTORY makes a non-dir path fail loudly
    rather than silently fsyncing the wrong object."""

def fsync_parents(path: Path, *, stop_at: Path) -> None:
    """fsync every directory from `path.parent` up to and including `stop_at`. Covers
    ancestor directories newly created by `mkdir(parents=True)` so that a freshly-created
    dir's own name is durable in *its* parent, not just the leaf entry. `stop_at` is the
    durable store root (`data_dir`); `path` must be at or under it."""

def fsync_tree(root: Path) -> None:
    """Bottom-up fsync of every regular file, then every subdir, then `root` itself
    (`os.walk(topdown=False)`), so child durability precedes parent. For partitioned
    trees whose part-files pyarrow wrote without exposing a handle we can fsync directly."""
```

All helpers `os.open(...)` → `os.fsync(fd)` → close (Linux semantics; the threat model is a
single local Linux FS, so macOS/NFS divergences are out of scope and noted in the
docstrings). They fail loudly with path context on error.

**Why `fsync_parents` (not just the leaf parent):** the publish sites `mkdir(parents=True)`
the target's ancestors (e.g. `snapshots/<dataset>/<snapshot_id>/`). Fsyncing only the leaf
`target.parent` makes the payload's entry durable *inside* that dir, but the dir's own name
— and every ancestor created in the same `mkdir` — is a dirty entry in *its* parent. After
power loss those ancestor entries can be lost, leaving the payload on durable blocks but
unreachable while the manifest names it. Walking up to `data_dir` closes that gap; where
the ancestors already existed the fsync is a cheap near-no-op.

### The invariant

At every publish site: **payload bytes durable → `os.replace` → parent-dir chain durable**,
all completed *before* the manifest append (which remains the single commit point). A power
cut between payload-durable and manifest-commit leaves an orphan payload not named by the
manifest — never a correctness problem. On a *same-id* re-ingest it is re-adopted (dirs) /
atomically replaced (files); a *different-id* orphan (or an `ingest_file` orphan left under
a different filename) is simply leaked disk space until a GC sweep (out of scope, as in
#158) — harmless to correctness, not auto-reclaimed.

### Publish-site changes

`write_bytes_snapshot` and `_commit_bars_dir` take the store root (`data_dir`) so they can
pass `stop_at=data_dir` to `fsync_parents`. (`write_bytes_snapshot` is already called with
`self.data_dir`; `_commit_bars_dir` is a method on the store.)

| Site | File | Change |
|---|---|---|
| `write_bytes_snapshot` (universe / fundamentals / news single-file) | `files.py` | `fh.flush()` + `os.fsync(fh.fileno())` on the temp fd before `os.replace`; `fsync_parents(target_path, stop_at=data_dir)` after |
| `_commit_bars_dir` — **publish branch** (partitioned bars, both `ingest_bars` and `ingest_bars_streamed`) | `store.py` | `fsync_tree(staging_dir)` before `os.replace`; `fsync_parents(target, stop_at=data_dir)` after the rename |
| `_commit_bars_dir` — **adoption branch** (target already exists, `ENOTEMPTY`) | `store.py` | After successful `validate_partitioned_bars_dir`, run the **same durability barrier before the manifest append**: `fsync_tree(target)` + `fsync_parents(target, stop_at=data_dir)`. A concurrent/prior writer may have `os.replace`'d the dir into place but not yet fsync'd it; the adopter is about to commit the manifest, so it must independently guarantee the payload + dir entries are durable. (Skipped only when `find(snapshot_id)` shows the record is already committed — then we just return it.) |
| `ingest_file` (`shutil.copy2` → `os.replace`) | `store.py` | `fsync_file(staged)` before `os.replace` (copy2 does not fsync); `fsync_parents(target, stop_at=data_dir)` after. The staging *source* entry is **not** fsync'd: it is transient, and the rename's durability is carried by fsyncing the target's parent chain. |
| `SnapshotManifest._repair` (repair-by-rename) | `manifest.py` | `fsync_dir(self.path.parent)` after `os.replace` (the temp is already fsync'd). The manifest lives directly under `data_dir`, so a single parent fsync — no ancestor chain — suffices. |
| `SnapshotManifest.append_if_absent` | `manifest.py` | Capture `manifest_existed = self.path.exists()` **immediately after acquiring the lock, before `_repair`/append**. After the append, `fsync_dir(self.path.parent)` **only when `not manifest_existed`** (first creation makes a new dir-entry; appends to an existing inode don't change the parent dir, and the content fsync already covers them). If `_repair` ran, it already fsync'd the parent for its own rename; the conditional append-fsync covers the no-repair first-create case. |

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
validation on every `read_bars` would touch every partition and defeat its symbol/time
pushdown). Because it is off the hot path, verify does a **full read-back** of the payload
(not a footer-only peek): writeback order is not guaranteed, so a footer can survive while
interior data pages are lost — only actually reading every row group/column proves the
bytes are durable and decompressible. `data verify` is the explicit backstop to run after
an unclean shutdown or on demand.

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
                  "ok": false, "error": "…"}]}
  ```
- **Fails closed:** non-zero exit if any snapshot fails. Before dispatch, verify checks the
  payload path exists and is the expected type (dir for `parquet_dataset`, file otherwise);
  a missing path or type mismatch is itself a failure (not a crash).

### Validation dispatch (full read-back, by `storage_format` — explicit + closed)

The dispatch is a **closed** match on `storage_format`; an unrecognized value fails with
`"unsupported verify format: <fmt>"` rather than silently assuming byte-hash semantics.

| `storage_format` | Check | Catches |
|---|---|---|
| `parquet_dataset` (partitioned bars) | Read the full dataset (every partition, all row groups) and confirm the **summed row count equals `rec.row_count`**. Reuses a read-back that decompresses every part-file's pages. The **symbol-set is NOT cross-checked** (see note) — a missing partition shows up as a short row count. | missing part-file, torn/unreadable data pages, truncated dir |
| `parquet` (single-file: universe / fundamentals / news, and `.parquet` via `ingest_file`) | Fully read the parquet file (all row groups/columns — raises on any unreadable page) and confirm `num_rows == rec.row_count`. **Power-loss / readability check only — NOT content-integrity:** it does not re-derive the logical or byte content hash, so a same-row-count tampered file would pass. That is outside the power-loss threat model and documented in code. | truncated / torn / unreadable single-file parquet |
| anything else (`ingest_file` csv / generic `file`) | `sha256_file(path) == rec.content_hash`. | any truncation / corruption of a registered non-parquet file |

**Why the byte-hash branch is safe (and its one constraint):** the `"parquet"` branch
recomputes *no* hash (read-back + `num_rows` only), so it is agnostic to whether a record's
`content_hash` is a logical hash (fundamentals / news) or a byte hash (universe). The only
branch that recomputes a hash is the `else` branch, and today the only ingest path that
produces a non-`parquet`/non-`parquet_dataset` `storage_format` is `ingest_file`, whose
`content_hash` is a byte hash (`sha256_file`) by construction. So `sha256_file == content_hash`
is always correct on this branch *today*. The implementation documents this as a load-bearing
invariant: **any future ingest path that introduces a new non-parquet `storage_format` with a
logical (non-byte) hash MUST extend this dispatch** — the byte-hash branch is not a safe
default for a logical-hash format. (A `content_hash_kind` provenance column was considered
and declined: no such format exists, and it is a schema bump for a low-urgency feature.)

`sha256_file` already exists. The parquet read-back uses `pyarrow`/`pq` reads we already
depend on. **Implementation constraint (load-bearing):** the read-back must force every data
page — use `dataset.to_table(columns=None)` / `pq.read_table(path)` or full row-group/batch
iteration with all columns. It must **not** use `count_rows()`, footer `metadata.num_rows`,
or a zero/pruned-column read — those report row counts without decompressing data pages and
would miss exactly the data-page truncation this command exists to catch. The summed
row-count is then compared to `rec.row_count`. No new logical-hash recompute.

**Note on the dropped symbol-set check:** the original design cross-checked the partition
symbol set against `rec.metadata.symbols`. That field is the *caller-supplied request*
metadata, not an authoritative persisted partition set (`ingest_bars` does not force
caller-symbols == frame-symbols; `ingest_bars_streamed` records the requested list, not the
observed `seen_symbols`), so an exact-set check would **false-fail** healthy snapshots.
Truncation/missing-partition is already caught by the row-count check, so the symbol-set
equality is dropped from verify (it stays in `validate_partitioned_bars_dir`'s adoption use,
where `expected_symbols` is authoritative — derived from the just-ingested frame).

## Out of scope (deferred, with rationale)

- **Logical/byte content-hash recompute in verify for parquet** — verify reads the bytes
  back (proving durability/readability) but does not re-derive the logical hash, so it is
  not a tamper detector for parquet; silent media bit-flips on already-durable bytes are
  outside the power-loss threat model. (Generic `ingest_file` files *do* get a byte-hash
  comparison, since that is how their `content_hash` was computed.)
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
   real paths; `fsync_dir` on a non-directory raises (O_DIRECTORY); `fsync_tree` visits
   nested `symbol=<SYM>/part-*.parquet` (files fsync'd before their parent dir) — assert via
   a spy recording `(path, is_dir)` in walk order. `fsync_parents(path, stop_at=root)`
   fsyncs every dir from `path.parent` up to and including `root`, and **not** above `root`.
2. **`write_bytes_snapshot`** — spy on `os.fsync` + `os.replace`: the temp file is fsync'd
   *before* the replace and the parent chain (up to `data_dir`) *after*; no temp residue
   (existing test still green).
3. **`_commit_bars_dir` publish branch** — `fsync_tree(staging_dir)` completes before
   `os.replace`, and the parent chain after.
4. **`_commit_bars_dir` adoption branch** — when the target dir already exists and is
   adopted (no committed record yet), `fsync_tree(target)` + parent chain run *before* the
   manifest append (assert ordering via spy); when `find` already shows a committed record,
   it returns without re-fsync.
5. **`ingest_file`** — the staged copy is fsync'd before `os.replace`, parent chain after;
   the staging source dir is *not* fsync'd.
6. **manifest `_repair`** — parent dir fsync'd after the repair replace; committed records
   intact (extends the existing repair test).
7. **manifest `append_if_absent`** — parent dir fsync'd on first manifest creation; **not**
   fsync'd on a subsequent append to the existing file (assert the `manifest_existed`
   create-vs-append gating).

### Read-side (`data verify` — real behavioral tests)

8. Healthy partitioned-bars snapshot → `ok: true`, exit 0.
9. Corrupted part-file body (overwrite interior bytes / truncate the data, footer-independent)
   → the full read-back raises → `ok: false`, non-zero exit, error names the snapshot.
10. Missing part-file (delete one) → summed row-count mismatch → fails closed.
11. Healthy single-file parquet (fundamentals / news / universe) → ok.
12. Truncated single-file parquet (cut the tail) → full read raises → fails closed.
13. `ingest_file` generic file (csv) tampered after ingest → `sha256_file` mismatch → fails.
14. `verify` with no `--snapshot-id` over a mix of healthy + one damaged snapshot →
    aggregate counts correct, exit code reflects the failure.
15. `verify --snapshot-id` on an unknown id → `SnapshotNotFound` surfaced as an error
    (non-zero exit), not a crash; a payload path that is missing or the wrong type (dir vs
    file) → fails closed with a clear error, not a traceback.
16. A record with an unknown `storage_format` → `"unsupported verify format"` failure (the
    closed dispatch), not a silent byte-hash assumption.

## Files touched

- `algua/data/files.py` — `fsync_file`, `fsync_dir`, `fsync_parents`, `fsync_tree`; fsyncs
  inside `write_bytes_snapshot` (now takes `data_dir` for `stop_at`); a full-read-back
  parquet verifier helper.
- `algua/data/store.py` — `_commit_bars_dir` (publish + adoption durability barriers),
  `ingest_file` fsyncs; `verify_snapshot(s)` read-side method with the closed
  `storage_format` dispatch.
- `algua/data/manifest.py` — `manifest_existed` capture + parent-dir fsyncs in `_repair`
  and (conditional) `append_if_absent`.
- `algua/cli/data_cmd.py` — `data verify` subcommand.
- Tests under `tests/` (extend `test_data_store_publish.py`; new verify tests).

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
(`algua/data` stays I/O-bound but imports no research/contracts internals — boundary
unchanged).
