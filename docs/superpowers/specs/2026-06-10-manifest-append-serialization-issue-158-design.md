# Manifest append serialization + staged payload publish (#158)

**Status:** design (GATE-1 approved, 3 review rounds) · **Date:** 2026-06-10 · **Issue:** #158

## Problem

`SnapshotManifest.append` is an unserialized jsonl append, and every `DataStore` commit path —
`ingest_file`, `ingest_bars`, `_ingest_parquet` (universes), `ingest_fundamentals`, and
`ingest_bars_streamed` (plus its adopt-on-target-exists branch) — does an **unlocked**
`find(snapshot_id) -> write payload -> append(rec)`. With concurrent agent sessions on one data
dir now the normal operating mode, three real hazards exist:

1. **Double-append:** two concurrent ingests computing the same content-addressed `snapshot_id`
   both pass the `find` check and both append — duplicate manifest records for one snapshot.
2. **Torn non-final lines:** `_read_all` tolerates a torn (crash-truncated) **final** line only.
   A writer crashing mid-line followed by another writer's clean append turns the torn line into
   a malformed **non-final** line — a hard read failure for every reader.
3. **Committed-snapshot corruption (GATE-1 CRITICAL):** all non-streamed paths write payloads
   **directly to final content-addressed paths** (`copy_snapshot`, `write_bytes_snapshot`,
   `write_partitioned_bars`). A same-id loser that passed the unlocked pre-check can rewrite an
   already-committed snapshot's bytes; if it crashes mid-write, the manifest points at corrupt
   data. A crashed direct-write `ingest_bars` also leaves a **partial final dir** that the
   streamed adopt branch would adopt blindly.

## Invariants

- Exactly one manifest record per `snapshot_id`, ever.
- The manifest always parses for readers; dropping an *uncommitted* final tail is the only
  allowed degradation.
- A **committed** record (newline-terminated, durably appended) is never lost or rewritten, and
  always names complete on-disk data — under the **process-crash** threat model (power-loss
  durability is explicitly deferred, see Deferred).
- Ingest stays idempotent: re-ingesting identical content returns the existing record.
- Writers are cooperative CPython processes on one host, **local Linux filesystem** (the flock
  contract; documented, not runtime-enforced).

## Design

### A. Manifest write protocol — `SnapshotManifest.append_if_absent(rec)`

The public `append` is **removed**; `append_if_absent` is the only write path:

1. **Lock:** open the sidecar `manifest.jsonl.lock` **fresh per call** (create if absent; the
   lock file is NEVER unlinked by anything — cleanup tooling must skip it), take a blocking
   `fcntl.flock(LOCK_EX)`. After acquiring, run a **staleness check**: `os.fstat(fd)` vs
   `os.stat(path)` on `(st_dev, st_ino)`; on mismatch close + reopen, bounded retries (5), then
   raise a distinct error stating the lock file was replaced externally (environmental
   corruption, not an ingest failure). A fresh fd per call is load-bearing: `flock` is
   per-open-file-description, so a cached/shared fd would silently self-grant.
2. **Single read under the lock:** split the manifest bytes into committed lines
   (newline-terminated) and an *uncommitted tail* (no trailing newline — dropped whether or not
   it parses). Blank committed lines are skipped (compat). Any non-blank committed line that
   fails to parse ⇒ raise (real corruption).
3. **Dedup:** if `rec.snapshot_id` is among the committed records, return that existing record.
   The winner's record is canonical — its `created_at`/`data_path` stand; the loser's `rec` is
   discarded.
4. **Repair (only if an uncommitted tail exists):** write the committed prefix to a
   `tempfile.mkstemp` temp in the manifest's directory, flush + `os.fsync(temp_fd)` + close,
   then `os.replace` onto `manifest.jsonl`. Never truncate in place: a lock-free reader mid-read
   on a shrinking inode could splice old+new bytes into a malformed non-final line; the rename
   keeps the old inode complete. Best-effort cleanup of stale repair temps (crash residue) under
   the lock; the temp is unlinked in `finally` if the replace did not happen.
5. **Append:** open a fresh append handle (after any repair), write `json + "\n"` in one write,
   flush + `os.fsync`, release the lock (close the lock fd). Return `rec`.

### B. Reader protocol — newline is the commit marker

`_read_all` changes: a final line **lacking a trailing newline is dropped even if it parses**
(today a parseable no-newline final line is treated as committed — a semantic change, noted in
the docstring; manifests are machine-written with trailing newlines, so no real legacy
exposure). Any newline-terminated non-blank line that fails to parse ⇒ raise. Readers stay
**lock-free**: appends only grow the file (a racing reader sees at worst a partial final line ⇒
dropped); repair swaps inodes atomically (a reader sees the complete old or complete new file).

### C. Staged no-overwrite payload publish (all five paths)

Payload writes move off final paths; correctness comes from the locked re-check inside
`append_if_absent`, while the existing cheap **unlocked** `find` pre-check stays (skip
recomputation for known ids). Every ingest path **returns `append_if_absent(rec)`**, never its
local `rec`.

- **Dir payloads** (`ingest_bars`, joining `ingest_bars_streamed`'s existing protocol): write
  the dataset into `snapshots/_staging/<uuid>`, `os.replace` onto the final dir; staging removed
  in `finally`. On `ENOTEMPTY`/`EEXIST` with `target.is_dir()`: **validate, then adopt**.
- **Adoption validation** (new; legacy direct-write `ingest_bars` may have left partial final
  dirs): metadata-only checks — every `part-*.parquet` footer parses, `sum(num_rows)` equals the
  expected row count, and the `symbol=<SYM>` partition set equals the expected symbols. Mismatch
  ⇒ raise, fail closed (no auto-delete). This is a *partial-corruption detector*, not a
  cryptographic revalidation — content-addressing (same id ⇒ same logical content) plus
  atomic-rename publishing carries the rest. The streamed path's inner `manifest.find` re-check
  collapses into `append_if_absent`.
- **File payloads** (`ingest_file`, `_ingest_parquet`, `ingest_fundamentals`): write to a temp,
  then `os.replace(tmp, target)` — atomic publish; replacing an existing committed file is
  benign (same id ⇒ identical bytes; readers see the old or new inode, byte-identical) and
  self-heals legacy torn files. For `_ingest_parquet`/`ingest_fundamentals` the temp lives in
  the target's parent dir; for `ingest_file` it lives under `snapshots/_staging/` because it
  must exist *before* the id (and hence the target path) is known. The temp is unlinked in
  `finally` if the replace did not happen. No adoption needed for files.
- **`ingest_file` staging-hash (TOCTOU fix):** copy the external source ONCE into the staging
  temp, then sha256/count **the staging copy**, derive `snapshot_id` from that, and publish that
  exact artifact via `os.replace` (same filesystem — staging lives inside the data dir). (Today
  the source is hashed and then re-read for the copy — a mutating source could commit bytes
  that don't match `content_hash`.)
- **Documented `ingest_file` quirk:** `snapshot_id` excludes the source filename but `data_path`
  includes it. Same-content-different-filename races resolve to the winner's canonical record;
  the loser's published file may remain as a benign orphan inside the same content-addressed
  snapshot dir. Reads always resolve via `record.data_path`.

### D. Documentation

Manifest docstring records the contract: local-Linux-FS-only locking; the lock file is never
deleted; the returned record may be the concurrent winner's (`created_at`/`data_path` are the
winner's); newline-as-commit-marker semantics.

## Deferred (out of scope, with rationale)

- **Power-loss fsync discipline** for payload files/dirs + parent-dir fsyncs: process-crash
  safety is complete (payload publish precedes manifest append; page cache and completed renames
  survive process death). Power-loss atomicity is orthogonal, pre-existing, and store-wide —
  follow-up issue.
- **NFS/remote-FS runtime detection** (`doctor` check): threat model is local FS; doc note only.
- **Orphan-payload gc:** content-addressed orphans are re-adopted (dirs) or atomically replaced
  (files) on the next same-id ingest; `clear_staging` already sweeps staging residue.
- **Legacy-manifest migration command** for the newline-commit-marker change: manifests are
  machine-written with trailing newlines; docstring note suffices.
- **Canonical `ingest_file` filenames:** original filenames keep inspectable provenance; the
  orphan case is rare, benign, and documented.
- **SHA-256 collision scenarios:** out of threat model.

## Tests

1. **Interleaved same-id double-ingest** (logic-level dedup, single process, monkeypatch between
   rename and append; *not* claimed as lock coverage): exactly one manifest record; both calls
   return that record.
2. **N-subprocess concurrent appenders:** `multiprocessing` workers, barrier start, overlapping
   snapshot_ids, worker exceptions propagated ⇒ exactly one record per id + the manifest parses
   cleanly.
3. **Deterministic lock-holder contention:** a holder process takes the flock and signals
   (multiprocessing.Event/Pipe, not sentinel files); an appender process attempts
   `append_if_absent`; the parent asserts the appender produced no result before release, and a
   result after ⇒ proves real cross-process serialization.
4. **Lock staleness:** replace the lock file while a holder owns it; the acquire helper detects
   the dev/ino mismatch, retries, and raises the distinct error when retries exhaust.
5. **Repair:** torn unparseable tail AND parseable-but-no-newline tail ⇒ both dropped + repaired;
   committed records intact; file ends clean. Stale repair-temp residue is cleaned up.
6. **Reader behavior:** file without trailing newline ⇒ final line dropped, no exception; a
   malformed newline-terminated line ⇒ raises.
7. **Loser-returns-winner:** the losing ingest returns the winner's record (winner's
   `created_at`/`data_path`).
8. **Adoption validation:** a deliberately partial legacy final dir (missing partition / torn
   part file / wrong row count) ⇒ adopt raises, nothing appended.
9. **`ingest_file` source mutation:** mutate the source after ingest starts ⇒ the committed
   snapshot's bytes match its `content_hash` (the staging copy is the hashed artifact).

## Review trail (GATE 1)

- Round 1 (Codex + Gemini Flash + GLM-5.1): accepted — staged no-overwrite publish (Codex
  CRITICAL, verified against `files.py`); newline-as-commit-marker (Codex HIGH); repair-by-rename
  instead of in-place truncate (GLM HIGH); lock-fd lifecycle + inode staleness check (GLM HIGH);
  single-pass read under lock; test hardening. Declined: NFS runtime check, payload fsync
  discipline (power-loss class), orphan gc, randomized-sleep stress tests.
- Round 2 (Codex + Gemini Flash): accepted — `ingest_file` staging-hash TOCTOU fix (Codex HIGH);
  adoption validation for legacy partial dirs (Codex HIGH); repair-temp mechanics; explicit
  staleness failure mode; return-canonical-record contract + tests. Declined: same-id/different-
  content "races" (impossible by construction — id embeds the content hash), parent-dir fsync
  (power-loss class), migration command.
- Round 3 (Codex, scoped): **APPROVED**; residual nits folded in as documentation.
