# Manifest Append Serialization + Staged Payload Publish (#158) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serialize all manifest appends behind a flock-guarded `append_if_absent` and move every ingest payload write onto a staged, no-overwrite publish, so concurrent same-id ingests can neither double-append manifest records nor corrupt committed snapshots.

**Architecture:** `SnapshotManifest` gains the only write path, `append_if_absent(rec)`: a per-call `fcntl.flock` on a never-deleted sidecar `manifest.jsonl.lock` (with a dev/ino staleness check), one read under the lock, dedup, atomic repair-by-rename of any uncommitted tail, fsync'd append. Readers adopt newline-as-commit-marker. The five `DataStore` ingest paths stop writing payloads to final paths: dirs stage under `snapshots/_staging/<uuid>` then `os.replace` (with cheap metadata validation before adopting a pre-existing target), files write to a temp then `os.replace`. `ingest_file` hashes its staging copy (TOCTOU fix). Every path returns `append_if_absent(rec)`.

**Spec:** `docs/superpowers/specs/2026-06-10-manifest-append-serialization-issue-158-design.md` (GATE-1 approved). Read it before starting.

**Tech Stack:** Python 3.12, `fcntl.flock`, `tempfile.mkstemp`, `os.replace`, pyarrow parquet metadata, `multiprocessing` (fork) tests, pytest.

**Quality gate between tasks:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File map

- Modify: `algua/data/manifest.py` — newline-commit-marker `_read_all`; `append_if_absent` + lock/repair machinery; `ManifestLockReplacedError`; public `append` removed (Task 4).
- Modify: `algua/data/files.py` — `write_bytes_snapshot` becomes atomic (temp + `os.replace`); new `validate_partitioned_bars_dir`; `copy_snapshot` deleted (Task 4).
- Modify: `algua/data/store.py` — all five ingest paths: staged publish + `return append_if_absent(rec)`; `ingest_file` staging-hash; `ingest_bars` staged dir publish + adopt validation; streamed adopt branch validates.
- Create: `tests/test_manifest_append_if_absent.py` — single-process unit tests (reader semantics, dedup, repair, lock staleness).
- Create: `tests/test_manifest_concurrency.py` — real multi-process tests (N appenders, deterministic lock-holder contention).
- Create: `tests/test_data_store_publish.py` — store-level staged-publish tests (source mutation, interleaved same-id ingest, adoption validation, loser-returns-winner).
- Modify: `tests/test_data_read_bars.py:104` — `.append(legacy)` → `.append_if_absent(legacy)` (Task 4).

---

### Task 1: Reader semantics — newline is the commit marker

**Files:**
- Modify: `algua/data/manifest.py` (`_read_all`)
- Test: `tests/test_manifest_append_if_absent.py` (new file)

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for SnapshotManifest commit semantics and append_if_absent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from algua.data.manifest import SnapshotManifest
from algua.data.models import SnapshotMetadata, SnapshotRecord


def make_record(snapshot_id: str, created_at: str = "2026-01-01T00:00:00+00:00") -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_id=snapshot_id,
        metadata=SnapshotMetadata(
            dataset="bars", provider="p", symbols=("AAA",), start="2026-01-01",
            end="2026-01-01", as_of="2026-01-02T00:00:00+00:00", source="s", kind="bars",
            timeframe="1d", adjustment="none",
        ),
        row_count=1, content_hash="h",
        data_path=Path(f"snapshots/bars/{snapshot_id}"),
        created_at=created_at, storage_format="parquet_dataset",
    )


def committed_line(rec: SnapshotRecord) -> str:
    return json.dumps(rec.to_dict(), sort_keys=True) + "\n"


def test_read_drops_parseable_final_line_without_newline(tmp_path):
    # Newline is the commit marker: a final line that PARSES but lacks "\n" is uncommitted.
    manifest_path = tmp_path / "manifest.jsonl"
    committed = make_record("aaa1")
    uncommitted = make_record("bbb2")
    manifest_path.write_text(
        committed_line(committed) + committed_line(uncommitted).rstrip("\n"),
        encoding="utf-8",
    )
    recs = SnapshotManifest(manifest_path).list_records()
    assert [r.snapshot_id for r in recs] == ["aaa1"]


def test_read_drops_torn_unparseable_final_line(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        committed_line(make_record("aaa1")) + '{"snapshot_id": "torn',
        encoding="utf-8",
    )
    recs = SnapshotManifest(manifest_path).list_records()
    assert [r.snapshot_id for r in recs] == ["aaa1"]


def test_read_raises_on_corrupt_committed_line(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("not-json\n" + committed_line(make_record("aaa1")), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        SnapshotManifest(manifest_path).list_records()


def test_read_skips_blank_committed_lines(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        committed_line(make_record("aaa1")) + "\n" + committed_line(make_record("bbb2")),
        encoding="utf-8",
    )
    recs = SnapshotManifest(manifest_path).list_records()
    assert [r.snapshot_id for r in recs] == ["aaa1", "bbb2"]
```

- [ ] **Step 2: Run tests to verify the new semantics test fails**

Run: `uv run pytest tests/test_manifest_append_if_absent.py -v`
Expected: `test_read_drops_parseable_final_line_without_newline` FAILS (today a parseable no-newline final line is treated as committed); the other three PASS (existing behavior).

- [ ] **Step 3: Rewrite `_read_all` with newline-as-commit-marker**

Replace the whole `_read_all` in `algua/data/manifest.py`:

```python
    def _read_all(self) -> list[SnapshotRecord]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        return self._parse_committed(self._committed_prefix(raw))

    @staticmethod
    def _committed_prefix(raw: str) -> str:
        """Newline is the commit marker: everything after the last "\\n" (a crash-torn or
        in-flight append) is uncommitted and dropped, EVEN IF it parses as JSON."""
        cut = raw.rfind("\n")
        return raw[: cut + 1] if cut >= 0 else ""

    @staticmethod
    def _parse_committed(committed: str) -> list[SnapshotRecord]:
        records: list[SnapshotRecord] = []
        for line in committed.splitlines():
            if not line.strip():
                continue
            # A committed (newline-terminated) line that fails to parse is real corruption.
            records.append(SnapshotRecord.from_dict(json.loads(line)))
        return records
```

(The old `is_last`/`ends_clean` tolerance logic is removed entirely.)

- [ ] **Step 4: Run the new tests + existing manifest tests**

Run: `uv run pytest tests/test_manifest_append_if_absent.py tests/test_data_store.py -v -k "manifest or read"`
Expected: all PASS (the existing `test_manifest_tolerates_torn_trailing_line` and `test_manifest_raises_on_corrupt_nonfinal_line` in `tests/test_data_store.py` still pass under the new semantics).

- [ ] **Step 5: Commit**

```bash
git add algua/data/manifest.py tests/test_manifest_append_if_absent.py
git commit -m "feat(data): manifest reader treats newline as the commit marker (#158)"
```

---

### Task 2: `SnapshotManifest.append_if_absent` — lock, dedup, repair-by-rename

**Files:**
- Modify: `algua/data/manifest.py`
- Test: `tests/test_manifest_append_if_absent.py`

Keep the existing public `append` for now (store.py still calls it); it is removed in Task 4.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_manifest_append_if_absent.py`)

```python
def test_append_if_absent_appends_and_returns_rec(tmp_path):
    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")
    rec = make_record("aaa1")
    out = manifest.append_if_absent(rec)
    assert out is rec
    assert [r.snapshot_id for r in manifest.list_records()] == ["aaa1"]


def test_append_if_absent_returns_existing_winner(tmp_path):
    # Loser-returns-winner: the FIRST committed record is canonical; the second call's
    # rec (different created_at) is discarded.
    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")
    winner = make_record("aaa1", created_at="2026-01-01T00:00:00+00:00")
    loser = make_record("aaa1", created_at="2026-01-02T00:00:00+00:00")
    assert manifest.append_if_absent(winner) is winner
    out = manifest.append_if_absent(loser)
    assert out.created_at == winner.created_at
    assert len(manifest.list_records()) == 1


def test_append_if_absent_repairs_torn_tail(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        committed_line(make_record("aaa1")) + '{"snapshot_id": "torn', encoding="utf-8"
    )
    manifest = SnapshotManifest(manifest_path)
    manifest.append_if_absent(make_record("bbb2"))
    raw = manifest_path.read_text(encoding="utf-8")
    assert "torn" not in raw
    assert raw.endswith("\n")
    assert [r.snapshot_id for r in manifest.list_records()] == ["aaa1", "bbb2"]


def test_append_if_absent_repairs_parseable_uncommitted_tail(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        committed_line(make_record("aaa1")) + committed_line(make_record("ccc3")).rstrip("\n"),
        encoding="utf-8",
    )
    manifest = SnapshotManifest(manifest_path)
    manifest.append_if_absent(make_record("bbb2"))
    recs = manifest.list_records()
    # the uncommitted ccc3 tail was dropped by repair, not resurrected
    assert [r.snapshot_id for r in recs] == ["aaa1", "bbb2"]


def test_append_if_absent_cleans_stale_repair_temps(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    stale = tmp_path / "manifest-repair-deadbeef.tmp"
    stale.write_text("crash residue", encoding="utf-8")
    SnapshotManifest(manifest_path).append_if_absent(make_record("aaa1"))
    assert not stale.exists()


def test_append_if_absent_creates_lock_file_and_never_deletes_it(tmp_path):
    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")
    manifest.append_if_absent(make_record("aaa1"))
    assert (tmp_path / "manifest.jsonl.lock").exists()
    manifest.append_if_absent(make_record("bbb2"))
    assert (tmp_path / "manifest.jsonl.lock").exists()


def test_acquire_raises_distinct_error_when_lock_replaced(tmp_path, monkeypatch):
    # Force a permanent dev/ino mismatch: os.stat on the lock path reports a different inode
    # than the held fd. The bounded retry loop must exhaust and raise the distinct error.
    import os as _os

    from algua.data.manifest import ManifestLockReplacedError

    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")
    real_stat = _os.stat

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if str(path) == str(tmp_path / "manifest.jsonl.lock"):
            fake = list(result)
            fake[1] = result.st_ino + 1  # st_ino is index 1
            return _os.stat_result(fake)
        return result

    monkeypatch.setattr(_os, "stat", fake_stat)
    with pytest.raises(ManifestLockReplacedError):
        manifest.append_if_absent(make_record("aaa1"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest_append_if_absent.py -v`
Expected: all the new tests FAIL with `AttributeError: ... no attribute 'append_if_absent'` / `ImportError: ... ManifestLockReplacedError`.

- [ ] **Step 3: Implement `append_if_absent`**

In `algua/data/manifest.py`, add imports `import fcntl`, `import os`, `import tempfile`, and:

```python
class ManifestLockReplacedError(RuntimeError):
    """The sidecar manifest lock file was replaced/unlinked externally.

    This is environmental corruption (something deleted `manifest.jsonl.lock` out from under
    live writers — the lock file must NEVER be removed), not an ingest failure."""


_LOCK_ACQUIRE_RETRIES = 5
_REPAIR_TEMP_PREFIX = "manifest-repair-"
```

Update the `SnapshotManifest` class docstring to record the contract:

```python
class SnapshotManifest:
    """Append-only jsonl manifest of snapshot records.

    Concurrency contract (#158): all writes go through `append_if_absent`, serialized by a
    blocking `fcntl.flock` on the sidecar `<manifest>.lock` file. flock semantics make this a
    LOCAL-LINUX-FILESYSTEM-ONLY contract (no NFS/remote mounts). The lock file is created on
    first use and must NEVER be deleted — unlinking it while a writer holds it silently breaks
    mutual exclusion (cleanup tooling must skip it). Readers are lock-free: a newline is the
    commit marker, so a racing reader sees at worst an uncommitted final tail, which it drops.
    `append_if_absent` may return the concurrent winner's record rather than the caller's —
    the returned record is canonical (its `created_at`/`data_path` stand)."""
```

Then the write path:

```python
    def append_if_absent(self, rec: SnapshotRecord) -> SnapshotRecord:
        """Append `rec` unless a record with its snapshot_id is already committed; return the
        committed record (the caller's `rec`, or the concurrent winner's). Repairs any
        uncommitted tail (crash residue) before appending. The ONLY manifest write path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = self._acquire_lock()
        try:
            raw = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            committed = self._committed_prefix(raw)
            for existing in self._parse_committed(committed):
                if existing.snapshot_id == rec.snapshot_id:
                    return existing
            self._clean_stale_repair_temps()
            if committed != raw:
                self._repair(committed)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return rec
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _acquire_lock(self) -> int:
        """Blocking LOCK_EX on the sidecar lock file via a FRESH fd per call (flock is
        per-open-file-description: a cached/shared fd would silently self-grant). After
        acquiring, verify the path still names the locked inode — a mismatch means something
        replaced the lock file externally; retry bounded, then fail distinctly."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        for _ in range(_LOCK_ACQUIRE_RETRIES):
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                fd_stat = os.fstat(fd)
                try:
                    path_stat = os.stat(lock_path)
                except FileNotFoundError:
                    path_stat = None
                if path_stat is not None and (
                    (path_stat.st_dev, path_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
                ):
                    return fd
            except BaseException:
                os.close(fd)
                raise
            os.close(fd)
        raise ManifestLockReplacedError(
            f"lock file {lock_path} was replaced externally while acquiring; it must never "
            "be deleted (see SnapshotManifest contract)"
        )

    def _repair(self, committed: str) -> None:
        """Replace the manifest with its committed prefix via temp + atomic rename. Never
        truncate in place: a lock-free reader mid-read on a shrinking inode could splice
        old+new bytes into a malformed non-final line; the rename keeps the old inode
        complete, so a reader sees the whole old or whole new file."""
        temp_fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=_REPAIR_TEMP_PREFIX, suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as fh:
                fh.write(committed)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def _clean_stale_repair_temps(self) -> None:
        """Best-effort sweep of repair temps left by a crashed writer (we hold the lock, so
        no live writer owns one)."""
        for stale in self.path.parent.glob(f"{_REPAIR_TEMP_PREFIX}*"):
            try:
                stale.unlink()
            except OSError:
                continue
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_manifest_append_if_absent.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/data/manifest.py tests/test_manifest_append_if_absent.py
git commit -m "feat(data): SnapshotManifest.append_if_absent — flock-serialized dedup append with repair-by-rename (#158)"
```

---

### Task 3: `files.py` — atomic byte publish + adoption validation helper

**Files:**
- Modify: `algua/data/files.py`
- Test: `tests/test_data_store_publish.py` (new file)

- [ ] **Step 1: Write the failing tests**

```python
"""Staged/atomic payload-publish tests (#158)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from algua.data.files import (
    validate_partitioned_bars_dir,
    write_bytes_snapshot,
    write_partitioned_bars,
)


def _bars_canon(symbols: list[str], n: int = 2) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for i in range(n):
            rows.append({
                "ts": pd.Timestamp(f"2024-07-0{i + 1}T00:00:00+00:00"), "symbol": sym,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "adj_close": 1.0, "volume": 10.0,
            })
    return pd.DataFrame(rows)


def test_write_bytes_snapshot_publishes_atomically_no_temp_residue(tmp_path):
    write_bytes_snapshot(b"payload", tmp_path, Path("snapshots/x/id1/file.bin"))
    target_dir = tmp_path / "snapshots" / "x" / "id1"
    assert (target_dir / "file.bin").read_bytes() == b"payload"
    assert [p.name for p in target_dir.iterdir()] == ["file.bin"]  # no temp left behind


def test_write_bytes_snapshot_replaces_existing_identical_file(tmp_path):
    rel = Path("snapshots/x/id1/file.bin")
    write_bytes_snapshot(b"payload", tmp_path, rel)
    write_bytes_snapshot(b"payload", tmp_path, rel)  # same id => identical bytes; benign
    assert (tmp_path / rel).read_bytes() == b"payload"


def test_validate_partitioned_bars_dir_accepts_complete_dataset(tmp_path):
    canon = _bars_canon(["AAA", "BBB"])
    write_partitioned_bars(canon, tmp_path / "ds")
    validate_partitioned_bars_dir(
        tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"AAA", "BBB"}
    )


def test_validate_partitioned_bars_dir_rejects_missing_partition(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    with pytest.raises(ValueError, match="adoption"):
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=4, expected_symbols={"AAA", "BBB"}
        )


def test_validate_partitioned_bars_dir_rejects_wrong_row_count(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    with pytest.raises(ValueError, match="adoption"):
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=len(canon) + 1, expected_symbols={"AAA"}
        )


def test_validate_partitioned_bars_dir_rejects_torn_part_file(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    part = next((tmp_path / "ds").rglob("part-*.parquet"))
    part.write_bytes(part.read_bytes()[: part.stat().st_size // 2])  # truncate the footer
    with pytest.raises(Exception):  # pyarrow raises its own invalid-file error
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"AAA"}
        )


def test_validate_partitioned_bars_dir_handles_dotted_symbols(tmp_path):
    canon = _bars_canon(["BRK.B"])
    write_partitioned_bars(canon, tmp_path / "ds")
    validate_partitioned_bars_dir(
        tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"BRK.B"}
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_store_publish.py -v`
Expected: FAIL — `validate_partitioned_bars_dir` does not exist; the atomic-residue assertion may also fail against the current naive `write_bytes`.

- [ ] **Step 3: Implement in `algua/data/files.py`**

Add imports `import os`, `import tempfile`, `from urllib.parse import unquote`. Replace `write_bytes_snapshot` and add the validator:

```python
def write_bytes_snapshot(data: bytes, data_dir: Path, relative_path: Path) -> None:
    """Atomically publish `data` at `data_dir/relative_path` via a same-dir temp +
    `os.replace` (#158): a reader never observes a partially written file, and a same-id
    concurrent re-publish is benign (content-addressed => identical bytes; readers see the
    old or new inode, byte-identical)."""
    target_path = data_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".publish-")
    try:
        with os.fdopen(temp_fd, "wb") as fh:
            fh.write(data)
        os.replace(temp_name, target_path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def validate_partitioned_bars_dir(
    target: Path, *, expected_row_count: int, expected_symbols: set[str]
) -> None:
    """Cheap metadata-only validation of a PRE-EXISTING bars dataset dir before adopting it
    as a committed snapshot (#158): every part file's parquet footer must parse, the summed
    metadata row counts must equal `expected_row_count`, and the hive `symbol=` partition set
    must equal `expected_symbols`. This is a partial-corruption detector for dirs left by the
    legacy direct-write ingest (not a cryptographic revalidation — content-addressing carries
    the rest). Mismatch => raise; the caller fails closed (never auto-deletes)."""
    total_rows = 0
    seen_symbols: set[str] = set()
    for part in target.rglob("part-*.parquet"):
        total_rows += pq.ParquetFile(part).metadata.num_rows  # raises on a torn footer
        head = part.relative_to(target).parts[0]
        if not head.startswith("symbol="):
            raise ValueError(
                f"adoption validation failed for {target}: unexpected layout entry {head!r}"
            )
        seen_symbols.add(unquote(head[len("symbol=") :]))
    if total_rows != expected_row_count or seen_symbols != expected_symbols:
        raise ValueError(
            f"adoption validation failed for {target}: rows {total_rows} (expected "
            f"{expected_row_count}), symbols {sorted(seen_symbols)} (expected "
            f"{sorted(expected_symbols)}); refusing to adopt a partial/foreign dir"
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_store_publish.py tests/test_data_store.py tests/test_data_fundamentals_store.py -v`
Expected: all PASS (existing callers of `write_bytes_snapshot` keep working).

- [ ] **Step 5: Commit**

```bash
git add algua/data/files.py tests/test_data_store_publish.py
git commit -m "feat(data): atomic write_bytes_snapshot + bars-dir adoption validation (#158)"
```

---

### Task 4: Route all five ingest paths through staged publish + `append_if_absent`

**Files:**
- Modify: `algua/data/store.py` (`ingest_file`, `ingest_bars`, `_ingest_parquet`, `ingest_fundamentals`, `ingest_bars_streamed`)
- Modify: `algua/data/manifest.py` (delete public `append`)
- Modify: `algua/data/files.py` (delete `copy_snapshot`)
- Modify: `tests/test_data_read_bars.py:104` (`append` → `append_if_absent`)
- Test: `tests/test_data_store_publish.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_data_store_publish.py`)

```python
import json

from algua.data.manifest import SnapshotManifest
from algua.data.store import DataStore


def _ingest_bars(store: DataStore, symbols: list[str] = ["AAA"]):
    return store.ingest_bars(
        provider="t", symbols=symbols, start="2024-07-01", end="2024-07-02",
        as_of="2024-07-03", source="unit", frame=_bars_canon(symbols),
        timeframe="1d", adjustment="none",
    )


def test_ingest_file_hashes_the_staging_copy_not_the_live_source(tmp_path, monkeypatch):
    # TOCTOU fix: mutate the source AFTER the staging copy is taken; the committed snapshot's
    # bytes must match its content_hash (i.e. the pre-mutation content).
    import shutil as _shutil

    source = tmp_path / "src.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    store = DataStore(tmp_path / "data")

    real_copy2 = _shutil.copy2

    def copy_then_mutate(src, dst, **kwargs):
        result = real_copy2(src, dst, **kwargs)
        source.write_text("a,b\n9,9\n", encoding="utf-8")  # source mutates post-copy
        return result

    monkeypatch.setattr(_shutil, "copy2", copy_then_mutate)
    rec = store.ingest_file(
        dataset="alt", provider="p", symbols=["AAA"], start="2024-07-01", end="2024-07-02",
        as_of="2024-07-03", source="unit", file_path=source,
    )
    from algua.data.files import sha256_file

    assert sha256_file(tmp_path / "data" / rec.data_path) == rec.content_hash


def test_ingest_bars_leaves_no_staging_residue(tmp_path):
    store = DataStore(tmp_path)
    _ingest_bars(store)
    staging = tmp_path / "snapshots" / "_staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_interleaved_same_id_ingest_yields_one_record_and_same_result(tmp_path, monkeypatch):
    # Logic-level dedup (NOT lock coverage): a complete second same-id ingest runs in the
    # window between the first ingest's payload publish and its manifest append.
    store = DataStore(tmp_path)
    manifest = store.manifest
    real_append = SnapshotManifest.append_if_absent
    state = {"interleaved": False, "inner": None}

    def interleaving_append(self, rec):
        if not state["interleaved"]:
            state["interleaved"] = True
            inner_store = DataStore(tmp_path)  # fresh store, same data dir
            state["inner"] = _ingest_bars(inner_store)
        return real_append(self, rec)

    monkeypatch.setattr(SnapshotManifest, "append_if_absent", interleaving_append)
    outer = _ingest_bars(store)
    assert len(manifest.list_records()) == 1
    assert outer.snapshot_id == state["inner"].snapshot_id
    # loser-returns-winner: the outer (losing) call returned the inner winner's record
    assert outer.created_at == state["inner"].created_at


def test_ingest_bars_adopting_partial_legacy_dir_fails_closed(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest_bars(store, symbols=["AAA", "BBB"])
    target = tmp_path / rec.data_path
    # simulate a partial legacy direct-write dir: strip a partition + the manifest record
    import shutil as _shutil

    _shutil.rmtree(target / "symbol=BBB")
    (tmp_path / "manifest.jsonl").unlink()
    with pytest.raises(ValueError, match="adoption"):
        _ingest_bars(DataStore(tmp_path), symbols=["AAA", "BBB"])
    assert SnapshotManifest(tmp_path / "manifest.jsonl").list_records() == []


def test_reingest_identical_bars_is_idempotent_and_adopts(tmp_path):
    store = DataStore(tmp_path)
    first = _ingest_bars(store)
    second = _ingest_bars(DataStore(tmp_path))
    assert second.snapshot_id == first.snapshot_id
    assert second.created_at == first.created_at  # winner's record is canonical
    assert len(store.manifest.list_records()) == 1


def test_public_append_is_gone():
    assert not hasattr(SnapshotManifest, "append")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_store_publish.py -v`
Expected: the new tests FAIL (staging residue from direct writes, `append` still exists, partial-dir adoption doesn't validate, TOCTOU unfixed).

- [ ] **Step 3: Rewrite the store commit paths**

In `algua/data/store.py`:

**(a) `ingest_file` — staging-hash + atomic publish.** Replace the body after the `is_file()` check (keep `_metadata(...)` as-is, it doesn't depend on the hash):

```python
        staging_dir = self.data_dir / "snapshots" / "_staging" / uuid.uuid4().hex
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Copy the external source ONCE, then hash/count THE STAGING COPY and publish that
            # exact artifact (#158): a source mutating mid-ingest can no longer commit bytes
            # that don't match content_hash.
            staged = staging_dir / source_path.name
            shutil.copy2(source_path, staged)
            content_hash = sha256_file(staged)
            row_count = count_tabular_rows(staged)
            snapshot_id = _snapshot_id(metadata, content_hash)

            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing

            relative_path = (
                Path("snapshots") / _path_part(metadata.dataset) / snapshot_id / source_path.name
            )
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)

            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format=source_path.suffix.lower().lstrip(".") or "file",
            )
            return self.manifest.append_if_absent(rec)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
```

Note (document in the method docstring): `snapshot_id` excludes the source filename but `data_path` includes it — a same-content-different-filename race resolves to the winner's canonical record; the loser's published file may remain as a benign orphan in the same content-addressed snapshot dir; reads always resolve via `record.data_path`. Delete `copy_snapshot` from `algua/data/files.py` and its import here.

**(b) `ingest_bars` — staged dir publish with validated adoption.** Replace the write/append tail (everything from `relative_path = ...`):

```python
        relative_path = Path("snapshots") / metadata.dataset / snapshot_id
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet_dataset",
        )
        staging_dir = self.data_dir / "snapshots" / "_staging" / uuid.uuid4().hex
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_partitioned_bars(canon.sort_values(["symbol", "ts"]), staging_dir)
            return self._commit_bars_dir(
                rec, staging_dir, expected_symbols={str(s) for s in canon["symbol"].unique()}
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
```

**(c) New shared commit helper** (the atomic publish + validated-adopt + manifest-commit tail, shared by `ingest_bars` and the streamed path; the caller owns staging creation and cleanup):

```python
    def _commit_bars_dir(
        self, rec: SnapshotRecord, staging_dir: Path, *, expected_symbols: set[str]
    ) -> SnapshotRecord:
        """Atomically publish a fully-written staging dir at `rec.data_path` and commit the
        manifest record (#158). On rename collision (target dir already exists): if the id is
        already committed, return that record; otherwise VALIDATE the existing dir (legacy
        direct-write ingest could have left a partial dir) and adopt it. Fails closed on
        validation mismatch — never deletes the suspect dir. The caller owns `staging_dir`
        creation and `finally`-cleanup."""
        target = self.data_dir / rec.data_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staging_dir, target)
        except OSError as exc:
            # Adopt ONLY the expected "target dir already exists and is non-empty" failure.
            # Re-raise anything else (permission, I/O, cross-device).
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not target.is_dir():
                raise
            found = self.manifest.find(rec.snapshot_id)
            if found is not None:
                return found
            validate_partitioned_bars_dir(
                target,
                expected_row_count=rec.row_count or 0,
                expected_symbols=expected_symbols,
            )
        return self.manifest.append_if_absent(rec)
```

(Add `validate_partitioned_bars_dir` to the `algua.data.files` import.)

**(d) `ingest_bars_streamed`** — the chunk loop already writes into its own staging dir; replace its commit tail (from `relative_path = ...` through `self.manifest.append(rec)`) so it builds `rec` then delegates; keep its existing `finally: shutil.rmtree(staging_dir, ignore_errors=True)`:

```python
            relative_path = Path("snapshots") / metadata.dataset / snapshot_id  # a DIR
            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet_dataset",
            )
            return self._commit_bars_dir(rec, staging_dir, expected_symbols=seen_symbols_set)
```

**(e) `_ingest_parquet` and `ingest_fundamentals`** — one-line tails: `self.manifest.append(rec); return rec` becomes `return self.manifest.append_if_absent(rec)` (their payload writes are already atomic via the Task-3 `write_bytes_snapshot`).

**(f) Delete `SnapshotManifest.append`** in `algua/data/manifest.py`, and update `tests/test_data_read_bars.py:104` to `SnapshotManifest(tmp_path / "manifest.jsonl").append_if_absent(legacy)`.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all PASS — including the pre-existing streamed-ingest adoption tests (`test_streamed_ingest_adopts_orphan_dataset` now passes through validation; the orphan dir is complete, so it validates).

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/data/store.py algua/data/manifest.py algua/data/files.py tests/test_data_store_publish.py tests/test_data_read_bars.py
git commit -m "feat(data): staged no-overwrite payload publish; all ingest paths commit via append_if_absent (#158)"
```

---

### Task 5: Real multi-process concurrency tests

**Files:**
- Create: `tests/test_manifest_concurrency.py`

Worker functions MUST be module-level (picklable). Use `multiprocessing.get_context("fork")` explicitly (flock state is per-OFD; workers open their own fds, and fork is the Linux default this suite assumes).

- [ ] **Step 1: Write the tests**

```python
"""Multi-process tests for SnapshotManifest.append_if_absent (#158).

Real OS processes: flock serialization is cross-process; in-process probes can't prove it."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

from algua.data.manifest import SnapshotManifest
from algua.data.models import SnapshotMetadata, SnapshotRecord

_CTX = multiprocessing.get_context("fork")


def _record(snapshot_id: str, worker: int) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_id=snapshot_id,
        metadata=SnapshotMetadata(
            dataset="bars", provider="p", symbols=("AAA",), start="2026-01-01",
            end="2026-01-01", as_of="2026-01-02T00:00:00+00:00", source="s", kind="bars",
            timeframe="1d", adjustment="none",
        ),
        row_count=1, content_hash="h",
        data_path=Path(f"snapshots/bars/{snapshot_id}"),
        created_at=f"2026-01-01T00:00:0{worker}+00:00", storage_format="parquet_dataset",
    )


def _appender(manifest_path: str, worker: int, ids: list[str], barrier, errors) -> None:
    try:
        barrier.wait(timeout=30)
        manifest = SnapshotManifest(Path(manifest_path))
        for snapshot_id in ids:
            manifest.append_if_absent(_record(snapshot_id, worker))
    except Exception as exc:  # propagate to the parent — a silent worker is a vacuous pass
        errors.put(f"worker {worker}: {exc!r}")


def test_concurrent_appenders_one_record_per_id(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    n_workers = 4
    shared_ids = [f"id{i:04d}" for i in range(25)]  # every worker appends EVERY id
    barrier = _CTX.Barrier(n_workers)
    errors = _CTX.Queue()
    workers = [
        _CTX.Process(target=_appender, args=(str(manifest_path), w, shared_ids, barrier, errors))
        for w in range(n_workers)
    ]
    for p in workers:
        p.start()
    for p in workers:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert errors.empty(), errors.get()
    # exactly one committed record per id, file parses cleanly, every line newline-terminated
    raw = manifest_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    ids = [json.loads(line)["snapshot_id"] for line in raw.splitlines() if line.strip()]
    assert sorted(ids) == sorted(shared_ids)
    recs = SnapshotManifest(manifest_path).list_records()
    assert len(recs) == len(shared_ids)


def _holder(lock_path: str, held, release) -> None:
    import fcntl
    import os

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    held.set()
    release.wait(timeout=30)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _blocked_appender(manifest_path: str, attempting, results) -> None:
    manifest = SnapshotManifest(Path(manifest_path))
    attempting.set()
    rec = manifest.append_if_absent(_record("contended", 0))
    results.put(rec.snapshot_id)


def test_appender_blocks_until_lock_holder_releases(tmp_path):
    # Deterministic contention: a holder process owns the flock; the appender must produce
    # NO result while it is held, and complete promptly once released.
    manifest_path = tmp_path / "manifest.jsonl"
    lock_path = str(manifest_path) + ".lock"
    held, release, attempting = _CTX.Event(), _CTX.Event(), _CTX.Event()
    results = _CTX.Queue()
    holder = _CTX.Process(target=_holder, args=(lock_path, held, release))
    holder.start()
    assert held.wait(timeout=10)
    appender = _CTX.Process(target=_blocked_appender, args=(str(manifest_path), attempting, results))
    appender.start()
    assert attempting.wait(timeout=10)
    appender.join(timeout=0.5)  # generous beat: appender must still be blocked on the flock
    assert appender.is_alive(), "appender completed while the lock was held — no serialization"
    assert results.empty()
    release.set()
    appender.join(timeout=10)
    assert appender.exitcode == 0
    holder.join(timeout=10)
    assert results.get(timeout=5) == "contended"
    assert [r.snapshot_id for r in SnapshotManifest(manifest_path).list_records()] == ["contended"]
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_manifest_concurrency.py -v`
Expected: both PASS, in seconds (no sleeps on the success path; the only timing assumption is that a blocked appender does not finish within 0.5s, which the held flock guarantees).

- [ ] **Step 3: Stress-check determinism**

Run: `for i in 1 2 3 4 5; do uv run pytest tests/test_manifest_concurrency.py -q || break; done`
Expected: 5/5 green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_manifest_concurrency.py
git commit -m "test(data): multi-process manifest append serialization + contention proofs (#158)"
```

---

### Task 6: Full gate + docs sweep

- [ ] **Step 1: Verify no direct-append path survives**

Run: `grep -rn "manifest.append(" algua tests --include="*.py"`
Expected: no matches (everything goes through `append_if_absent`).

- [ ] **Step 2: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 3: Commit the spec + plan docs** (if not yet committed on the branch)

```bash
git add docs/superpowers/specs/2026-06-10-manifest-append-serialization-issue-158-design.md docs/superpowers/plans/2026-06-11-manifest-append-serialization-issue-158.md
git commit -m "docs(spec): manifest append serialization design + plan (#158)"
```
