# Power-Loss Durability (issue #184) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make snapshot payload bytes, directory trees, and parent-directory rename entries durable on a power cut (not just a process crash), and add an explicit `algua data verify` command that reads payloads back to detect power-loss damage.

**Architecture:** Add four fsync primitives to `algua/data/files.py` and call them at every publish site so the durability barrier (`payload bytes durable → os.replace → parent-dir chain durable`) completes *before* the manifest append (the single commit point). Add a `verify_snapshot`/`verify_snapshots` read-back path on `DataStore` and a `data verify` CLI command that fails closed (non-zero exit) on any damaged snapshot.

**Tech Stack:** Python 3.12, `os.fsync`/`os.open`, pyarrow/pyarrow.parquet, pandas, Typer CLI, pytest. Single local Linux filesystem (the threat model; fsync helpers are Linux-specific by design).

**Spec:** `docs/superpowers/specs/2026-06-13-power-loss-durability-issue-184-design.md`

**Quality gate (run between tasks):**
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- **Modify `algua/data/files.py`** — add `fsync_file`, `fsync_dir`, `fsync_parents`, `fsync_tree` (durability primitives); add `parquet_file_row_count`, `parquet_dataset_row_count` (full read-back verifiers); add the temp-fsync + parent-chain fsync inside `write_bytes_snapshot`.
- **Modify `algua/data/store.py`** — fsyncs in `ingest_file` and `_commit_bars_dir` (publish + adoption barriers); new `verify_snapshot` / `verify_snapshots` methods.
- **Modify `algua/data/manifest.py`** — `manifest_existed` capture + parent-dir fsync in `append_if_absent`; parent-dir fsync in `_repair`.
- **Modify `algua/cli/data_cmd.py`** — new `data verify` command.
- **Tests:** new `tests/test_data_durability.py` (fsync helpers + placement/ordering spies + verify behavior); extend `tests/test_data_store_publish.py` only if a touched test needs it.

All durability helpers live in `files.py` (the data-layer I/O module); dispatch/orchestration lives in `store.py`; CLI wiring in `data_cmd.py`. `manifest.py` importing `fsync_dir` from `files.py` introduces no cycle (`files.py` imports only `algua.data.schema`).

---

## Task 1: fsync primitives in `files.py`

**Files:**
- Modify: `algua/data/files.py` (add after `sha256_bytes`, near the top imports)
- Test: `tests/test_data_durability.py` (new)

- [ ] **Step 1: Write the failing tests for the helpers**

Create `tests/test_data_durability.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from algua.data import files


def test_fsync_file_and_dir_run_on_real_paths(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    files.fsync_file(f)  # must not raise
    files.fsync_dir(tmp_path)  # must not raise


def test_fsync_dir_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    with pytest.raises(OSError):  # O_DIRECTORY on a file -> ENOTDIR
        files.fsync_dir(f)


def test_fsync_tree_visits_files_before_their_parent_dirs(tmp_path: Path, monkeypatch) -> None:
    # symbol=A/part-0.parquet + symbol=B/part-0.parquet
    for sym in ("A", "B"):
        d = tmp_path / f"symbol={sym}"
        d.mkdir()
        (d / "part-0.parquet").write_bytes(b"data")

    order: list[tuple[str, bool]] = []
    real_open = os.open

    def spy_open(path, flags, *a, **k):
        fd = real_open(path, flags, *a, **k)
        order.append((str(path), bool(flags & os.O_DIRECTORY)))
        return fd

    monkeypatch.setattr(os, "open", spy_open)
    files.fsync_tree(tmp_path)

    # every file fsync precedes the fsync of the dir that contains it
    for i, (path, is_dir) in enumerate(order):
        if is_dir:
            children = [p for p, d in order[:i] if Path(p).parent == Path(path) and not d]
            file_children = [
                p for p in (tmp_path / Path(path).name).rglob("*") if p.is_file()
            ] if Path(path) != tmp_path else []
            # all file entries directly under `path` already appeared
            for fc in file_children:
                assert any(p == str(fc) for p, d in order[:i] and order)  # appeared earlier
    # root itself is fsynced last
    assert order[-1] == (str(tmp_path), True)


def test_fsync_parents_walks_up_to_stop_at_inclusive(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    leaf = root / "snapshots" / "bars" / "abc123"
    leaf.mkdir(parents=True)
    payload = leaf / "symbol=A"
    payload.mkdir()

    fsynced: list[str] = []
    real_open = os.open

    def spy_open(path, flags, *a, **k):
        if flags & os.O_DIRECTORY:
            fsynced.append(str(path))
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", spy_open)
    files.fsync_parents(payload, stop_at=root)

    # fsyncs payload.parent (leaf), bars, snapshots, root — and nothing above root
    assert str(leaf) in fsynced
    assert str(root / "snapshots" / "bars") in fsynced
    assert str(root / "snapshots") in fsynced
    assert str(root) in fsynced
    assert str(root.parent) not in fsynced
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data_durability.py -v`
Expected: FAIL with `AttributeError: module 'algua.data.files' has no attribute 'fsync_file'`.

- [ ] **Step 3: Implement the four helpers in `files.py`**

In `algua/data/files.py`, insert after `sha256_bytes` (line 30):

```python
def fsync_file(path: Path) -> None:
    """fsync a regular file's data to stable storage. Linux-only: a read-only fd still
    flushes the inode's dirty data pages. (Threat model is a single local Linux FS;
    macOS/NFS fsync semantics differ and are out of scope.)"""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation entry within it becomes durable. O_DIRECTORY
    makes a non-directory path fail loudly (ENOTDIR) instead of silently fsyncing the wrong
    object. Linux-only (see `fsync_file`)."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_parents(path: Path, *, stop_at: Path) -> None:
    """fsync every directory from `path.parent` up to and including `stop_at` (the durable
    store root). Covers ancestor directories newly created by `mkdir(parents=True)`: fsyncing
    only the leaf parent leaves a freshly-created intermediate dir's own name un-durable in
    *its* parent. `path` must be at or under `stop_at`."""
    stop_at = stop_at.resolve()
    current = path.resolve().parent
    while True:
        fsync_dir(current)
        if current == stop_at or current == current.parent:
            break
        current = current.parent


def fsync_tree(root: Path) -> None:
    """Bottom-up fsync of every regular file, then every subdirectory, then `root` itself
    (`os.walk(topdown=False)`), so child durability precedes the parent's. For partitioned
    trees whose part-files pyarrow wrote without exposing a handle, we reopen+fsync each."""
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        for name in filenames:
            fsync_file(d / name)
        fsync_dir(d)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data_durability.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the quality gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/files.py tests/test_data_durability.py
git commit -m "feat(184): fsync primitives (file/dir/parents/tree) for power-loss durability"
```

---

## Task 2: fsync inside `write_bytes_snapshot`

**Files:**
- Modify: `algua/data/files.py:59-75` (`write_bytes_snapshot`)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_durability.py`:

```python
def test_write_bytes_snapshot_fsyncs_temp_before_replace_then_parents(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)

    rel = Path("snapshots") / "universes" / "snap1" / "universe.parquet"
    files.write_bytes_snapshot(b"payload-bytes", tmp_path, rel)

    assert (tmp_path / rel).read_bytes() == b"payload-bytes"
    # the temp file is fsynced BEFORE the rename; parent-chain dirs are fsynced AFTER
    assert events[0] == "fsync"  # temp file
    assert "replace" in events
    assert events.index("fsync") < events.index("replace")
    assert events.count("fsync") >= 2  # temp + at least one parent dir
    assert events[-1] == "fsync"  # last parent-chain fsync after the replace
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_data_durability.py::test_write_bytes_snapshot_fsyncs_temp_before_replace_then_parents -v`
Expected: FAIL — currently no fsync happens, so `events[0]` is `"replace"`.

- [ ] **Step 3: Implement the fsyncs**

Replace `write_bytes_snapshot` body (`algua/data/files.py:59-75`) with:

```python
def write_bytes_snapshot(data: bytes, data_dir: Path, relative_path: Path) -> None:
    """Atomically publish `data` at `data_dir/relative_path` via a same-dir temp +
    `os.replace` (#158): a reader never observes a partially written file, and a same-id
    concurrent re-publish is benign (content-addressed => identical bytes; readers see the
    old or new inode, byte-identical). Power-loss durable (#184): the temp's bytes are
    fsynced before the rename, and the target's parent-dir chain (up to `data_dir`) after."""
    target_path = data_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".publish-")
    try:
        with os.fdopen(temp_fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, target_path)
        fsync_parents(target_path, stop_at=data_dir)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_data_durability.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/files.py tests/test_data_durability.py
git commit -m "feat(184): fsync temp + parent chain in write_bytes_snapshot"
```

---

## Task 3: fsync in `ingest_file`

**Files:**
- Modify: `algua/data/store.py:128-129` (the `os.replace` in `ingest_file`)
- Modify: `algua/data/store.py:17-29` (import the new helpers from `files`)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_durability.py`:

```python
import pandas as pd

from algua.data.store import DataStore


def test_ingest_file_fsyncs_staged_before_replace_then_parents(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(src, index=False)
    store = DataStore(tmp_path / "store")

    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda s, d: (events.append("replace"), real_replace(s, d))[1]
    )

    store.ingest_file(
        source_path=src, dataset="custom", provider="local", symbols=["AAPL"],
        start="2026-01-02", end="2026-01-02", as_of="2026-01-03T00:00:00+00:00",
        source="fixture", kind="custom",
    )
    # a payload fsync precedes its replace; parent-chain fsyncs follow it
    assert "fsync" in events and "replace" in events
    first_replace = events.index("replace")
    assert events.index("fsync") < first_replace
    assert events[first_replace + 1 :].count("fsync") >= 1
```

(Confirm `ingest_file`'s signature against `algua/data/store.py` — match the keyword args it actually accepts; adjust the call above if needed.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_data_durability.py::test_ingest_file_fsyncs_staged_before_replace_then_parents -v`
Expected: FAIL — no payload fsync before the replace today (the only fsync is the manifest append, which comes after).

- [ ] **Step 3: Implement**

In `algua/data/store.py`, extend the `files` import block (lines 17-29) to include the new helpers:

```python
from algua.data.files import (
    BARS_STREAMED_HASH_ALGO,
    compose_bars_symbol_hash,
    count_tabular_rows,
    frame_to_parquet_bytes,
    fsync_file,
    fsync_parents,
    fsync_tree,
    logical_bars_hash,
    parquet_dataset_row_count,
    parquet_file_row_count,
    read_partitioned_bars,
    sha256_bytes,
    sha256_file,
    validate_partitioned_bars_dir,
    write_bytes_snapshot,
    write_partitioned_bars,
)
```

(`parquet_dataset_row_count` / `parquet_file_row_count` are added in Task 6; importing them now is fine only after Task 6 lands. To keep each task independently green, add `fsync_file`, `fsync_parents`, `fsync_tree` in this task and add the two `parquet_*_row_count` names in Task 6's import edit.)

For THIS task, add only:

```python
    fsync_file,
    fsync_parents,
    fsync_tree,
```

Then replace `algua/data/store.py:128-129`:

```python
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
```

with:

```python
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            fsync_file(staged)  # copy2 does not fsync; make the bytes durable before publish
            os.replace(staged, target)
            fsync_parents(target, stop_at=self.data_dir)  # rename entry + new ancestors durable
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_data_durability.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/store.py tests/test_data_durability.py
git commit -m "feat(184): fsync staged file + parent chain in ingest_file"
```

---

## Task 4: durability barriers in `_commit_bars_dir` (publish + adoption)

**Files:**
- Modify: `algua/data/store.py:199-225` (`_commit_bars_dir`)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_durability.py`:

```python
def _bars_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True)
    return pd.DataFrame(
        {
            "timestamp": list(idx) + list(idx),
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "open": [1.0, 1.1, 2.0, 2.1], "high": [1.0, 1.1, 2.0, 2.1],
            "low": [1.0, 1.1, 2.0, 2.1], "close": [1.0, 1.1, 2.0, 2.1],
            "adj_close": [1.0, 1.1, 2.0, 2.1], "volume": [10.0, 11.0, 20.0, 21.0],
        }
    )


def _ingest_bars(store: DataStore):
    return store.ingest_bars(
        provider="fixture", symbols=["AAPL", "MSFT"], start="2026-01-02", end="2026-01-05",
        as_of="2026-01-06T00:00:00+00:00", source="fixture", frame=_bars_frame(),
    )


def test_commit_bars_publish_fsyncs_tree_before_replace_then_parents(
    tmp_path: Path, monkeypatch
) -> None:
    store = DataStore(tmp_path / "store")
    events: list[str] = []
    import algua.data.store as store_mod
    real_replace = os.replace
    real_tree = store_mod.fsync_tree
    real_parents = store_mod.fsync_parents
    monkeypatch.setattr(store_mod, "fsync_tree", lambda p: (events.append("tree"), real_tree(p))[1])
    monkeypatch.setattr(os, "replace", lambda s, d: (events.append("replace"), real_replace(s, d))[1])
    monkeypatch.setattr(store_mod, "fsync_parents", lambda p, *, stop_at: (events.append("parents"), real_parents(p, stop_at=stop_at))[1])

    _ingest_bars(store)
    assert events[:3] == ["tree", "replace", "parents"]


def test_commit_bars_adoption_fsyncs_before_manifest_append(tmp_path: Path, monkeypatch) -> None:
    # First ingest publishes the dir. Delete the manifest record so a second ingest of the
    # SAME id re-enters _commit_bars_dir, hits ENOTEMPTY (target exists), and ADOPTS it.
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    manifest_path = (tmp_path / "store" / "manifest.jsonl")
    manifest_path.write_text("")  # drop the committed record, keep the payload dir

    import algua.data.store as store_mod
    events: list[str] = []
    real_tree = store_mod.fsync_tree
    monkeypatch.setattr(store_mod, "fsync_tree", lambda p: (events.append(f"tree:{p.name}"), real_tree(p))[1])
    orig_append = store.manifest.append_if_absent
    monkeypatch.setattr(store.manifest, "append_if_absent", lambda r: (events.append("append"), orig_append(r))[1])

    again = _ingest_bars(store)
    assert again.snapshot_id == rec.snapshot_id
    # the adoption barrier fsync_tree(target) ran BEFORE the manifest append
    assert any(e.startswith("tree:") for e in events)
    assert events.index(next(e for e in events if e.startswith("tree:"))) < events.index("append")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_durability.py -k commit_bars -v`
Expected: FAIL — `_commit_bars_dir` does not yet call `fsync_tree`/`fsync_parents`.

- [ ] **Step 3: Implement**

Replace `_commit_bars_dir` (`algua/data/store.py:199-225`) with:

```python
    def _commit_bars_dir(
        self, rec: SnapshotRecord, staging_dir: Path, *, expected_symbols: set[str]
    ) -> SnapshotRecord:
        """Atomically publish a fully-written staging dir at `rec.data_path` and commit the
        manifest record (#158). On rename collision (target dir already exists): if the id is
        already committed, return that record; otherwise VALIDATE the existing dir (legacy
        direct-write ingest could have left a partial dir) and adopt it. Fails closed on
        validation mismatch — never deletes the suspect dir. The caller owns `staging_dir`
        creation and `finally`-cleanup.

        Power-loss durable (#184): on the publish branch the staging tree is fsynced before
        the rename and the target's parent chain after; on the adoption branch the same
        barrier (tree + parent chain) runs before the manifest append, since a concurrent or
        prior writer may have renamed the dir into place without fsyncing it and we are about
        to commit it."""
        target = self.data_dir / rec.data_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fsync_tree(staging_dir)  # all part-files + dir entries durable before publish
            os.replace(staging_dir, target)
            fsync_parents(target, stop_at=self.data_dir)
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
            # Independent durability barrier: the adopter is about to commit the manifest.
            fsync_tree(target)
            fsync_parents(target, stop_at=self.data_dir)
        return self.manifest.append_if_absent(rec)
```

Note: the `fsync_tree(staging_dir)` + `fsync_parents` now sit *inside* the `try`, so if the `os.replace` raises `ENOTEMPTY` the `fsync_parents(target, ...)` line is skipped (the `os.replace` raised first) and control moves to the `except` adoption branch — correct.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_durability.py -k commit_bars -v`
Expected: PASS. Then run the streamed-ingest tests to confirm no regression: `uv run pytest tests/test_data_ingest_streamed.py -v`.

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/store.py tests/test_data_durability.py
git commit -m "feat(184): durability barriers in _commit_bars_dir (publish + adoption)"
```

---

## Task 5: parent-dir fsync in `manifest._repair` and `append_if_absent`

**Files:**
- Modify: `algua/data/manifest.py:1-9` (import `fsync_dir`)
- Modify: `algua/data/manifest.py:50-72` (`append_if_absent`)
- Modify: `algua/data/manifest.py:102-120` (`_repair`)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_durability.py`:

```python
from algua.data.manifest import SnapshotManifest
from algua.data.models import SnapshotMetadata, SnapshotRecord


def _rec(snapshot_id: str) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_id=snapshot_id,
        metadata=SnapshotMetadata(
            dataset="bars", provider="p", symbols=["AAPL"], start="2026-01-02",
            end="2026-01-02", as_of="2026-01-03T00:00:00+00:00", source="s", kind="bars",
        ),
        row_count=1, content_hash="h",
        data_path=Path("snapshots/bars") / snapshot_id, created_at="2026-01-03T00:00:00+00:00",
        storage_format="parquet_dataset",
    )


def test_append_fsyncs_parent_only_on_first_creation(tmp_path: Path, monkeypatch) -> None:
    import algua.data.manifest as man_mod
    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")

    dir_fsyncs: list[str] = []
    real = man_mod.fsync_dir
    monkeypatch.setattr(man_mod, "fsync_dir", lambda p: (dir_fsyncs.append(str(p)), real(p))[1])

    manifest.append_if_absent(_rec("aaaaaaaaaaaaaaaa"))  # first creation
    assert str(tmp_path) in dir_fsyncs  # parent dir fsynced on create

    dir_fsyncs.clear()
    manifest.append_if_absent(_rec("bbbbbbbbbbbbbbbb"))  # append to existing file
    assert str(tmp_path) not in dir_fsyncs  # NOT fsynced on a plain append


def test_repair_fsyncs_parent_after_rename(tmp_path: Path, monkeypatch) -> None:
    import algua.data.manifest as man_mod
    path = tmp_path / "manifest.jsonl"
    # one committed record (newline-terminated) + an uncommitted torn tail (no newline)
    good = '{"x": 1}\n'  # not parsed by _repair; _repair only rewrites the committed prefix
    manifest = SnapshotManifest(path)
    path.write_text(good + "uncommitted-no-newline")

    dir_fsyncs: list[str] = []
    real = man_mod.fsync_dir
    monkeypatch.setattr(man_mod, "fsync_dir", lambda p: (dir_fsyncs.append(str(p)), real(p))[1])

    manifest._repair(good.encode("utf-8"))
    assert path.read_text() == good
    assert str(tmp_path) in dir_fsyncs
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_durability.py -k "append_fsyncs or repair_fsyncs" -v`
Expected: FAIL with `AttributeError: module 'algua.data.manifest' has no attribute 'fsync_dir'`.

- [ ] **Step 3: Implement**

In `algua/data/manifest.py`, add the import (after line 7, `from pathlib import Path`):

```python
from algua.data.files import fsync_dir
from algua.data.models import SnapshotRecord
```

Replace `append_if_absent` (`algua/data/manifest.py:50-72`):

```python
    def append_if_absent(self, rec: SnapshotRecord) -> SnapshotRecord:
        """Append `rec` unless a record with its snapshot_id is already committed; return the
        committed record (the caller's `rec`, or the concurrent winner's). Repairs any
        uncommitted tail (crash residue) before appending. The ONLY manifest write path.

        Power-loss durable (#184): the appended bytes are fsynced (existing), and on the FIRST
        creation of the manifest file its new parent-directory entry is fsynced too. Plain
        appends to an existing inode do not change the parent dir, so they skip the dir fsync.
        A `_repair` rewrite makes its own parent fsync (see `_repair`)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = self._acquire_lock()
        try:
            manifest_existed = self.path.exists()
            raw = self.path.read_bytes() if manifest_existed else b""
            committed = self._committed_prefix(raw)
            for existing in self._parse_committed(committed.decode("utf-8")):
                if existing.snapshot_id == rec.snapshot_id:
                    return existing
            self._clean_stale_repair_temps()
            if committed != raw:
                self._repair(committed)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if not manifest_existed:
                fsync_dir(self.path.parent)
            return rec
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
```

Replace `_repair` (`algua/data/manifest.py:102-120`) — add the parent fsync after the rename:

```python
    def _repair(self, committed: bytes) -> None:
        """Replace the manifest with its committed prefix via temp + atomic rename. Never
        truncate in place: a lock-free reader mid-read on a shrinking inode could splice
        old+new bytes into a malformed non-final line; the rename keeps the old inode
        complete, so a reader sees the whole old or whole new file. Power-loss durable (#184):
        the temp is fsynced (existing) and the parent dir is fsynced after the rename so the
        replaced dir entry is durable."""
        temp_fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f"{self.path.name}{_REPAIR_TEMP_SUFFIX}", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "wb") as fh:
                fh.write(committed)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_name, self.path)
            fsync_dir(self.path.parent)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_durability.py -k "append_fsyncs or repair_fsyncs" -v`
Expected: PASS. Then `uv run pytest tests/test_manifest_concurrency.py -v` to confirm no regression.

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/manifest.py tests/test_data_durability.py
git commit -m "feat(184): manifest parent-dir fsync on first-create + repair-by-rename"
```

---

## Task 6: full read-back verifiers in `files.py`

**Files:**
- Modify: `algua/data/files.py` (add `parquet_file_row_count`, `parquet_dataset_row_count` after `read_partitioned_bars`)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_durability.py`:

```python
import pyarrow as pa
import pyarrow.parquet as pq


def test_parquet_file_row_count_reads_back_and_counts(tmp_path: Path) -> None:
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3, 4]}), p)
    assert files.parquet_file_row_count(p) == 4


def test_parquet_file_row_count_raises_on_truncated_file(tmp_path: Path) -> None:
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": list(range(1000))}), p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # lop off the tail (footer + pages)
    with pytest.raises(Exception):  # pyarrow raises on an unreadable/torn file
        files.parquet_file_row_count(p)


def test_parquet_dataset_row_count_sums_all_partitions(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    target = (tmp_path / "store") / rec.data_path
    assert files.parquet_dataset_row_count(target) == rec.row_count  # 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_durability.py -k "row_count" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'parquet_file_row_count'`.

- [ ] **Step 3: Implement**

In `algua/data/files.py`, add after `read_partitioned_bars`/`_ts_scalar` (end of file, ~line 193):

```python
def parquet_file_row_count(path: Path) -> int:
    """Full read-back of a single-file parquet: materialize the entire table (all columns,
    all row groups) so every data page is decompressed — power-loss truncation in the
    interior raises here — and return the row count. NOT a footer-only `metadata.num_rows`
    peek: that would report a count without touching the data pages (#184)."""
    return pq.read_table(path).num_rows


def parquet_dataset_row_count(dest_dir: Path) -> int:
    """Full read-back of a hive-partitioned bars dataset: read every partition's every column
    (`to_table(columns=None)`), forcing decompression of all data pages, and return the total
    row count. Must NOT use `count_rows()`/footer metadata/pruned-column reads — those skip the
    data pages this check exists to validate (#184)."""
    dataset = pads.dataset(dest_dir, format="parquet", partitioning=_BARS_PARTITIONING)
    return dataset.to_table(columns=None).num_rows
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_durability.py -k "row_count" -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/files.py tests/test_data_durability.py
git commit -m "feat(184): full read-back parquet row-count verifiers"
```

---

## Task 7: `verify_snapshot` / `verify_snapshots` on `DataStore`

**Files:**
- Modify: `algua/data/store.py:17-29` (add the two `parquet_*_row_count` imports)
- Modify: `algua/data/store.py` (add methods after `get_snapshot`, ~line 435)
- Test: `tests/test_data_durability.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_durability.py`:

```python
def test_verify_snapshot_healthy_bars(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    store.verify_snapshot(rec)  # must not raise


def test_verify_snapshot_detects_corrupt_part_file(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    part = next(((tmp_path / "store") / rec.data_path).rglob("*.parquet"))
    part.write_bytes(part.read_bytes()[: 8])  # corrupt the body
    with pytest.raises(Exception):
        store.verify_snapshot(rec)


def test_verify_snapshot_detects_missing_partition(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    sym_dir = next(p for p in ((tmp_path / "store") / rec.data_path).iterdir() if p.is_dir())
    for f in sym_dir.rglob("*"):
        if f.is_file():
            f.unlink()
    f_dir = sym_dir
    for child in sorted(f_dir.rglob("*"), reverse=True):
        child.rmdir() if child.is_dir() else None
    f_dir.rmdir()
    with pytest.raises(ValueError):  # row count now short
        store.verify_snapshot(rec)


def test_verify_snapshot_healthy_single_file_parquet(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = store.ingest_universe(
        universe="sp100", symbols=["AAPL", "MSFT"], effective_date="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00", source="fixture",
    )
    store.verify_snapshot(rec)  # must not raise


def test_verify_snapshot_detects_truncated_single_file(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = store.ingest_universe(
        universe="sp100", symbols=["AAPL", "MSFT"], effective_date="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00", source="fixture",
    )
    p = (tmp_path / "store") / rec.data_path
    p.write_bytes(p.read_bytes()[: 8])
    with pytest.raises(Exception):
        store.verify_snapshot(rec)


def test_verify_snapshot_byte_hash_branch_detects_tamper(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(src, index=False)
    store = DataStore(tmp_path / "store")
    rec = store.ingest_file(
        source_path=src, dataset="custom", provider="local", symbols=["AAPL"],
        start="2026-01-02", end="2026-01-02", as_of="2026-01-03T00:00:00+00:00",
        source="fixture", kind="custom",
    )
    p = (tmp_path / "store") / rec.data_path
    p.write_text("a\n9\n9\n9\n")  # same row count, different bytes
    with pytest.raises(ValueError):
        store.verify_snapshot(rec)


def test_verify_snapshot_missing_payload_path_fails_closed(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    import shutil as _sh
    _sh.rmtree((tmp_path / "store") / rec.data_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        store.verify_snapshot(rec)


def test_verify_snapshots_aggregates_and_flags_failures(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    good = _ingest_bars(store)
    bad = store.ingest_universe(
        universe="sp100", symbols=["AAPL"], effective_date="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00", source="fixture",
    )
    p = (tmp_path / "store") / bad.data_path
    p.write_bytes(p.read_bytes()[: 8])
    results = store.verify_snapshots()
    by_id = {r["snapshot_id"]: r for r in results}
    assert by_id[good.snapshot_id]["ok"] is True
    assert by_id[bad.snapshot_id]["ok"] is False
    assert by_id[bad.snapshot_id]["error"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_data_durability.py -k verify -v`
Expected: FAIL with `AttributeError: 'DataStore' object has no attribute 'verify_snapshot'`.

- [ ] **Step 3: Implement**

First, complete the `files` import in `algua/data/store.py` (the block edited in Task 3) by adding the two verifier names alphabetically:

```python
    parquet_dataset_row_count,
    parquet_file_row_count,
```

Then add to `DataStore`, right after `get_snapshot` (`algua/data/store.py:435`):

```python
    def verify_snapshot(self, rec: SnapshotRecord) -> None:
        """Power-loss read-back of one snapshot's payload (#184). Reads the bytes back to prove
        they are durable and decompressible, and checks the row count against the record. Raises
        on any damage (the caller decides how to surface it). Dispatch by `storage_format`:

        - ``parquet_dataset`` (bars): full read of every partition; summed rows == ``row_count``.
        - ``parquet`` (universe/fundamentals/news, or a ``.parquet`` via ``ingest_file``): full
          read of the single file; ``num_rows == row_count``. Readability check, NOT a
          content-hash recompute.
        - anything else (``ingest_file`` csv/generic): ``sha256_file == content_hash`` (a full
          read). Fails closed: a record whose ``content_hash`` is not a byte hash would report a
          (false) failure rather than a false pass — that signals the dispatch needs extending.
        """
        target = self.data_dir / rec.data_path
        fmt = rec.storage_format
        if fmt == "parquet_dataset":
            if not target.is_dir():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload dir missing at {target}")
            rows = parquet_dataset_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        elif fmt == "parquet":
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            rows = parquet_file_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        else:
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            actual = sha256_file(target)
            if actual != rec.content_hash:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: content hash {actual} != {rec.content_hash}"
                )

    def verify_snapshots(
        self, snapshot_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Verify one snapshot (`snapshot_id`) or all committed snapshots. Returns one result
        row per snapshot: ``{snapshot_id, dataset, storage_format, ok, error}``. Never raises for
        a damaged payload — the damage is captured in the row (`ok=False`); the caller decides
        the exit code. A missing `snapshot_id` itself raises `SnapshotNotFound`."""
        records = (
            [self.get_snapshot(snapshot_id)] if snapshot_id is not None else self.list_snapshots()
        )
        results: list[dict[str, Any]] = []
        for rec in records:
            row: dict[str, Any] = {
                "snapshot_id": rec.snapshot_id,
                "dataset": rec.dataset,
                "storage_format": rec.storage_format,
                "ok": True,
                "error": None,
            }
            try:
                self.verify_snapshot(rec)
            except Exception as exc:  # noqa: BLE001 - any read-back failure is a verify failure
                row["ok"] = False
                row["error"] = str(exc)
            results.append(row)
        return results
```

(Confirm `rec.dataset` is a valid accessor on `SnapshotRecord` — `read_bars` uses `rec.dataset` at `store.py:449`, so it is.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_data_durability.py -k verify -v`
Expected: PASS (all verify tests).

- [ ] **Step 5: Run the gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/store.py tests/test_data_durability.py
git commit -m "feat(184): DataStore.verify_snapshot(s) full read-back by storage_format"
```

---

## Task 8: `data verify` CLI command

**Files:**
- Modify: `algua/cli/data_cmd.py` (add `verify` command after `inspect`, ~line 324)
- Test: `tests/test_cli_data.py` (extend) — confirm the test module exists and follow its `CliRunner` pattern.

- [ ] **Step 1: Inspect the existing CLI test pattern**

Run: `uv run pytest tests/test_cli_data.py -q` and read `tests/test_cli_data.py` to copy its invocation pattern (Typer `CliRunner`, `ALGUA_DB_PATH`/`data_dir` fixture, JSON parse of `result.stdout`, `result.exit_code`).

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_cli_data.py` (adapt the store/data-dir fixture to match the module's existing helpers):

```python
def test_data_verify_all_healthy_exits_zero(tmp_path, monkeypatch):
    # build a store with one healthy snapshot via the same fixture the other tests use,
    # then invoke `data verify`. Expect exit_code 0 and ok:true rows.
    ...  # arrange a healthy snapshot using this module's existing store fixture
    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["failed"] == 0
    assert payload["verified"] >= 1


def test_data_verify_flags_damage_and_exits_nonzero(tmp_path, monkeypatch):
    ...  # arrange a snapshot, then corrupt its payload bytes on disk
    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failed"] == 1
    assert any(not s["ok"] for s in payload["snapshots"])


def test_data_verify_unknown_snapshot_id_exits_nonzero(tmp_path, monkeypatch):
    result = runner.invoke(app, ["data", "verify", "--snapshot-id", "deadbeefdeadbeef"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
```

(Fill the `...` arrange blocks using `tests/test_cli_data.py`'s existing store/data-dir setup — e.g. it already ingests fixtures and reads `out["snapshot"]["storage_format"]` at line 107, so reuse that scaffolding.)

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_cli_data.py -k verify -v`
Expected: FAIL — no `verify` command registered (Typer exits with usage error / non-2 code).

- [ ] **Step 4: Implement the command**

Add to `algua/cli/data_cmd.py` after `inspect` (line 324):

```python
@data_app.command("verify")
@json_errors(ValueError, LookupError, FileNotFoundError)
def verify(
    snapshot_id: str = typer.Option(None, "--snapshot-id", help="verify one snapshot"),
) -> None:
    """Power-loss backstop (#184): read each snapshot's payload back from disk and check it
    against its record. Reports one row per snapshot and exits non-zero if any failed."""
    results = _store().verify_snapshots(snapshot_id)
    failed = sum(1 for r in results if not r["ok"])
    emit(
        {
            "ok": failed == 0,
            "verified": len(results),
            "failed": failed,
            "snapshots": results,
        }
    )
    raise typer.Exit(code=0 if failed == 0 else 1)
```

Note: an unknown `--snapshot-id` makes `verify_snapshots` raise `SnapshotNotFound` (a `LookupError`), which `json_errors` renders as `{"ok": false, "error": ...}` + exit 1 — matching `test_data_verify_unknown_snapshot_id_exits_nonzero`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/test_cli_data.py -k verify -v`
Expected: PASS.

- [ ] **Step 6: Run the full gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/data_cmd.py tests/test_cli_data.py
git commit -m "feat(184): data verify CLI — read-back backstop, fails closed"
```

---

## Task 9: docs + final gate

**Files:**
- Modify: `CLAUDE.md` (command surface — add `data verify`)

- [ ] **Step 1: Add the command to the CLAUDE.md command surface**

In `CLAUDE.md`, under "## Command surface", after the `data inspect` line, add:

```markdown
- `uv run algua data verify [--snapshot-id ID]` — power-loss backstop: read each snapshot's
  payload back from disk (full read-back) and check it against its record; emits per-snapshot
  JSON and exits non-zero if any snapshot is damaged.
```

- [ ] **Step 2: Run the full quality gate one last time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass; no import-boundary violations (manifest→files is intra-package).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(184): document data verify in the command surface"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Write-side fsync (`fsync_file/fsync_dir/fsync_parents/fsync_tree`) → Task 1.
- `write_bytes_snapshot` temp + parent-chain fsync → Task 2.
- `ingest_file` fsync → Task 3.
- `_commit_bars_dir` publish + adoption barriers → Task 4.
- manifest `_repair` + conditional `append_if_absent` parent fsync → Task 5.
- `data verify` full read-back + closed dispatch + fail-closed → Tasks 6 (helpers), 7 (store), 8 (CLI).
- All 15 spec test cases map to Task 1/2/3/4/5 (write-side 1-7) and Task 6/7/8 (read-side 8-15).

**Type consistency:** `fsync_parents(path, *, stop_at)`, `fsync_tree(root)`, `parquet_file_row_count(path)`, `parquet_dataset_row_count(dest_dir)`, `verify_snapshot(rec)`, `verify_snapshots(snapshot_id=None)` are used identically wherever referenced. The `files` import block in `store.py` is edited once in Task 3 (fsync names) and once in Task 7 (the two `parquet_*_row_count` names) — keep both edits so each task stays green on its own.

**Declined (from GATE-1, not implemented):** fsync of the `ingest_file` staging source dir; a `content_hash_kind` schema column; an explicit "unsupported verify format" branch (the byte-hash catch-all fails closed instead).
