from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algua.data import files
from algua.data.manifest import SnapshotManifest
from algua.data.models import SnapshotMetadata, SnapshotRecord
from algua.data.store import DataStore


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

    # root itself is fsynced last; every symbol dir fsync comes after its own part file
    assert order[-1] == (str(tmp_path), True)
    for sym in ("A", "B"):
        d = str(tmp_path / f"symbol={sym}")
        f = str(tmp_path / f"symbol={sym}" / "part-0.parquet")
        file_idx = next(i for i, (p, is_dir) in enumerate(order) if p == f and not is_dir)
        dir_idx = next(i for i, (p, is_dir) in enumerate(order) if p == d and is_dir)
        assert file_idx < dir_idx


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

    assert str(leaf) in fsynced
    assert str(root / "snapshots" / "bars") in fsynced
    assert str(root / "snapshots") in fsynced
    assert str(root) in fsynced
    assert str(root.parent) not in fsynced


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
        file_path=src, dataset="custom", provider="local", symbols=["AAPL"],
        start="2026-01-02", end="2026-01-02", as_of="2026-01-03T00:00:00+00:00",
        source="fixture", kind="custom",
    )
    # a payload fsync precedes its replace; parent-chain fsyncs follow it
    assert "fsync" in events and "replace" in events
    first_replace = events.index("replace")
    assert events.index("fsync") < first_replace
    assert events[first_replace + 1 :].count("fsync") >= 1


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
    monkeypatch.setattr(
        store_mod, "fsync_tree", lambda p: (events.append("tree"), real_tree(p))[1]
    )
    monkeypatch.setattr(
        os, "replace", lambda s, d: (events.append("replace"), real_replace(s, d))[1]
    )
    monkeypatch.setattr(
        store_mod,
        "fsync_parents",
        lambda p, *, stop_at: (events.append("parents"), real_parents(p, stop_at=stop_at))[1],
    )

    _ingest_bars(store)
    assert events[:3] == ["tree", "replace", "parents"]


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


def test_append_fsyncs_parent_on_every_append(tmp_path: Path, monkeypatch) -> None:
    import algua.data.manifest as man_mod
    manifest = SnapshotManifest(tmp_path / "manifest.jsonl")

    dir_fsyncs: list[str] = []
    real = man_mod.fsync_dir
    monkeypatch.setattr(man_mod, "fsync_dir", lambda p: (dir_fsyncs.append(str(p)), real(p))[1])

    manifest.append_if_absent(_rec("aaaaaaaaaaaaaaaa"))  # first creation
    assert str(tmp_path) in dir_fsyncs  # parent dir fsynced

    dir_fsyncs.clear()
    manifest.append_if_absent(_rec("bbbbbbbbbbbbbbbb"))  # append to existing file
    assert str(tmp_path) in dir_fsyncs  # parent dir fsynced unconditionally (commit-point safety)


def test_repair_fsyncs_parent_after_rename(tmp_path: Path, monkeypatch) -> None:
    import algua.data.manifest as man_mod
    path = tmp_path / "manifest.jsonl"
    good = '{"x": 1}\n'  # _repair just rewrites the committed prefix bytes verbatim
    manifest = SnapshotManifest(path)
    path.write_text(good + "uncommitted-no-newline")

    dir_fsyncs: list[str] = []
    real = man_mod.fsync_dir
    monkeypatch.setattr(man_mod, "fsync_dir", lambda p: (dir_fsyncs.append(str(p)), real(p))[1])

    manifest._repair(good.encode("utf-8"))
    assert path.read_text() == good
    assert str(tmp_path) in dir_fsyncs


def test_commit_bars_adoption_fsyncs_before_manifest_append(tmp_path: Path, monkeypatch) -> None:
    # First ingest publishes the dir. Blank the manifest so a second ingest of the SAME id
    # re-enters _commit_bars_dir, hits ENOTEMPTY (target exists), and ADOPTS it.
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    manifest_path = (tmp_path / "store" / "manifest.jsonl")
    manifest_path.write_text("")  # drop the committed record, keep the payload dir

    import algua.data.store as store_mod
    events: list[str] = []
    real_tree = store_mod.fsync_tree
    real_parents = store_mod.fsync_parents
    monkeypatch.setattr(
        store_mod,
        "fsync_tree",
        lambda p: (events.append(f"tree:{p.name}"), real_tree(p))[1],
    )
    monkeypatch.setattr(
        store_mod,
        "fsync_parents",
        lambda p, *, stop_at: (
            events.append(f"parents:{p.name}"), real_parents(p, stop_at=stop_at)
        )[1],
    )
    orig_append = store.manifest.append_if_absent
    monkeypatch.setattr(
        store.manifest,
        "append_if_absent",
        lambda r: (events.append("append"), orig_append(r))[1],
    )

    again = _ingest_bars(store)
    assert again.snapshot_id == rec.snapshot_id
    # the ADOPTION barrier fsync_tree(target) + fsync_parents(target) ran BEFORE the manifest
    # append (identified by the snapshot-id dir name, not the staging uuid)
    sid = rec.snapshot_id
    assert f"tree:{sid}" in events
    assert f"parents:{sid}" in events
    assert events.index(f"tree:{sid}") < events.index("append")
    assert events.index(f"parents:{sid}") < events.index("append")


def test_parquet_file_row_count_reads_back_and_counts(tmp_path: Path) -> None:
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3, 4]}), p)
    assert files.parquet_file_row_count(p) == 4


def test_parquet_file_row_count_raises_on_truncated_file(tmp_path: Path) -> None:
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": list(range(1000))}), p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # lop off the tail (footer + pages)
    with pytest.raises((OSError, ValueError)):
        files.parquet_file_row_count(p)


def test_parquet_dataset_row_count_sums_all_partitions(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    target = (tmp_path / "store") / rec.data_path
    assert files.parquet_dataset_row_count(target) == rec.row_count  # 4


def test_verify_snapshot_healthy_bars(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    store.verify_snapshot(rec)  # must not raise


def test_verify_snapshot_detects_corrupt_part_file(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    part = next(((tmp_path / "store") / rec.data_path).rglob("*.parquet"))
    part.write_bytes(part.read_bytes()[:8])  # corrupt the body
    with pytest.raises((OSError, ValueError)):
        store.verify_snapshot(rec)


def test_verify_snapshot_detects_missing_partition(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    rec = _ingest_bars(store)
    target = (tmp_path / "store") / rec.data_path
    sym_dir = next(p for p in target.iterdir() if p.is_dir())
    import shutil as _sh
    _sh.rmtree(sym_dir)  # drop one whole symbol partition
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
    p.write_bytes(p.read_bytes()[:8])
    with pytest.raises((OSError, ValueError)):
        store.verify_snapshot(rec)


def test_verify_snapshot_byte_hash_branch_detects_tamper(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(src, index=False)
    store = DataStore(tmp_path / "store")
    rec = store.ingest_file(
        file_path=src, dataset="custom", provider="local", symbols=["AAPL"],
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
    p.write_bytes(p.read_bytes()[:8])
    results = store.verify_snapshots()
    by_id = {r["snapshot_id"]: r for r in results}
    assert by_id[good.snapshot_id]["ok"] is True
    assert by_id[bad.snapshot_id]["ok"] is False
    assert by_id[bad.snapshot_id]["error"]
