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
from algua.data.manifest import SnapshotManifest
from algua.data.store import DataStore


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
    with pytest.raises(Exception):  # noqa: B017 — pyarrow raises its own invalid-file error; type is not our API
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"AAA"}
        )


def test_validate_partitioned_bars_dir_handles_dotted_symbols(tmp_path):
    canon = _bars_canon(["BRK.B"])
    write_partitioned_bars(canon, tmp_path / "ds")
    validate_partitioned_bars_dir(
        tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"BRK.B"}
    )


def _ingest_bars(store: DataStore, symbols: list[str] = ["AAA"]):  # noqa: B006 — read-only default
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
