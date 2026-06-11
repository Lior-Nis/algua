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
    stale = tmp_path / "manifest.jsonl.repair-deadbeef.tmp"
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


def test_entire_content_is_uncommitted_single_line_no_newline(tmp_path):
    # A manifest whose ENTIRE content is one uncommitted line (no newline anywhere).
    # list_records() must return [] and append_if_absent must repair to empty then append.
    manifest_path = tmp_path / "manifest.jsonl"
    uncommitted = make_record("zzz9")
    manifest_path.write_bytes(
        json.dumps(uncommitted.to_dict(), sort_keys=True).encode("utf-8")
        # no trailing newline — never committed
    )
    manifest = SnapshotManifest(manifest_path)
    assert manifest.list_records() == []
    new_rec = make_record("aaa1")
    out = manifest.append_if_absent(new_rec)
    assert out is new_rec
    recs = manifest.list_records()
    # zzz9 was uncommitted and must NOT appear; only the newly appended aaa1 is present
    assert [r.snapshot_id for r in recs] == ["aaa1"]


def test_torn_multi_byte_utf8_tail_drops_without_raising(tmp_path):
    # A torn tail that splits a multi-byte UTF-8 character must be silently dropped.
    # "é" encodes as b"\xc3\xa9"; writing only the first byte (b"\xc3") after a committed
    # line simulates an OS crash mid-write of a multi-byte sequence.
    manifest_path = tmp_path / "manifest.jsonl"
    committed = make_record("aaa1")
    manifest_path.write_bytes(
        json.dumps(committed.to_dict(), sort_keys=True).encode("utf-8")
        + b"\n"
        + b"\xc3"  # first byte of "é" — torn tail, not a complete UTF-8 sequence
    )
    # Reading must not raise UnicodeDecodeError
    manifest = SnapshotManifest(manifest_path)
    recs = manifest.list_records()
    assert [r.snapshot_id for r in recs] == ["aaa1"]
    # append_if_absent must repair the torn tail and append successfully
    new_rec = make_record("bbb2")
    out = manifest.append_if_absent(new_rec)
    assert out is new_rec
    assert [r.snapshot_id for r in manifest.list_records()] == ["aaa1", "bbb2"]
