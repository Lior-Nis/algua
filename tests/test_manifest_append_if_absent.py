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
