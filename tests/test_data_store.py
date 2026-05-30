from pathlib import Path

import pytest

from algua.data.store import DataStore, SnapshotNotFound


def test_ingest_file_copies_payload_and_records_manifest(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    store = DataStore(tmp_path / "data")

    rec = store.ingest_file(
        dataset="daily-bars",
        provider="local",
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00",
        source="fixture",
        file_path=source,
    )

    assert rec.dataset == "daily-bars"
    assert rec.symbols == ("AAPL",)
    assert rec.row_count == 1
    assert rec.snapshot_id
    assert (tmp_path / "data" / rec.data_path).read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    assert store.list_snapshots() == [rec]


def test_ingest_same_file_is_idempotent(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    store = DataStore(tmp_path / "data")

    first = store.ingest_file(
        dataset="daily-bars",
        provider="local",
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00",
        source="fixture",
        file_path=source,
    )
    second = store.ingest_file(
        dataset="daily-bars",
        provider="local",
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00",
        source="fixture",
        file_path=source,
    )

    assert second == first
    assert len(store.list_snapshots()) == 1


def test_get_snapshot_raises_for_unknown_id(tmp_path):
    store = DataStore(tmp_path / "data")

    with pytest.raises(SnapshotNotFound):
        store.get_snapshot("missing")


def test_ingest_rejects_invalid_date_bounds(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n", encoding="utf-8")
    store = DataStore(tmp_path / "data")

    with pytest.raises(ValueError, match="start must be <= end"):
        store.ingest_file(
            dataset="daily-bars",
            provider="local",
            symbols=["AAPL"],
            start="2026-01-03",
            end="2026-01-02",
            as_of="2026-01-03T00:00:00+00:00",
            source="fixture",
            file_path=source,
        )


def test_ingest_requires_existing_file(tmp_path):
    store = DataStore(tmp_path / "data")

    with pytest.raises(FileNotFoundError):
        store.ingest_file(
            dataset="daily-bars",
            provider="local",
            symbols=["AAPL"],
            start="2026-01-02",
            end="2026-01-02",
            as_of="2026-01-03T00:00:00+00:00",
            source="fixture",
            file_path=Path("missing.csv"),
        )
