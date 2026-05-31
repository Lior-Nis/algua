from pathlib import Path

import pandas as pd
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


def test_ingest_non_csv_records_unknown_row_count(tmp_path):
    source = tmp_path / "bars.txt"
    source.write_text("not,csv,contract-specific\n", encoding="utf-8")
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

    assert rec.row_count is None
    assert store.get_snapshot(rec.snapshot_id).row_count is None


def test_ingest_bars_writes_parquet_snapshot_with_provenance(tmp_path):
    store = DataStore(tmp_path / "data")
    frame = pd.DataFrame(
        {
            "ts": ["2026-01-02", "2026-01-03"],
            "symbol": ["AAPL", "AAPL"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "close": [100.0, 101.0],
            "adj_close": [99.5, 100.5],
            "volume": [1000.0, 1100.0],
        }
    )

    rec = store.ingest_bars(
        provider="fixture",
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-03",
        as_of="2026-01-04T00:00:00+00:00",
        source="fixture",
        frame=frame,
        timeframe="1d",
        adjustment="none",
        source_metadata={"fixture": "true"},
    )

    assert rec.dataset == "bars"
    assert rec.kind == "bars"
    assert rec.storage_format == "parquet"
    assert rec.row_count == 2
    assert rec.metadata.timeframe == "1d"
    assert rec.metadata.adjustment == "none"
    assert rec.metadata.source_metadata == {"fixture": "true"}
    saved = pd.read_parquet(tmp_path / "data" / rec.data_path)
    assert list(saved["close"]) == [100.0, 101.0]


def test_ingest_bars_rejects_frames_outside_bar_schema(tmp_path):
    store = DataStore(tmp_path / "data")
    frame = pd.DataFrame({"ts": ["2026-01-02"], "symbol": ["AAPL"], "close": [100.0]})

    with pytest.raises(ValueError, match="missing bar columns"):
        store.ingest_bars(
            provider="fixture",
            symbols=["AAPL"],
            start="2026-01-02",
            end="2026-01-02",
            as_of="2026-01-03T00:00:00+00:00",
            source="fixture",
            frame=frame,
        )

    assert store.list_snapshots() == []


def test_ingest_universe_writes_point_in_time_membership(tmp_path):
    store = DataStore(tmp_path / "data")

    rec = store.ingest_universe(
        universe="core",
        symbols=["msft", "AAPL", "AAPL"],
        effective_date="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00",
        source="manual",
    )

    assert rec.dataset == "universes"
    assert rec.kind == "universe"
    assert rec.symbols == ("AAPL", "MSFT")
    assert rec.row_count == 2
    saved = pd.read_parquet(tmp_path / "data" / rec.data_path)
    assert list(saved["symbol"]) == ["AAPL", "MSFT"]


def test_summary_groups_snapshots_by_dataset(tmp_path):
    source = tmp_path / "bars.csv"
    source.write_text("ts,symbol,close\n2026-01-02,AAPL,100\n", encoding="utf-8")
    store = DataStore(tmp_path / "data")
    store.ingest_file(
        dataset="daily-bars",
        provider="local",
        symbols=["AAPL"],
        start="2026-01-02",
        end="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00",
        source="fixture",
        file_path=source,
    )

    summary = store.summary()

    assert summary["snapshots"] == 1
    assert summary["datasets"][0]["dataset"] == "daily-bars"
    assert summary["datasets"][0]["symbols"] == ["AAPL"]


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
