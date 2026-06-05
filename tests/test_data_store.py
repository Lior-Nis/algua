from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from algua.data.files import count_tabular_rows
from algua.data.store import DataStore, SnapshotNotFound


def test_count_tabular_rows_streams_csv(tmp_path):
    source = tmp_path / "rows.csv"
    source.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    assert count_tabular_rows(source) == 3


def test_count_tabular_rows_counts_parquet(tmp_path):
    source = tmp_path / "rows.parquet"
    pd.DataFrame({"a": [1, 2, 3, 4]}).to_parquet(source, index=False)
    assert count_tabular_rows(source) == 4


def test_count_tabular_rows_unknown_format_is_none(tmp_path):
    source = tmp_path / "rows.txt"
    source.write_text("nope\n", encoding="utf-8")
    assert count_tabular_rows(source) is None


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
            "ts": ["2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00"],
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
    frame = pd.DataFrame(
        {"ts": ["2026-01-02T00:00:00+00:00"], "symbol": ["AAPL"], "close": [100.0]}
    )

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


def _bars_frame(ts: list[str]) -> pd.DataFrame:
    n = len(ts)
    return pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["AAPL"] * n,
            "open": [99.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [98.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "adj_close": [99.5 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        }
    )


def test_ingest_bars_is_idempotent_on_identical_content(tmp_path):
    # Regression for #55: a byte-identical dataset must produce a stable snapshot_id / content_hash
    # so re-ingestion dedups instead of writing a second snapshot.
    ts = ["2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00"]
    store = DataStore(tmp_path / "data")
    kwargs = dict(
        provider="fixture", symbols=["AAPL"], start="2026-01-02", end="2026-01-03",
        as_of="2026-01-04T00:00:00+00:00", source="fixture", timeframe="1d", adjustment="none",
    )
    first = store.ingest_bars(frame=_bars_frame(ts), **kwargs)
    second = store.ingest_bars(frame=_bars_frame(ts), **kwargs)

    assert second == first
    assert len(store.list_snapshots()) == 1


def test_content_hash_is_canonical_parquet_bytes(tmp_path):
    # #55: hash is the sha256 of the canonical on-disk parquet bytes, so what's hashed is exactly
    # what's stored and dedup is reproducible across runs.
    from algua.data.files import sha256_file

    ts = ["2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00"]
    store = DataStore(tmp_path / "data")
    rec = store.ingest_bars(
        provider="fixture", symbols=["AAPL"], start="2026-01-02", end="2026-01-03",
        as_of="2026-01-04T00:00:00+00:00", source="fixture", frame=_bars_frame(ts),
    )
    assert rec.content_hash == sha256_file(tmp_path / "data" / rec.data_path)


def test_from_dict_requires_schema_version():
    # #61: no silent v1 fallback — a record without schema_version is rejected.
    from algua.data.models import SnapshotRecord

    payload = {
        "snapshot_id": "abc", "dataset": "bars", "provider": "p", "symbols": ["AAPL"],
        "start": "2026-01-02", "end": "2026-01-02", "as_of": "2026-01-03T00:00:00+00:00",
        "source": "s", "kind": "bars", "row_count": 1, "content_hash": "h",
        "data_path": "snapshots/bars/abc/bars.parquet", "storage_format": "parquet",
        "created_at": "2026-01-03T00:00:00+00:00",
    }
    with pytest.raises(KeyError):
        SnapshotRecord.from_dict(payload)


def test_normalize_symbols_is_shared_public_helper():
    # #63: one normalization helper (strip/upper/sort/dedup) reused across layers.
    from algua.data.store import normalize_symbols

    assert normalize_symbols([" aapl ", "MSFT", "aapl"]) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError):
        normalize_symbols([" ", ""])


def test_dataset_kind_constants_match_records(tmp_path):
    # #62: dataset/kind routing keys are centralized constants the store actually uses.
    from algua.data.models import Dataset, Kind

    store = DataStore(tmp_path / "data")
    rec = store.ingest_universe(
        universe="core", symbols=["AAPL"], effective_date="2026-01-02",
        as_of="2026-01-03T00:00:00+00:00", source="manual",
    )
    assert rec.dataset == Dataset.UNIVERSES.value
    assert rec.kind == Kind.UNIVERSE.value


def test_read_universe_builds_sorted_effective_dated_timeline(tmp_path):
    # #7: a time-varying universe = multiple snapshots sharing the NAME, one per effective_date.
    # read_universe returns them as an effective-date-sorted timeline of UniverseSnapshots.
    from algua.data.models import UniverseSnapshot

    store = DataStore(tmp_path / "data")
    store.ingest_universe(
        universe="core", symbols=["AAPL", "MSFT"], effective_date="2026-03-01",
        as_of="2026-03-02T00:00:00+00:00", source="manual",
    )
    store.ingest_universe(
        universe="core", symbols=["AAPL"], effective_date="2026-01-01",
        as_of="2026-01-02T00:00:00+00:00", source="manual",
    )
    # A different universe name must not bleed into `core`.
    store.ingest_universe(
        universe="other", symbols=["TSLA"], effective_date="2026-01-01",
        as_of="2026-01-02T00:00:00+00:00", source="manual",
    )

    timeline = store.read_universe("core")

    assert all(isinstance(s, UniverseSnapshot) for s in timeline)
    assert [s.effective_date for s in timeline] == [date(2026, 1, 1), date(2026, 3, 1)]
    assert timeline[0].symbols == frozenset({"AAPL"})
    assert timeline[1].symbols == frozenset({"AAPL", "MSFT"})


def test_read_universe_empty_for_unknown_name(tmp_path):
    store = DataStore(tmp_path / "data")
    store.ingest_universe(
        universe="core", symbols=["AAPL"], effective_date="2026-01-01",
        as_of="2026-01-02T00:00:00+00:00", source="manual",
    )
    assert store.read_universe("nonexistent") == []


def test_read_universe_rejects_ambiguous_duplicate_effective_date(tmp_path):
    # Two snapshots for the SAME name + SAME effective_date but DIFFERENT membership =>
    # the as-of-date answer is ambiguous; refuse rather than silently pick one.
    store = DataStore(tmp_path / "data")
    store.ingest_universe(
        universe="core", symbols=["AAPL"], effective_date="2026-01-01",
        as_of="2026-01-02T00:00:00+00:00", source="manual",
    )
    store.ingest_universe(
        universe="core", symbols=["MSFT"], effective_date="2026-01-01",
        as_of="2026-01-03T00:00:00+00:00", source="manual",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        store.read_universe("core")


def test_read_universe_allows_identical_duplicate_effective_date(tmp_path):
    # Same effective_date with IDENTICAL membership is not ambiguous (idempotent re-ingest of
    # the exact same parquet dedups to one snapshot anyway; even if two records exist they agree).
    store = DataStore(tmp_path / "data")
    store.ingest_universe(
        universe="core", symbols=["AAPL", "MSFT"], effective_date="2026-01-01",
        as_of="2026-01-02T00:00:00+00:00", source="manual",
    )
    store.ingest_universe(
        universe="core", symbols=["MSFT", "AAPL"], effective_date="2026-01-01",
        as_of="2026-01-09T00:00:00+00:00", source="manual",
    )
    timeline = store.read_universe("core")
    assert [s.effective_date for s in timeline] == [date(2026, 1, 1)]
    assert timeline[0].symbols == frozenset({"AAPL", "MSFT"})
