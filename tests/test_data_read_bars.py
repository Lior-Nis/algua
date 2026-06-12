from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.data.store import DataStore, SnapshotNotFound


def _ingest(store: DataStore):
    frame = pd.DataFrame({
        "ts": [
            "2024-07-01T00:00:00+00:00", "2024-07-01T00:00:00+00:00",
            "2024-07-02T00:00:00+00:00", "2024-07-02T00:00:00+00:00",
        ],
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "open": [10.0, 20.0, 11.0, 21.0], "high": [10.0, 20.0, 11.0, 21.0],
        "low": [10.0, 20.0, 11.0, 21.0], "close": [10.0, 20.0, 11.0, 21.0],
        "adj_close": [10.0, 20.0, 11.0, 21.0], "volume": [100.0, 200.0, 110.0, 210.0],
    })
    return store.ingest_bars(
        provider="test", symbols=["AAA", "BBB"], start="2024-07-01", end="2024-07-02",
        as_of="2024-07-03", source="unit-test", frame=frame, timeframe="1d", adjustment="none",
    )


def _make_non_bars_snapshot(store: DataStore, tmp_path):
    return store.ingest_universe(
        universe="test-universe",
        symbols=["AAA", "BBB"],
        effective_date="2024-07-01",
        as_of="2024-07-03",
        source="unit-test",
    )


def test_read_bars_returns_bar_schema(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    out = store.read_bars(rec.snapshot_id)
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2024-07-01", tz="UTC")
    assert len(out) == 4


def test_read_bars_unknown_id_raises(tmp_path):
    with pytest.raises(SnapshotNotFound):
        DataStore(tmp_path).read_bars("does-not-exist")


def test_read_bars_rejects_non_bars_dataset(tmp_path):
    store = DataStore(tmp_path)
    rec = _make_non_bars_snapshot(store, tmp_path)
    with pytest.raises(ValueError):
        store.read_bars(rec.snapshot_id)


def test_read_bars_pushes_down_symbol_and_window(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)  # AAA/BBB on 2024-07-01 and 2024-07-02
    out = store.read_bars(
        rec.snapshot_id, symbols=["AAA"],
        start=datetime(2024, 7, 1, tzinfo=UTC), end=datetime(2024, 7, 2, tzinfo=UTC),
    )
    validate_bars(out)
    assert set(out["symbol"]) == {"AAA"}
    assert out.index.max() == pd.Timestamp("2024-07-01", tz="UTC")  # 07-02 == end, excluded


def test_read_bars_empty_window_returns_typed_empty(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    out = store.read_bars(
        rec.snapshot_id, symbols=["AAA"],
        start=datetime(2030, 1, 1, tzinfo=UTC), end=datetime(2030, 1, 2, tzinfo=UTC),
    )
    validate_bars(out)
    assert out.empty
    assert list(out.columns) == BAR_COLUMNS


def test_read_bars_rejects_legacy_single_file_snapshot(tmp_path):
    from pathlib import Path

    from algua.data.manifest import SnapshotManifest
    from algua.data.models import SnapshotMetadata, SnapshotRecord

    store = DataStore(tmp_path)
    _ingest(store)
    # Forge a manifest record claiming the old single-file layout for a bars dataset.
    legacy = SnapshotRecord(
        snapshot_id="legacyid00000000",
        metadata=SnapshotMetadata(
            dataset="bars", provider="p", symbols=("AAA",), start="2024-07-01",
            end="2024-07-01", as_of="2024-07-02T00:00:00+00:00", source="s", kind="bars",
            timeframe="1d", adjustment="none",
        ),
        row_count=1, content_hash="h",
        data_path=Path("snapshots/bars/legacyid00000000/bars.parquet"),
        created_at="2024-07-02T00:00:00+00:00", storage_format="parquet",
    )
    SnapshotManifest(tmp_path / "manifest.jsonl").append_if_absent(legacy)
    with pytest.raises(ValueError, match="legacy"):
        store.read_bars("legacyid00000000")
