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
