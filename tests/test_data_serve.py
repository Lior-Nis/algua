from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import DataProvider
from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore


def _ingest(store: DataStore):
    frame = pd.DataFrame({
        "ts": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02", "2024-07-03"],
        "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA"],
        "open": [10.0, 20.0, 11.0, 21.0, 12.0], "high": [10.0, 20.0, 11.0, 21.0, 12.0],
        "low": [10.0, 20.0, 11.0, 21.0, 12.0], "close": [10.0, 20.0, 11.0, 21.0, 12.0],
        "adj_close": [10.0, 20.0, 11.0, 21.0, 12.0], "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    return store.ingest_bars(
        provider="test", symbols=["AAA", "BBB"], start="2024-07-01", end="2024-07-03",
        as_of="2024-07-04", source="unit-test", frame=frame, timeframe="1d", adjustment="none",
    )


def test_get_bars_rejects_timeframe_mismatch(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)  # ingested with timeframe="1d"
    provider = StoreBackedProvider(store, rec.snapshot_id)
    with pytest.raises(ValueError):
        provider.get_bars(["AAA"], datetime(2024, 7, 1, tzinfo=UTC),
                          datetime(2024, 7, 3, tzinfo=UTC), "1h")


def test_provider_satisfies_protocol_and_filters(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    provider = StoreBackedProvider(store, rec.snapshot_id)
    assert isinstance(provider, DataProvider)
    assert provider.snapshot_id == rec.snapshot_id

    out = provider.get_bars(
        ["AAA"], datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 3, tzinfo=UTC), "1d"
    )
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert set(out["symbol"]) == {"AAA"}                              # symbol filter
    assert out.index.min() == pd.Timestamp("2024-07-01", tz="UTC")
    # half-open [start, end): the 07-03 bar sits on `end` and is excluded
    assert out.index.max() == pd.Timestamp("2024-07-02", tz="UTC")


def test_get_bars_end_boundary_is_half_open(tmp_path):
    """A bar timestamped exactly at `end` is excluded (look-ahead-safe [start, end))."""
    store = DataStore(tmp_path)
    rec = _ingest(store)
    provider = StoreBackedProvider(store, rec.snapshot_id)

    out = provider.get_bars(
        ["AAA"], datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 2, tzinfo=UTC), "1d"
    )

    assert list(out.index) == [pd.Timestamp("2024-07-01", tz="UTC")]  # 07-02 == end, excluded
