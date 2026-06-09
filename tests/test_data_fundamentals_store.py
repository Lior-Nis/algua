from pathlib import Path

import pandas as pd
import pytest

from algua.data.fundamentals_schema import validate_fundamentals
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "vendorX"],  # restate
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def _store(tmp_path: Path) -> DataStore:
    return DataStore(tmp_path)


def test_ingest_then_read_roundtrips_and_validates(tmp_path):
    store = _store(tmp_path)
    rec = store.ingest_fundamentals(
        provider="vendorX", symbols=["AAPL"], as_of="2025-09-01T00:00:00Z",
        source="vendorX", frame=_raw(),
    )
    assert rec.dataset == "fundamentals"
    back = store.read_fundamentals(rec.snapshot_id)
    validate_fundamentals(back)
    assert len(back) == 2


def test_ingest_is_idempotent(tmp_path):
    store = _store(tmp_path)
    a = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-09-01T00:00:00Z", source="vendorX", frame=_raw())
    b = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-09-01T00:00:00Z", source="vendorX", frame=_raw())
    assert a.snapshot_id == b.snapshot_id


def test_ingest_rejects_knowable_after_as_of(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="as_of"):
        store.ingest_fundamentals(provider="vendorX", symbols=["AAPL"],
                                  as_of="2025-06-01T00:00:00Z", source="vendorX", frame=_raw())


def test_read_filters_symbols(tmp_path):
    store = _store(tmp_path)
    two = pd.concat([_raw(), pd.DataFrame(
        [["MSFT", "2025-03-31", "revenue", 50.0, "2025-05-01T13:00:00Z", "vendorX"]],
        columns=_raw().columns)], ignore_index=True)
    rec = store.ingest_fundamentals(provider="vendorX", symbols=["AAPL", "MSFT"],
                                    as_of="2025-09-01T00:00:00Z", source="vendorX", frame=two)
    only = store.read_fundamentals(rec.snapshot_id, symbols=["AAPL"])
    assert set(only["symbol"]) == {"AAPL"}
