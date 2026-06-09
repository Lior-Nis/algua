import pandas as pd

from algua.contracts.types import FundamentalsProvider
from algua.data.hindsight import query_fundamentals
from algua.data.serve import StoreBackedFundamentalsProvider
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame(
        [
            ["AAPL", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "v"],
            ["AAPL", "2025-03-31", "revenue", 110.0, "2025-08-01T13:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def _seed(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_fundamentals(provider="v", symbols=["AAPL"],
                                    as_of="2025-09-01T00:00:00Z", source="v", frame=_raw())
    return store, rec.snapshot_id


def test_as_of_provider_satisfies_protocol_and_returns_full_history(tmp_path):
    store, sid = _seed(tmp_path)
    prov = StoreBackedFundamentalsProvider(store, sid)
    assert isinstance(prov, FundamentalsProvider)
    assert prov.snapshot_id == sid
    out = prov.get_fundamentals(["AAPL"], pd.Timestamp("2025-12-31", tz="UTC"))
    assert len(out) == 2  # provider returns full bitemporal history; engine masks per t


def test_get_fundamentals_excludes_at_or_after_end(tmp_path):
    store, sid = _seed(tmp_path)
    prov = StoreBackedFundamentalsProvider(store, sid)
    out = prov.get_fundamentals(["AAPL"], pd.Timestamp("2025-06-01", tz="UTC"))
    assert len(out) == 1  # only the 2025-05-01 row is knowable before end


def test_hindsight_returns_everything(tmp_path):
    store, sid = _seed(tmp_path)
    out = query_fundamentals(store, sid, symbols=["AAPL"])
    assert len(out) == 2
