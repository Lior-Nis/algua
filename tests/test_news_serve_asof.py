from datetime import UTC, datetime

import pandas as pd

from algua.data.serve import StoreBackedNewsProvider
from algua.data.store import DataStore


def _raw(article_id, symbols, ka, pub):
    return {"source": "r", "article_id": article_id, "symbols": symbols,
            "published_at": pub, "knowable_at": ka, "headline": "h"}


def test_provider_returns_history_before_end_with_tombstones(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([
        _raw("a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        _raw("a1", ["AAPL"],         "2023-01-10T00:00:00Z", "2023-01-01T00:00:00Z"),
        _raw("a2", ["AAPL"],         "2023-02-01T00:00:00Z", "2023-02-01T00:00:00Z"),
    ])
    rec = store.ingest_news(provider="test", as_of="2023-03-01T00:00:00Z", frame=raw)
    prov = StoreBackedNewsProvider(store, rec.snapshot_id)
    end = datetime(2023, 1, 15, tzinfo=UTC)
    out = prov.get_news(["AAPL", "MSFT"], end)
    # knowable_at < end -> the Feb a2 row is excluded; the MSFT tombstone (Jan 10) is included
    assert out["knowable_at"].max() < pd.Timestamp(end)
    assert ((out["symbol"] == "MSFT") & out["retracted"]).any()
    assert prov.snapshot_id == rec.snapshot_id
