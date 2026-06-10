import pandas as pd

from algua.data.hindsight import query_news
from algua.data.store import DataStore


def _raw():
    return pd.DataFrame([
        {"source": "reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": None, "body": None},
    ])


def test_query_news_returns_full_history(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw())
    out = query_news(store, rec.snapshot_id)
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]


def test_query_news_symbol_filter(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw())
    out = query_news(store, rec.snapshot_id, symbols=["MSFT"])
    assert list(out["symbol"].unique()) == ["MSFT"]
