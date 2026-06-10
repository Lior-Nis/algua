import pandas as pd
import pytest

from algua.data.news_schema import (
    explode_news_symbols,
    logical_news_hash,
    to_news_schema,
)
from algua.data.store import DataStore


def _raw(tmp_path):
    return pd.DataFrame([
        {"source": "Reuters", "article_id": "a1", "symbols": "AAPL,MSFT",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": "http://x/1", "body": "b"},
    ])


def test_ingest_then_read_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="testfeed", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    assert rec.dataset == "news"
    out = store.read_news(rec.snapshot_id)
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert rec.metadata.source == "testfeed"             # metadata.source = provider label
    assert "reuters" in rec.metadata.source_metadata["row_sources"]


def test_ingest_is_deterministic_dedup(tmp_path):
    store = DataStore(tmp_path)
    r1 = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    r2 = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    assert r1.snapshot_id == r2.snapshot_id


def test_ingest_rejects_knowable_after_as_of(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="as_of"):
        store.ingest_news(provider="f", as_of="2025-01-01T00:00:00Z", frame=_raw(tmp_path))


def test_read_news_symbol_pushdown(tmp_path):
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=_raw(tmp_path))
    out = store.read_news(rec.snapshot_id, symbols=["AAPL"])
    assert list(out["symbol"].unique()) == ["AAPL"]


def test_nullable_text_survives_parquet_roundtrip_and_hash_is_stable(tmp_path):
    # GATE-2: null url/body must come back as pd.NA (not the string "None"/"nan") through parquet,
    # and the read-back frame must hash identically to the ingested snapshot (no dtype drift).
    raw = pd.DataFrame([
        {"source": "reuters", "article_id": "a1", "symbols": "AAPL",
         "published_at": "2025-01-02T13:00:00Z", "knowable_at": "2025-01-02T13:00:00Z",
         "headline": "h", "url": None, "body": None},
    ])
    canon_before = to_news_schema(explode_news_symbols(raw.copy()))
    store = DataStore(tmp_path)
    rec = store.ingest_news(provider="f", as_of="2025-01-03T00:00:00Z", frame=raw)
    out = store.read_news(rec.snapshot_id)
    assert out["url"].isna().all() and out["body"].isna().all()
    assert logical_news_hash(out) == logical_news_hash(canon_before) == rec.content_hash
