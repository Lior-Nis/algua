import pandas as pd

from algua.backtest.engine import _news_as_of
from algua.data.news_schema import explode_news_symbols, to_news_schema


def _news():
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
         "headline": "h1"},
        {"source": "r", "article_id": "a1", "symbols": ["AAPL"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-10T00:00:00Z",
         "headline": "h2"},
    ])
    return to_news_schema(explode_news_symbols(raw))


def test_as_of_before_drop_shows_both():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-05T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert not out["retracted"].any()  # tombstones never surface in the as-of view


def test_as_of_after_drop_excludes_retracted_symbol():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-15T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL"}  # MSFT's latest revision is a tombstone -> dropped


def test_as_of_excludes_future_knowable():
    out = _news_as_of(_news(), pd.Timestamp("2022-12-31T00:00:00Z"))
    assert out.empty


def test_as_of_empty_preserves_columns_and_dtypes():
    out = _news_as_of(_news(), pd.Timestamp("2022-01-01T00:00:00Z"))
    assert list(out.columns) == list(_news().columns)
    assert out["retracted"].dtype == _news()["retracted"].dtype
