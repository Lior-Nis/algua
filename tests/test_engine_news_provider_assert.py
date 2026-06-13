"""Strengthen _assert_news_shape — rejects bad provider frames (parallel to fundamentals)."""
from __future__ import annotations

import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, _assert_news_shape
from algua.data.news_schema import explode_news_symbols, to_news_schema


def _good_news_frame() -> pd.DataFrame:
    """A valid, contract-shaped news frame."""
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
         "headline": "h1"},
    ])
    return to_news_schema(explode_news_symbols(raw))


def test_good_frame_passes():
    _assert_news_shape(_good_news_frame())  # no raise


def test_missing_column_raises():
    bad = _good_news_frame().drop(columns=["knowable_at"])
    with pytest.raises(BacktestError, match="missing columns"):
        _assert_news_shape(bad)


def test_tz_naive_published_at_raises():
    bad = _good_news_frame().copy()
    bad["published_at"] = bad["published_at"].dt.tz_localize(None)
    with pytest.raises(BacktestError, match="tz-aware UTC"):
        _assert_news_shape(bad)


def test_knowable_before_published_raises():
    # Mutate a valid frame post-construction so published_at > knowable_at (validation
    # already passed during to_news_schema).
    bad = _good_news_frame().copy()
    bad.loc[0, "published_at"] = pd.Timestamp("2023-02-01T00:00:00Z")  # > knowable_at
    with pytest.raises(BacktestError, match="must be >= 'published_at'"):
        _assert_news_shape(bad)


def test_non_bool_retracted_raises():
    bad = _good_news_frame().copy()
    bad["retracted"] = bad["retracted"].astype("object")
    with pytest.raises(BacktestError, match="non-nullable bool"):
        _assert_news_shape(bad)


def test_duplicate_bitemporal_key_raises():
    good = _good_news_frame()
    dup = pd.concat([good, good], ignore_index=True)
    with pytest.raises(BacktestError, match="duplicate"):
        _assert_news_shape(dup)
