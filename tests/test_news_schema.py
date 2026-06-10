import pandas as pd
import pytest

from algua.data.news_schema import NEWS_COLUMNS, empty_news, validate_news


def _row(**over):
    base = {
        "source": "reuters",
        "article_id": "a1",
        "symbol": "AAPL",
        "published_at": pd.Timestamp("2025-01-02T13:00:00Z"),
        "knowable_at": pd.Timestamp("2025-01-02T13:00:00Z"),
        "headline": "Apple ships",
        "url": "http://x/1",
        "body": "body text",
    }
    base.update(over)
    return base


def _frame(rows):
    return pd.DataFrame(rows, columns=list(NEWS_COLUMNS))


def test_valid_frame_passes_unchanged():
    df = _frame([_row()])
    assert validate_news(df) is df


def test_rejects_wrong_columns():
    df = _frame([_row()]).drop(columns=["body"])
    with pytest.raises(ValueError, match="columns"):
        validate_news(df)


def test_rejects_null_in_required_string():
    df = _frame([_row(headline=None)])
    with pytest.raises(ValueError, match="headline"):
        validate_news(df)


def test_allows_null_url_and_body():
    df = _frame([_row(url=pd.NA, body=pd.NA)])
    assert validate_news(df) is df


def test_rejects_tz_naive_timestamps():
    df = _frame([_row(knowable_at=pd.Timestamp("2025-01-02T13:00:00"))])
    with pytest.raises(ValueError, match="knowable_at"):
        validate_news(df)


def test_rejects_knowable_before_published():
    df = _frame([_row(
        published_at=pd.Timestamp("2025-01-02T13:00:00Z"),
        knowable_at=pd.Timestamp("2025-01-02T12:00:00Z"),
    )])
    with pytest.raises(ValueError, match="published_at"):
        validate_news(df)


def test_equal_published_and_knowable_passes():
    t = pd.Timestamp("2025-01-02T13:00:00Z")
    df = _frame([_row(published_at=t, knowable_at=t)])
    assert validate_news(df) is df


def test_rejects_duplicate_key():
    df = _frame([_row(), _row(headline="dup but same key")])
    # same (source, article_id, symbol, knowable_at) -> revision-content inconsistency or key dup
    with pytest.raises(ValueError):
        sorted_df = df.sort_values(
            ["symbol", "source", "article_id", "knowable_at"]
        ).reset_index(drop=True)
        validate_news(sorted_df)


def test_rejects_published_at_varying_within_article():
    df = _frame([
        _row(knowable_at=pd.Timestamp("2025-01-02T13:00:00Z"),
             published_at=pd.Timestamp("2025-01-02T13:00:00Z")),
        _row(knowable_at=pd.Timestamp("2025-01-03T13:00:00Z"),
             published_at=pd.Timestamp("2025-01-02T09:00:00Z"), headline="rev"),
    ])
    df = df.sort_values(["symbol", "source", "article_id", "knowable_at"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="published_at"):
        validate_news(df)


def test_rejects_inconsistent_revision_content():
    t = pd.Timestamp("2025-01-02T13:00:00Z")
    df = _frame([
        _row(symbol="AAPL", knowable_at=t, headline="h1"),
        _row(symbol="MSFT", knowable_at=t, headline="h2"),  # same revision, different headline
    ])
    df = df.sort_values(["symbol", "source", "article_id", "knowable_at"]).reset_index(drop=True)
    with pytest.raises(ValueError, match="headline"):
        validate_news(df)


def test_empty_news_is_contract_shaped():
    e = empty_news()
    assert list(e.columns) == list(NEWS_COLUMNS)
    assert len(e) == 0
    assert str(e["knowable_at"].dtype) == "datetime64[ns, UTC]"
