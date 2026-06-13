import numpy as np
import pandas as pd
import pytest

from algua.data.news_schema import (
    NEWS_COLUMNS,
    empty_news,
    explode_news_symbols,
    logical_news_hash,
    to_news_schema,
    validate_news,
)


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
        "retracted": False,
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


# ── Task 4: explode_news_symbols ──────────────────────────────────────────────


def _raw_row(**over):
    base = {
        "source": "reuters",
        "article_id": "a1",
        "symbols": "AAPL,MSFT",
        "published_at": "2025-01-02T13:00:00Z",
        "knowable_at": "2025-01-02T13:00:00Z",
        "headline": "two names",
        "url": "http://x/1",
        "body": "b",
    }
    base.update(over)
    return base


def test_explode_comma_string():
    out = explode_news_symbols(pd.DataFrame([_raw_row()]))
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]
    assert set(out["headline"]) == {"two names"}
    assert list(out.columns) == list(NEWS_COLUMNS)


def test_explode_list_form_and_dedup_and_case():
    out = explode_news_symbols(pd.DataFrame([_raw_row(symbols=[" aapl ", "AAPL", "msft"])]))
    assert sorted(out["symbol"]) == ["AAPL", "MSFT"]  # stripped, upper, de-duped within article


def test_explode_adds_missing_optional_columns():
    row = _raw_row()
    del row["url"]
    del row["body"]
    out = explode_news_symbols(pd.DataFrame([row]))
    assert out["url"].isna().all() and out["body"].isna().all()


def test_explode_rejects_zero_symbols():
    with pytest.raises(ValueError, match="symbol"):
        explode_news_symbols(pd.DataFrame([_raw_row(symbols="  ,  ")]))


def test_explode_rejects_missing_required_input_column():
    row = _raw_row()
    del row["headline"]
    with pytest.raises(ValueError, match="missing"):
        explode_news_symbols(pd.DataFrame([row]))


# ── Task 5: to_news_schema + logical_news_hash ───────────────────────────────


def test_to_news_schema_normalizes_and_validates():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(source="Reuters", symbols="aapl")]))
    canon = to_news_schema(raw)
    assert canon["source"].iloc[0] == "reuters"   # source canonicalized (strip+lower)
    assert canon["symbol"].iloc[0] == "AAPL"       # symbol upper
    assert str(canon["knowable_at"].dtype) == "datetime64[ns, UTC]"


def test_to_news_schema_is_idempotent():
    raw = explode_news_symbols(pd.DataFrame([_raw_row()]))
    once = to_news_schema(raw)
    twice = to_news_schema(once)
    assert once.equals(twice)


def test_to_news_schema_requires_knowable_at():
    raw = explode_news_symbols(pd.DataFrame([_raw_row()])).drop(columns=["knowable_at"])
    with pytest.raises(ValueError):
        to_news_schema(raw)


def test_to_news_schema_canonicalizes_null_distinct_from_empty():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(body=None)]))
    canon = to_news_schema(raw)
    assert canon["body"].isna().all()


def test_hash_stable_under_row_order():
    a = to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(symbols="AAPL,MSFT")])))
    b = a.iloc[::-1].reset_index(drop=True)
    assert logical_news_hash(a) == logical_news_hash(to_news_schema(b))


def test_hash_distinguishes_null_empty_and_none_string():
    base = _raw_row()
    h_null = logical_news_hash(
        to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": None}])))
    )
    h_empty = logical_news_hash(
        to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": ""}])))
    )
    h_none = logical_news_hash(
        to_news_schema(explode_news_symbols(pd.DataFrame([{**base, "body": "None"}])))
    )
    assert len({h_null, h_empty, h_none}) == 3


def test_hash_changes_with_headline():
    h1 = logical_news_hash(
        to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(headline="a")])))
    )
    h2 = logical_news_hash(
        to_news_schema(explode_news_symbols(pd.DataFrame([_raw_row(headline="b")])))
    )
    assert h1 != h2


# --- GATE-2 regressions: null/blank required fields, tz robustness ---


def test_to_news_schema_rejects_null_required_field():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(symbols="AAPL")]))
    raw.loc[0, "source"] = None  # null source must NOT become the string "none"
    with pytest.raises(ValueError, match="source"):
        to_news_schema(raw)


def test_to_news_schema_rejects_blank_identity():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(symbols="AAPL")]))
    raw.loc[0, "article_id"] = "   "
    with pytest.raises(ValueError, match="article_id"):
        to_news_schema(raw)


def test_explode_rejects_null_symbols_field():
    # a null `symbols` value must reject (zero symbols), never become the literal symbol "NONE"
    with pytest.raises(ValueError, match="symbol"):
        explode_news_symbols(pd.DataFrame([_raw_row(symbols=None)]))


def test_to_news_schema_rejects_naive_timestamps():
    raw = explode_news_symbols(pd.DataFrame([_raw_row(
        symbols="AAPL", published_at="2025-01-02T13:00:00", knowable_at="2025-01-02T13:00:00")]))
    with pytest.raises(ValueError, match="tz-aware"):
        to_news_schema(raw)


def test_nat_is_treated_as_null_not_literal():
    # pd.NaT in a nullable field canonicalizes to null (not the string "NaT")
    raw = explode_news_symbols(pd.DataFrame([_raw_row(symbols="AAPL", url=pd.NaT)]))
    canon = to_news_schema(raw)
    assert canon["url"].isna().all()
    # pd.NaT as the symbols value rejects as zero-symbol (not the literal symbol "NAT")
    with pytest.raises(ValueError, match="symbol"):
        explode_news_symbols(pd.DataFrame([_raw_row(symbols=pd.NaT)]))


def test_to_news_schema_normalizes_mixed_tz_offsets():
    raw = explode_news_symbols(pd.DataFrame([
        _raw_row(article_id="a1", symbols="AAPL",
                 published_at="2025-01-02T13:00:00Z", knowable_at="2025-01-02T13:00:00Z"),
        _raw_row(article_id="a2", symbols="MSFT",
                 published_at="2025-01-02T08:00:00-05:00", knowable_at="2025-01-02T08:00:00-05:00"),
    ]))
    canon = to_news_schema(raw)
    assert str(canon["knowable_at"].dtype) == "datetime64[ns, UTC]"
    assert (canon["knowable_at"].dt.hour == 13).all()  # the -05:00 row normalizes to 13:00Z


# ── retracted column (issue #132, signal-lane slice) ─────────────────────────


def _raw(source, article_id, symbols, ka, headline="h", pub=None):
    return {
        "source": source, "article_id": article_id, "symbols": symbols,
        "published_at": pub or ka, "knowable_at": ka, "headline": headline,
    }


def test_empty_news_has_retracted_bool_column():
    e = empty_news()
    assert "retracted" in e.columns
    assert e["retracted"].dtype == np.dtype("bool")


def test_explode_emits_retracted_false_for_normal_rows():
    raw = pd.DataFrame([_raw("Reuters", "a1", ["AAPL", "MSFT"], "2023-01-01T00:00:00Z")])
    out = to_news_schema(explode_news_symbols(raw))
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert out["retracted"].dtype == np.dtype("bool")
    assert not out["retracted"].any()


def test_validate_rejects_non_bool_retracted():
    out = to_news_schema(explode_news_symbols(
        pd.DataFrame([_raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z")])))
    bad = out.copy()
    bad["retracted"] = bad["retracted"].astype("object")
    bad.loc[bad.index[0], "retracted"] = "false"
    with pytest.raises(ValueError, match="retracted"):
        validate_news(bad)


def test_hash_changes_with_retracted():
    out = to_news_schema(explode_news_symbols(
        pd.DataFrame([_raw("r", "a1", ["AAPL"], "2023-01-01T00:00:00Z")])))
    flipped = out.copy()
    flipped["retracted"] = True
    assert logical_news_hash(out) != logical_news_hash(flipped)
