from datetime import date

import pandas as pd
import pytest

from algua.data.fundamentals_schema import (
    empty_fundamentals,
    logical_fundamentals_hash,
    to_fundamentals_schema,
    validate_fundamentals,
)


def _raw(rows):
    return pd.DataFrame(rows, columns=[
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ])


def _ok_rows():
    return _raw([
        ["aapl", "2025-03-31", "revenue", 100.0, "2025-05-01T13:00:00Z", "vendorX"],
        ["AAPL", "2025-03-31", "eps", float("nan"), "2025-05-01T13:00:00Z", "vendorX"],
    ])


def test_to_schema_normalizes_and_validates():
    out = to_fundamentals_schema(_ok_rows())
    assert list(out.columns) == [
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ]
    assert (out["symbol"] == "AAPL").all()  # upper-cased
    assert isinstance(out["fiscal_period_end"].iloc[0], date)
    assert str(out["knowable_at"].dt.tz) == "UTC"
    assert str(out["value"].dtype) == "float64"  # NaN preserved
    validate_fundamentals(out)


def test_knowable_at_must_be_ge_fiscal_period_end():
    bad = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-03-30T00:00:00Z", "v"]])
    with pytest.raises(ValueError, match="knowable_at"):
        to_fundamentals_schema(bad)


def test_same_day_filing_is_valid():
    ok = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-03-31T09:00:00Z", "v"]])
    out = to_fundamentals_schema(ok)  # must NOT raise
    assert len(out) == 1


def test_naive_knowable_at_rejected():
    bad = _raw([["AAPL", "2025-03-31", "revenue", 1.0, "2025-05-01T13:00:00", "v"]])
    with pytest.raises(ValueError, match="tz-aware"):
        to_fundamentals_schema(bad)


def test_bitemporal_key_uniqueness_enforced():
    dup = _raw([
        ["AAPL", "2025-03-31", "revenue", 1.0, "2025-05-01T13:00:00Z", "v"],
        ["AAPL", "2025-03-31", "revenue", 2.0, "2025-05-01T13:00:00Z", "v"],
    ])
    with pytest.raises(ValueError, match="unique"):
        to_fundamentals_schema(dup)


def test_empty_is_contract_shaped():
    e = empty_fundamentals()
    assert list(e.columns) == [
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ]
    assert len(e) == 0


def test_logical_hash_is_order_independent():
    a = to_fundamentals_schema(_ok_rows())
    shuffled = _ok_rows().iloc[::-1].reset_index(drop=True)
    b = to_fundamentals_schema(shuffled)
    assert logical_fundamentals_hash(a) == logical_fundamentals_hash(b)
