"""FIX F: validator rejects ±inf in the value column."""
from __future__ import annotations

import pandas as pd
import pytest

from algua.data.fundamentals_schema import to_fundamentals_schema


def _raw_with_value(value):
    return pd.DataFrame(
        [[" aapl", "2025-03-31", "revenue", value, "2025-05-01T13:00:00Z", "vendorX"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def test_positive_inf_rejected():
    with pytest.raises(ValueError, match="inf"):
        to_fundamentals_schema(_raw_with_value(float("inf")))


def test_negative_inf_rejected():
    with pytest.raises(ValueError, match="inf"):
        to_fundamentals_schema(_raw_with_value(float("-inf")))


def test_nan_still_allowed():
    """NaN is explicitly permitted (reported-but-unavailable); must not be rejected."""
    import math
    out = to_fundamentals_schema(_raw_with_value(float("nan")))
    assert len(out) == 1
    assert math.isnan(out["value"].iloc[0])
