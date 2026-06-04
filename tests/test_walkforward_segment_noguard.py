"""Tests for #50: _segment_record dead code removed — start/end always use direct index access."""
import pandas as pd

from algua.backtest.walkforward import _segment_record


def _daily_returns(n: int) -> pd.Series:
    import pandas as pd
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series([0.01] * n, index=dates)


def test_segment_record_non_empty_has_start_end():
    returns = _daily_returns(10)
    rec = _segment_record(returns, 0, 10)
    assert rec["start"] is not None
    assert rec["end"] is not None
    assert rec["n_bars"] == 10


def test_segment_record_start_end_are_date_strings():
    returns = _daily_returns(5)
    rec = _segment_record(returns, 0, 5)
    # Must be strings in "YYYY-MM-DD" format, not None
    assert isinstance(rec["start"], str) and len(rec["start"]) == 10
    assert isinstance(rec["end"], str) and len(rec["end"]) == 10
