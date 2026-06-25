import json

import numpy as np
import pandas as pd
import pytest

from algua.backtest.result import BacktestResult, series_frame


def _result(returns):
    return BacktestResult(
        strategy="mom", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-03"},
        seed=0, snapshot_id=None, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def test_series_frame_columns_and_iso_dates():
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    frame, meta = series_frame(_result(pd.Series([0.01, -0.0, 0.02], index=idx)))
    assert list(frame.columns) == ["date", "ret"]
    assert frame["date"].tolist() == [
        "2023-01-01T00:00:00", "2023-01-02T00:00:00", "2023-01-03T00:00:00"]
    assert frame["ret"].tolist() == [0.01, 0.0, 0.02]


def test_series_frame_canonicalizes_negative_zero():
    idx = pd.to_datetime(["2023-01-01"])
    frame, _ = series_frame(_result(pd.Series([-0.0], index=idx)))
    assert not np.signbit(frame["ret"].to_numpy()[0])  # +0.0, not -0.0


def test_series_frame_metadata_carries_full_identity_minus_returns():
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    _, meta = series_frame(_result(pd.Series([0.01, 0.02], index=idx)))
    payload = json.loads(meta["algua.result_json"])
    assert payload["strategy"] == "mom"
    assert payload["config_hash"] == "cfg"
    assert "returns" not in payload  # to_dict already excludes it


def test_series_frame_raises_on_none_returns():
    """series_frame must raise ValueError (not AssertionError) when returns is None (finding #2)."""
    with pytest.raises(ValueError, match="series_frame requires non-None returns"):
        series_frame(_result(None))
