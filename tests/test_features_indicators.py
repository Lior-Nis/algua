import numpy as np
import pandas as pd
import pytest

from algua.features.indicators import momentum, zscore


def test_momentum_is_trailing_return():
    s = pd.Series([10.0, 11.0, 12.0, 13.2])
    # momentum over lookback=3 at the last point: 13.2/10 - 1 = 0.32
    assert momentum(s, lookback=3).iloc[-1] == pytest.approx(0.32)


def test_momentum_insufficient_history_is_nan():
    s = pd.Series([10.0, 11.0])
    assert np.isnan(momentum(s, lookback=3).iloc[-1])


def test_zscore_centers_and_scales():
    s = pd.Series([1.0, 2.0, 3.0])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert z.iloc[-1] > 0 and z.iloc[0] < 0


def test_zscore_zero_variance_is_nan():
    """A degenerate (flat) cross-section is undefined, not a flat zero tie."""
    s = pd.Series([5.0, 5.0, 5.0])
    z = zscore(s)
    assert z.isna().all()


def test_zscore_zero_variance_drops_out():
    """NaN lets downstream `.dropna()` discard a degenerate cross-section."""
    s = pd.Series([5.0, 5.0, 5.0])
    assert zscore(s).dropna().empty
