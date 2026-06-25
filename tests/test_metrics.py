import numpy as np
import pandas as pd

from algua.backtest.metrics import metrics_from_returns


def test_metrics_include_skew_and_raw_kurtosis():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, size=2000))
    m = metrics_from_returns(r)
    assert "skewness" in m and "kurtosis" in m
    # raw (Pearson) kurtosis: ~3 for a normal series, NOT ~0 (excess)
    assert abs(m["skewness"]) < 0.25
    assert 2.5 < m["kurtosis"] < 3.5


def test_metrics_moments_finite_on_degenerate_input():
    # empty, single-element, and constant series must never inject NaN/inf
    for r in (pd.Series([], dtype=float), pd.Series([0.01]), pd.Series([0.01, 0.01, 0.01])):
        m = metrics_from_returns(r)
        assert m["skewness"] == 0.0 and m["kurtosis"] == 0.0
        assert all(np.isfinite(v) for v in m.values())
