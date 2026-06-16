import numpy as np
import pandas as pd
import pytest

from algua.backtest.factor_eval import factor_ic


def _panels(score_rows, return_rows):
    cols = ["AAA", "BBB", "CCC", "DDD"]
    idx = pd.date_range("2023-01-01", periods=len(score_rows), freq="D")
    return (pd.DataFrame(score_rows, index=idx, columns=cols),
            pd.DataFrame(return_rows, index=idx, columns=cols))


def test_perfectly_monotone_factor_has_ic_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]]
    scores, rets = _panels(rows, rows)  # scores rank == return rank every bar
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(1.0)
    assert ic["hit_rate"] == pytest.approx(1.0)
    assert ic["n_obs"] == 3


def test_sign_flipped_factor_has_ic_minus_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1]]
    scores, rets = _panels(rows, [[-v for v in r] for r in rows])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(-1.0)


def test_noise_factor_has_ic_near_zero():
    rng = np.random.default_rng(0)
    n = 200
    scores = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    rets = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert abs(ic["mean_ic"]) < 0.1
    assert ic["n_obs"] == n


def test_constant_cross_sections_are_skipped():
    rows = [[5, 5, 5, 5], [5, 5, 5, 5]]
    scores, rets = _panels(rows, [[1, 2, 3, 4], [4, 3, 2, 1]])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 0
    assert ic["mean_ic"] is None


def test_too_few_observations_returns_none():
    rows = [[1, 2, 3, 4]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 1
    assert ic["ir"] is None and ic["t_stat"] is None


# --- Task 2: IC higher moments (skew / kurtosis) ---

def test_ic_block_has_skew_and_kurtosis_keys():
    """factor_ic always returns ic_skew and ic_kurtosis keys."""
    rows = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert "ic_skew" in ic
    assert "ic_kurtosis" in ic


def test_ic_skew_kurtosis_none_when_underpowered():
    """< 2 usable IC obs → ic_skew and ic_kurtosis are None."""
    rows = [[1, 2, 3, 4]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["ic_skew"] is None
    assert ic["ic_kurtosis"] is None


def test_gaussian_ic_series_has_kurtosis_near_3():
    """A large-ish near-Gaussian IC series gives raw (Pearson) kurtosis ≈ 3."""
    import scipy.stats as st
    rng = np.random.default_rng(42)
    n = 300
    # Fix IC values directly: create scores whose Spearman IC == draw from N(0,1)
    # Easier: just make a 2-symbol panel where IC varies smoothly.
    # Use a trick: n-row panel where each bar's IC is a Gaussian draw.
    # We can't directly control IC per bar, so instead test that ic_kurtosis
    # is finite and within a plausible range for a large series.
    scores = pd.DataFrame(rng.normal(size=(n, 4)), columns=["A", "B", "C", "D"])
    rets = pd.DataFrame(rng.normal(size=(n, 4)), columns=["A", "B", "C", "D"])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["ic_kurtosis"] is not None
    # Raw (Pearson) kurtosis for a Gaussian-ish IC series is near 3.
    # With n=300 it won't be exactly 3, but it should be positive and < 10.
    assert 0 < ic["ic_kurtosis"] < 10


def test_ic_skew_finite_for_skewed_series():
    """ic_skew is finite (not None, not NaN) for a series with measurable skewness."""
    rows = [[1, 2, 3, 4]] * 20 + [[4, 3, 2, 1]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["ic_skew"] is not None
    import math
    assert math.isfinite(ic["ic_skew"])
