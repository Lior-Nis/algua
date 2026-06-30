import math

import numpy as np
import pandas as pd

from algua.backtest._constants import ANN
from algua.backtest.metrics import _max_drawdown, metrics_from_returns


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


# ---- Dashboard metrics (#348): Calmar / Sortino / tail ratio / hit rate ----------------

DASHBOARD_KEYS = ("hit_rate", "tail_ratio", "sortino", "cagr", "calmar")


def test_dashboard_keys_present_and_legacy_unchanged():
    # The new keys are additive: every pre-#348 key keeps its exact value.
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, size=500))
    m = metrics_from_returns(r)
    for k in DASHBOARD_KEYS:
        assert k in m
    # Legacy values recomputed independently must be identical.
    assert m["total_return"] == float((1.0 + r).prod() - 1.0)
    assert m["ann_return"] == float(r.mean() * ANN)
    assert m["sharpe"] == float((r.mean() * ANN) / (r.std() * np.sqrt(ANN)))


def test_hit_rate_strict_zero_is_a_miss():
    r = pd.Series([0.01, -0.01, 0.0, 0.02])  # 2 of 4 strictly positive; the 0.0 is a miss
    assert metrics_from_returns(r)["hit_rate"] == 0.5
    assert 0.0 <= metrics_from_returns(r)["hit_rate"] <= 1.0


def test_tail_ratio_symmetric_is_about_one_and_sentinel_on_no_left_tail():
    rng = np.random.default_rng(2)
    sym = pd.Series(rng.normal(0.0, 0.01, size=5000))
    tr = metrics_from_returns(sym)["tail_ratio"]
    assert 0.8 < tr < 1.25  # symmetric distribution -> tails roughly balanced
    # A zero left tail (p5 == 0, here a non-negative series with zeros at the bottom) ->
    # the denominator vanishes -> finite sentinel 0.0.
    nonneg = pd.Series([0.0, 0.0, 0.0, 0.01, 0.02, 0.03])
    assert np.percentile(nonneg.to_numpy(), 5) == 0.0  # precondition for the sentinel
    assert metrics_from_returns(nonneg)["tail_ratio"] == 0.0


def test_tail_ratio_matches_percentile_definition():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0003, 0.012, size=400))
    p95, p5 = np.percentile(r.to_numpy(), [95, 5])
    assert metrics_from_returns(r)["tail_ratio"] == float(p95) / abs(float(p5))


def test_sortino_exact_formula_and_exceeds_sharpe_on_left_skew():
    # Left-skewed-but-net-positive series: downside dev < total vol -> Sortino > Sharpe.
    r = pd.Series([0.02, 0.02, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02])
    m = metrics_from_returns(r)
    excess = float(r.mean() * ANN)
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2))) * np.sqrt(ANN)
    assert math.isclose(m["sortino"], excess / dd, rel_tol=1e-12)
    assert m["sortino"] > m["sharpe"] > 0.0


def test_sortino_sentinel_when_no_downside():
    r = pd.Series([0.01, 0.02, 0.0, 0.03])  # no strictly-negative return -> dd == 0
    assert metrics_from_returns(r)["sortino"] == 0.0


def test_calmar_uses_negative_drawdown_convention():
    r = pd.Series([0.10, -0.20, 0.05, 0.05, 0.10])
    m = metrics_from_returns(r)
    mdd = _max_drawdown(r)
    assert mdd < 0.0  # drawdown is the NEGATIVE convention
    assert math.isclose(m["calmar"], m["cagr"] / abs(mdd), rel_tol=1e-12)


def test_cagr_guards_total_loss_base():
    # A >= -100% cumulative return makes the compounding base <= 0 -> sentinel 0.0, finite.
    r = pd.Series([-1.0, 0.5, 0.5])  # 1 + total_return == 0
    m = metrics_from_returns(r)
    assert m["cagr"] == 0.0
    assert m["calmar"] == 0.0 or np.isfinite(m["calmar"])
    assert all(np.isfinite(v) for v in m.values())


def test_dashboard_finite_on_degenerate_input():
    for r in (pd.Series([], dtype=float), pd.Series([0.01]), pd.Series([0.01, 0.01, 0.01])):
        m = metrics_from_returns(r)
        for k in DASHBOARD_KEYS:
            assert k in m and np.isfinite(m[k])
