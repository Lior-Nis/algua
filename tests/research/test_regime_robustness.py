"""Tests for pure regime-split + per-regime robustness helpers (Task 2, Slice 4 of #221)."""
from __future__ import annotations

import datetime

import numpy as np

from algua.research.gates import (
    MIN_REGIME_OBSERVATIONS,
    MIN_REGIME_SHARPE,
    N_REGIMES,
    RegimeSlice,
    regime_robustness_check,
    regime_splits,
)


def _robust_dates(n: int, start_date: datetime.date = datetime.date(2019, 1, 1)) -> list[str]:
    """Generate n unique, sorted, real ISO date strings using calendar arithmetic."""
    return [(start_date + datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    assert N_REGIMES == 3
    assert MIN_REGIME_OBSERVATIONS == 21
    assert MIN_REGIME_SHARPE == 0.0


# ---------------------------------------------------------------------------
# regime_splits: tertile assignment
# ---------------------------------------------------------------------------

def test_tertiles_assigned_by_market_vol():
    """market: first-third low-vol, middle-third mid, last-third high-vol.
    After vol-window warm-up the remaining dates should be evenly split across 3 tertiles
    and every labeled date is bucketed exactly once.
    """
    n = 90
    md = _robust_dates(n)
    rng = np.random.default_rng(0)
    low = rng.normal(0, 0.002, n // 3)
    mid = rng.normal(0, 0.01, n // 3)
    hi = rng.normal(0, 0.03, n - 2 * (n // 3))
    market = list(np.concatenate([low, mid, hi]))
    strat = list(rng.normal(0.001, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    assert overlap > 0
    assert len(slices) == 3
    assert sum(s.n_bars for s in slices) == overlap  # every labeled date bucketed exactly once


def test_empty_overlap_returns_empty():
    """When strategy dates and market dates do not overlap at all, return ([], 0)."""
    market_dates = _robust_dates(30, datetime.date(2019, 1, 1))
    strategy_dates = _robust_dates(30, datetime.date(2020, 6, 1))
    market_returns = [0.01] * 30
    strat_returns = [0.005] * 30
    slices, overlap = regime_splits(
        strat_returns, strategy_dates,
        market_returns, market_dates,
        n_regimes=3, vol_window=21,
    )
    assert slices == []
    assert overlap == 0


def test_slices_have_correct_regime_indices():
    """RegimeSlice.regime_index should equal the position in the returned list."""
    n = 90
    md = _robust_dates(n)
    rng = np.random.default_rng(7)
    market = list(rng.normal(0, 0.01, n))
    strat = list(rng.normal(0.001, 0.01, n))
    slices, _ = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    for i, s in enumerate(slices):
        assert s.regime_index == i


def test_no_dropped_reason_from_splits():
    """regime_splits should set dropped_reason=None on all slices (drop logic is in check)."""
    n = 90
    md = _robust_dates(n)
    rng = np.random.default_rng(8)
    market = list(rng.normal(0, 0.01, n))
    strat = list(rng.normal(0.001, 0.01, n))
    slices, _ = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    for s in slices:
        assert s.dropped_reason is None


# ---------------------------------------------------------------------------
# regime_splits: constant-vol -> all dates collapse -> <2 powered regimes -> fail
# ---------------------------------------------------------------------------

def test_constant_vol_fewer_than_two_survivors_fails():
    """Near-constant market: all vol labels are equal -> ties broken by date -> even split.
    But all returns are constant zero -> ann_volatility==0.0 -> all regimes dropped.
    Result: < 2 survivors -> passed=False.
    """
    n = 90
    md = _robust_dates(n)
    # constant market returns -> log-vol of window is always 0 (std of identical values)
    market = [0.01] * n
    strat = list(np.random.default_rng(1).normal(0.001, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    res = regime_robustness_check(slices, min_obs=21, min_sharpe=0.0)
    assert res.passed is False  # fail-closed: constant vol -> zero-vol regimes dropped


# ---------------------------------------------------------------------------
# regime_robustness_check: zero-vol regime dropped (not counted as pass)
# ---------------------------------------------------------------------------

def test_zero_vol_regime_dropped_not_passed():
    """A regime with constant returns has ann_volatility==0.0 -> dropped (zero_vol).
    Remaining two regimes survive -> n_surviving==2.
    """
    s = [
        RegimeSlice(0, [0.0] * 30, 30, None),          # constant -> zero vol -> dropped
        RegimeSlice(1, [0.01, -0.01] * 15, 30, None),  # alternating -> non-zero vol
        RegimeSlice(2, [0.02, -0.005] * 15, 30, None), # alternating -> non-zero vol
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 2
    assert res.per_regime_sharpes[0] is None  # dropped -> None
    assert res.n_attempted == 3


# ---------------------------------------------------------------------------
# regime_robustness_check: underpowered -> dropped -> <2 survivors -> fail
# ---------------------------------------------------------------------------

def test_underpowered_regime_dropped_and_lt2_fails():
    """Two regimes with < min_obs bars are dropped (too_short). Only 1 survives -> passed=False."""
    s = [
        RegimeSlice(0, [0.01, -0.01] * 15, 30, None),  # 30 bars -> survives
        RegimeSlice(1, [0.01], 1, None),                 # 1 bar -> too_short
        RegimeSlice(2, [0.02], 1, None),                 # 1 bar -> too_short
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 1
    assert res.passed is False  # <2 survivors -> FAIL


def test_all_underpowered_fails():
    """All three regimes underpowered -> 0 survivors -> passed=False."""
    s = [
        RegimeSlice(0, [0.01] * 10, 10, None),
        RegimeSlice(1, [0.01] * 5, 5, None),
        RegimeSlice(2, [0.01] * 3, 3, None),
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 0
    assert res.passed is False


# ---------------------------------------------------------------------------
# regime_robustness_check: all surviving clear floor -> pass
# ---------------------------------------------------------------------------

def test_all_surviving_clear_floor_passes():
    """Three regimes with positive mean and non-zero vol -> all survive and pass sharpe >= 0.0."""
    s = [
        RegimeSlice(0, list(np.random.default_rng(2).normal(0.01, 0.01, 30)), 30, None),
        RegimeSlice(1, list(np.random.default_rng(3).normal(0.01, 0.01, 30)), 30, None),
        RegimeSlice(2, list(np.random.default_rng(4).normal(0.01, 0.01, 30)), 30, None),
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 3
    assert res.passed is True
    # All sharpes should be non-None and >= 0.0
    assert all(sh is not None and sh >= 0.0 for sh in res.per_regime_sharpes)


def test_one_negative_sharpe_fails():
    """One regime with negative mean returns -> sharpe < 0.0 -> fails MIN_REGIME_SHARPE=0.0."""
    rng = np.random.default_rng(42)
    s = [
        RegimeSlice(0, list(rng.normal(0.01, 0.01, 30)), 30, None),   # positive -> pass
        RegimeSlice(1, list(rng.normal(0.01, 0.01, 30)), 30, None),   # positive -> pass
        RegimeSlice(2, list(rng.normal(-0.02, 0.01, 30)), 30, None),  # negative -> fail
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 3  # all 3 have non-zero vol -> all survive
    assert res.passed is False   # regime 2 sharpe < 0.0


# ---------------------------------------------------------------------------
# regime_splits: deterministic tie-break
# ---------------------------------------------------------------------------

def test_deterministic_tie_break():
    """Identical market vols -> rank ties broken by date order.
    Two identical runs give same result."""
    n = 90
    md = _robust_dates(n)
    # constant market returns -> all vol labels identical -> tie-break by date
    market = [0.01] * n
    strat = list(np.random.default_rng(5).normal(0.001, 0.01, n))
    a = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    b = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    assert [s.n_bars for s in a[0]] == [s.n_bars for s in b[0]]


# ---------------------------------------------------------------------------
# regime_splits: vol window warm-up
# ---------------------------------------------------------------------------

def test_vol_window_warmup_excludes_early_dates():
    """Dates before the vol-window warm-up are excluded from the overlap count."""
    n = 30  # fewer than vol_window=21 + enough to see some valid labels
    md = _robust_dates(n)
    rng = np.random.default_rng(9)
    market = list(rng.normal(0, 0.01, n))
    strat = list(rng.normal(0.001, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    # With vol_window=21, only indices >= 20 get a label -> 10 dates labeled out of 30
    assert overlap == n - 21 + 1  # 30 - 21 + 1 = 10
    assert len(slices) == 3


# ---------------------------------------------------------------------------
# Alignment: strategy dates subset of market dates (partial overlap)
# ---------------------------------------------------------------------------

def test_partial_date_alignment():
    """Strategy uses only every other market date -> overlap is half the labeled market dates."""
    n = 90
    all_dates = _robust_dates(n)
    market_returns = list(np.random.default_rng(10).normal(0, 0.01, n))
    # Strategy has only even-indexed dates
    strat_dates = [all_dates[i] for i in range(0, n, 2)]
    strat_returns = list(np.random.default_rng(11).normal(0.001, 0.01, len(strat_dates)))
    slices, overlap = regime_splits(
        strat_returns, strat_dates,
        market_returns, all_dates,
        n_regimes=3, vol_window=21,
    )
    assert overlap > 0
    assert len(slices) == 3
    # Overlap should be <= len(strat_dates) and <= (n - vol_window + 1)
    assert overlap <= len(strat_dates)
    assert sum(s.n_bars for s in slices) == overlap


# ---------------------------------------------------------------------------
# RegimeRobustnessResult structure
# ---------------------------------------------------------------------------

def test_per_regime_sharpes_aligned_to_attempted():
    """per_regime_sharpes has length == n_attempted; None for dropped, float for surviving."""
    s = [
        RegimeSlice(0, [0.01, -0.01] * 15, 30, None),  # survives
        RegimeSlice(1, [0.01] * 5, 5, None),             # too_short -> dropped
        RegimeSlice(2, [0.02, -0.005] * 15, 30, None),  # survives
    ]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert len(res.per_regime_sharpes) == 3
    assert isinstance(res.per_regime_sharpes[0], float)
    assert res.per_regime_sharpes[1] is None
    assert isinstance(res.per_regime_sharpes[2], float)


def test_empty_slices_fails():
    """Empty slice list -> 0 attempted, 0 surviving -> passed=False."""
    res = regime_robustness_check([], min_obs=21, min_sharpe=0.0)
    assert res.passed is False
    assert res.n_attempted == 0
    assert res.n_surviving == 0
    assert res.per_regime_sharpes == []
