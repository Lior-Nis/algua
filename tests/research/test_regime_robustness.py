"""Tests for pure regime-split + per-regime robustness helpers (Task 2, Slice 4 of #221)."""
from __future__ import annotations

import datetime
import math

import numpy as np

from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    MIN_REGIME_OBSERVATIONS,
    MIN_REGIME_SHARPE,
    N_REGIMES,
    GateCriteria,
    RegimeSlice,
    evaluate_gate,
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
    """CONSTANT market vol + uniformly POSITIVE strategy -> fails via n_surviving < 2.

    Mechanism (value-based tertiles):
    - Market: all returns identical -> all vol labels == 0 -> t1 == t2 == 0
    - vol <= t1 is True for ALL dates -> ALL dates land in regime 0
    - Regimes 1 and 2 are EMPTY (n_bars == 0)
    - regime_robustness_check drops empty regimes (n_bars < min_obs=21 -> too_short)
    - 1 survivor (or 0 if regime 0 is also dropped) -> n_surviving < 2 -> passed=False

    Strategy returns are uniformly positive with variance so they are NOT zero-vol.
    This proves the gate fails via the SURVIVOR COUNT path (< 2 non-empty regimes),
    not via zero_vol dropping after an equal-count split.
    """
    n = 90
    md = _robust_dates(n)
    # Constant market returns: all vol labels equal 0 -> ALL dates collapse into regime 0
    market = [0.01] * n
    # Uniformly positive strategy with variance (not zero-vol) to distinguish failure path:
    # if the strategy were zero-vol, zero_vol drop would trigger first on regime 0.
    # We want: regime 0 has positive non-zero-vol strategy, regimes 1 & 2 are empty (n_bars=0)
    # -> regimes 1 & 2 dropped (too_short) -> n_surviving < 2 -> passed=False.
    strat = list(np.random.default_rng(1).normal(0.005, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    # With value-based tertiles: regime 0 has all overlap bars; regimes 1 & 2 are empty
    assert slices[0].n_bars == overlap  # ALL labeled dates collapse into regime 0
    assert slices[1].n_bars == 0        # regime 1: EMPTY
    assert slices[2].n_bars == 0        # regime 2: EMPTY
    res = regime_robustness_check(slices, min_obs=21, min_sharpe=0.0)
    # Fails via the survivor count path (< 2 non-empty regimes), not zero_vol dropping
    assert res.n_surviving < 2
    assert res.passed is False


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


# ---------------------------------------------------------------------------
# Task 3: evaluate_gate wiring — regime_robustness AND-check + audit
# ---------------------------------------------------------------------------

_HOLDOUT = {
    "sharpe": 7.0,
    "total_return": 0.2,
    "n_bars": 252,
    "skewness": 0.0,
    "kurtosis": 3.0,
}
_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _make_wf(
    *,
    holdout_returns: tuple[list[float], list[str]] | None = None,
    # market_returns: stored on wf.market_returns, but evaluate_gate reads from
    # its own market_returns kwarg (passed separately at each call site — not from wf).
    market_returns: tuple[list[float], list[str]] | None = None,
    sharpe: float = 7.0,
) -> WalkForwardResult:
    """Build a WalkForwardResult that passes all non-regime gate checks.
    Optionally carry holdout_returns and market_returns for regime wiring tests.
    Note: market_returns here populates wf.market_returns (unused by evaluate_gate itself —
    evaluate_gate reads from its own market_returns kwarg, passed separately at call sites).
    """
    return WalkForwardResult(
        strategy="test_strat",
        config_hash="c",
        data_source="synthetic",
        snapshot_id=None,
        timeframe="1d",
        seed=None,
        period={"start": "2020-01-01", "end": "2021-01-01"},
        windows=4,
        holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_HOLDOUT, "sharpe": sharpe},
        stability=dict(_STAB),
        holdout_returns=holdout_returns,
        market_returns=market_returns,
    )


def _make_regime_returns(
    n: int = 252,
    seed: int = 0,
    positive_mean: float = 0.005,
    start_date: datetime.date = datetime.date(2020, 1, 1),
) -> tuple[list[float], list[str]]:
    """Build a return series with n positive-mean bars and ISO date strings."""
    rng = np.random.default_rng(seed)
    rets = list(rng.normal(positive_mean, 0.01, n))
    dates = _robust_dates(n, start_date)
    return (rets, dates)


# --- binding: regime_robustness check APPENDED + audit populated ----------

def test_regime_check_appended_when_binding():
    """With holdout_returns + market_returns + sufficient overlap -> regime_robustness present."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    names = [c["name"] for c in d.checks]
    assert "regime_robustness" in names


def test_regime_check_method_vol_tertile_when_binding():
    """When the check is binding, regime_method must be 'vol_tertile'."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    assert d.regime_method == "vol_tertile"
    assert d.regime_robustness_binding is True


def test_regime_audit_fields_populated_when_binding():
    """n_regimes_attempted, n_regimes_surviving, per_regime_sharpes set when binding."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    assert d.n_regimes_attempted is not None and d.n_regimes_attempted > 0
    assert d.n_regimes_surviving is not None
    assert d.per_regime_sharpes is not None and len(d.per_regime_sharpes) > 0


def test_regime_check_in_to_dict():
    """All five regime audit fields appear in to_dict()."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    dct = d.to_dict()
    assert "regime_method" in dct
    assert "n_regimes_attempted" in dct
    assert "n_regimes_surviving" in dct
    assert "per_regime_sharpes" in dct
    assert "regime_robustness_binding" in dct


# --- FAIL-CLOSED: binding + <2 survivors -> check FAILED -> gate passed=False ------

def test_regime_binding_negative_sharpe_regime_fails_gate():
    """Strategy has strongly negative returns in the high-vol regime -> regime_robustness FAILED.
    A strategy that passes the aggregate holdout_sharpe (sharpe=7.0) must fail overall.

    Mechanism:
    - Market: first 63 bars low-vol, last 189 bars high-vol (clear tertile split).
    - vol_window=21 -> from bar 20 onward all dates get a vol label.
    - Strategy: good positive returns for bars 0..188, strong negative returns for bars 189..251
      (the high-vol regime) -> sharpe in high-vol regime < 0.0 -> regime check FAILS.
    """
    n = 252
    dates = _robust_dates(n)
    rng = np.random.default_rng(77)
    # Market: first third calm, then two thirds volatile
    market_low = list(rng.normal(0, 0.001, 84))
    market_high = list(rng.normal(0, 0.03, n - 84))
    market_rets = market_low + market_high
    # Strategy: very positive first 84+84=168 bars, very negative last 84 bars
    strat_good = list(rng.normal(0.02, 0.01, 168))
    strat_bad = list(rng.normal(-0.05, 0.01, n - 168))
    strat_rets = strat_good + strat_bad

    market = (market_rets, dates)
    wf = _make_wf(holdout_returns=(strat_rets, dates), market_returns=market, sharpe=7.0)
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=market)
    regime_check = next(c for c in d.checks if c["name"] == "regime_robustness")
    assert regime_check["passed"] is False
    assert d.passed is False


# --- market_returns=None -> NO check, regime_method="unavailable" ----------

def test_no_market_returns_omits_check():
    """market_returns=None -> no regime_robustness check in checks list."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    names = [c["name"] for c in d.checks]
    assert "regime_robustness" not in names


def test_no_market_returns_method_unavailable():
    """market_returns=None -> regime_method='unavailable', binding=False."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    assert d.regime_method == "unavailable"
    assert d.regime_robustness_binding is False
    assert d.n_regimes_attempted is None
    assert d.n_regimes_surviving is None
    assert d.per_regime_sharpes is None


def test_no_holdout_returns_omits_check():
    """wf.holdout_returns=None -> no regime check, regime_method='unavailable'."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=None)  # no holdout returns
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    names = [c["name"] for c in d.checks]
    assert "regime_robustness" not in names
    assert d.regime_method == "unavailable"


# --- insufficient overlap -> omit + regime_method="insufficient_overlap" ---

def test_insufficient_overlap_omits_check():
    """When overlap_n < MIN_REGIME_OVERLAP_BARS -> no check, method='insufficient_overlap'."""
    # strategy dates and market dates partially overlap but < MIN_REGIME_OVERLAP_BARS
    market_dates = _robust_dates(40, datetime.date(2020, 1, 1))  # only 40 market dates
    # Same dates -> overlap but vol_window=21 -> at most 20 labeled -> < 63
    strat_dates = _robust_dates(40, datetime.date(2020, 1, 1))
    # vol_window=21 -> at most 40-21+1=20 labeled -> < MIN_REGIME_OVERLAP_BARS (63)
    market = ([0.005] * 40, market_dates)
    strat_rets = list(np.random.default_rng(50).normal(0.005, 0.01, 40))
    wf = _make_wf(holdout_returns=(strat_rets, strat_dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=market)
    names = [c["name"] for c in d.checks]
    assert "regime_robustness" not in names
    assert d.regime_method == "insufficient_overlap"
    assert d.regime_robustness_binding is False


# --- tighten-only property: new.passed => old.passed ----------------------

def test_tighten_only_passing_wf():
    """Tighten-only: if the regime check passes, the overall gate outcome is the SAME as
    the no-regime baseline (the regime check can only FAIL gates that would otherwise pass,
    never PASS gates that would otherwise fail).
    """
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates))
    # without market_returns -> baseline
    d_old = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    # with market_returns -> same wf, regime check added
    d_new = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    # If new passes, old must also pass (new.passed => old.passed)
    if d_new.passed:
        assert d_old.passed


def test_tighten_only_cannot_repair_failing_gate():
    """A gate that fails existing checks still fails even when regime check passes.
    (Demonstrates regime check is AND — cannot rescue a failing gate.)
    """
    rets, dates = _make_regime_returns(n=252)
    # Low sharpe so holdout_sharpe check FAILS
    wf = _make_wf(holdout_returns=(rets, dates), sharpe=-5.0)
    d_no_regime = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    d_with_regime = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    assert d_no_regime.passed is False
    assert d_with_regime.passed is False  # regime passing cannot rescue


# --- per_regime_sharpes null-coercion in to_dict --------------------------

def test_per_regime_sharpes_null_coerced_in_to_dict():
    """to_dict() should null-coerce non-finite per_regime_sharpes entries."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(rets, dates))
    dct = d.to_dict()
    if dct["per_regime_sharpes"] is not None:
        for v in dct["per_regime_sharpes"]:
            assert v is None or (isinstance(v, float) and math.isfinite(v))


# --- binding independence from dsr_binding --------------------------------

def test_regime_binding_independent_of_dsr_binding():
    """Regime check should bind independently of dsr_binding flag."""
    rets, dates = _make_regime_returns(n=252)
    wf = _make_wf(holdout_returns=(rets, dates), market_returns=(rets, dates))
    # Without dsr_binding
    d = evaluate_gate(wf, GateCriteria(), pit_ok=True,
                      dsr_binding=False, market_returns=(rets, dates))
    assert d.regime_robustness_binding is True  # regime is independent of DSR


# ---------------------------------------------------------------------------
# Finding 1 (additional): two distinct vol levels -> at most 2 non-empty regimes
# ---------------------------------------------------------------------------

def test_value_based_tertiles_collapse_a_clustered_vol_distribution():
    """Value-based tertiles (not equal-count rank split) must let one vol cluster absorb MORE
    than a third of the dates. A constant-return low block gives many identical (zero) rolling-vol
    labels; a noisy block gives higher vols. With value-based thresholds, all the zero-vol dates
    fall into regime 0 — so regime 0 holds well over overlap/3 bars, which an equal-count
    ``np.array_split`` could never produce (it caps each regime near overlap/3). This is the
    discriminating property behind the constant-vol fail-close."""
    vol_window = 21
    n = 60
    md = _robust_dates(n)
    rng = np.random.default_rng(42)
    # First 40 bars: constant return -> every fully-contained 21-window has std 0 -> vol 0.
    # Last 20 bars: noisy -> higher vol. Boundary windows mix.
    market = [0.001] * 40 + list(rng.normal(0.0, 0.03, 20))
    # strategy is NOT constant so its regimes are not zero-vol-dropped
    strat = list(rng.normal(0.005, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=vol_window)
    assert overlap > 0
    assert len(slices) == 3
    # The lowest-vol regime absorbs the big zero-vol cluster: strictly more than an equal split.
    assert slices[0].n_bars > overlap / 3
    # Every produced vol-derived assignment is finite-driven (no NaN leaked): bars sum to overlap.
    assert sum(s.n_bars for s in slices) == overlap


# ---------------------------------------------------------------------------
# Finding 4: NaN in vol window -> date excluded from tertile assignment
# ---------------------------------------------------------------------------

def test_nan_in_vol_window_excludes_date():
    """A market series with a NaN in a window -> that date is excluded from overlap.

    When 1 + r <= 0 (e.g. r = -1.0, a total loss), the log is undefined, so the ENTIRE
    date's vol label is skipped. That date must not appear in any regime or in overlap_n.
    """
    n = 40
    md = _robust_dates(n)
    vol_window = 21
    rng = np.random.default_rng(7)
    # Normal market returns except one extreme value that makes the log undefined
    market = list(rng.normal(0, 0.01, n))
    # Inject a return of -1.0 at position 10 -> for ALL windows containing position 10
    # (indices 10 through 10 + vol_window - 1 = 30), the log is undefined -> those dates
    # have no vol label and are excluded. This is the existing guard, confirmed by test.
    market[10] = -1.0  # 1 + (-1.0) = 0.0 <= 0.0 -> log guard triggers
    strat = list(rng.normal(0.005, 0.01, n))
    slices_with_nan, overlap_with_nan = regime_splits(
        strat, md, market, md, n_regimes=3, vol_window=vol_window
    )
    # Compare to a clean market (no NaN window)
    market_clean = list(rng.normal(0, 0.01, n))
    slices_clean, overlap_clean = regime_splits(
        strat, md, market_clean, md, n_regimes=3, vol_window=vol_window
    )
    # The NaN-affected market must exclude at least the windows containing index 10:
    # windows i=10..30 (21 windows) are poisoned -> overlap_with_nan < overlap_clean
    assert overlap_with_nan < overlap_clean
    # And the excluded date must not be counted in overlap or in any regime
    assert sum(s.n_bars for s in slices_with_nan) == overlap_with_nan


def test_genuine_nonfinite_market_return_excluded():
    """A genuine NaN/inf market return (NOT just 1+r<=0) must produce NO vol label — `1+r<=0`
    is False for NaN, so the finiteness guard must catch it; otherwise a NaN vol label would
    count toward overlap and poison the quantile thresholds."""
    n = 40
    md = _robust_dates(n)
    vol_window = 21
    rng = np.random.default_rng(11)
    strat = list(rng.normal(0.005, 0.01, n))
    clean = list(rng.normal(0, 0.01, n))
    _, overlap_clean = regime_splits(strat, md, clean, md, n_regimes=3, vol_window=vol_window)
    for bad in (float("nan"), float("inf"), float("-inf")):
        market = list(rng.normal(0, 0.01, n))
        market[10] = bad
        slices_bad, overlap_bad = regime_splits(
            strat, md, market, md, n_regimes=3, vol_window=vol_window)
        # the poisoned windows (those containing index 10) must be excluded -> fewer labels
        assert overlap_bad < overlap_clean, f"{bad!r} not excluded from overlap"
        # every per-regime sharpe / label that remains is finite (no NaN leaked through)
        assert sum(s.n_bars for s in slices_bad) == overlap_bad
