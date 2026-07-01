"""Tests for the market-beta / idiosyncratic-alpha screen (#328).

The promotion gate compares RAW Sharpe (risk_free=0, no benchmark subtraction), so a persistently
net-long or LEVERED market-beta book in a bull market posts a high raw Sharpe with ~zero true alpha.
The `idiosyncratic_alpha` AND-check regresses the holdout returns on the PIT market benchmark and
requires the annualized appraisal ratio (residual alpha / residual vol) to clear the floor.
"""
from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd

from algua.backtest.metrics import metrics_from_returns
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    IR_MIN_APPRAISAL_RATIO,
    IR_MIN_OVERLAP_BARS,
    IR_MIN_VOL,
    MIN_HOLDOUT_OBSERVATIONS,
    MIN_REGIME_VOL,
    GateCriteria,
    evaluate_gate,
    information_ratio,
)


def _dates(n: int, start: datetime.date = datetime.date(2020, 1, 1)) -> list[str]:
    return [(start + datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    assert IR_MIN_OVERLAP_BARS == MIN_HOLDOUT_OBSERVATIONS == 63
    assert IR_MIN_APPRAISAL_RATIO == 0.3
    assert IR_MIN_VOL == MIN_REGIME_VOL == 1e-9


# ---------------------------------------------------------------------------
# Pure function: information_ratio
# ---------------------------------------------------------------------------

def test_market_neutral_alpha_passes():
    """beta ~ 0, steady idiosyncratic alpha -> high appraisal ratio, not degenerate, >= floor."""
    n = 252
    d = _dates(n)
    rng = np.random.default_rng(1)
    market = list(rng.normal(0.0004, 0.01, n))          # zero-ish-mean market
    # strategy: uncorrelated positive drift + independent noise (genuine alpha, ~no beta)
    strat = list(0.001 + rng.normal(0.0, 0.008, n))
    res = information_ratio(strat, d, market, d)
    assert res.degenerate is False
    assert res.overlap_n == n
    assert res.market_beta is not None and abs(res.market_beta) < 0.5   # near market-neutral
    assert res.appraisal_ratio is not None and res.appraisal_ratio >= IR_MIN_APPRAISAL_RATIO


def test_levered_market_beta_fails():
    """The issue's exact scenario: strat = 2*market in a BULL market. High raw Sharpe, ~zero alpha
    once beta is netted out -> appraisal ratio ~0 -> BELOW the floor (would fail the gate)."""
    n = 252
    d = _dates(n)
    rng = np.random.default_rng(2)
    market = list(rng.normal(0.0006, 0.01, n))          # secular bull market (positive mean)
    # zero-mean idiosyncratic component => NO true alpha, only levered market beta
    raw_noise = rng.normal(0.0, 0.006, n)
    noise = raw_noise - raw_noise.mean()
    strat = [2.0 * m + z for m, z in zip(market, noise, strict=True)]
    res = information_ratio(strat, d, market, d)
    assert res.degenerate is False
    assert res.market_beta is not None and abs(res.market_beta - 2.0) < 0.2   # recovers the beta
    # The RAW Sharpe of the levered-beta book is high (the beta-as-alpha trap)...
    raw_sharpe = metrics_from_returns(pd.Series(strat))["sharpe"]
    assert raw_sharpe > 0.5
    # ...but its idiosyncratic appraisal ratio is far below the floor -> the gate would fail it.
    assert res.appraisal_ratio is not None
    assert res.appraisal_ratio < IR_MIN_APPRAISAL_RATIO


def test_constant_market_is_degenerate():
    """A (near-)constant market series -> beta undefined -> degenerate (fail closed)."""
    n = 100
    d = _dates(n)
    strat = list(np.random.default_rng(3).normal(0.001, 0.01, n))
    market = [0.001] * n            # constant -> zero variance
    res = information_ratio(strat, d, market, d)
    assert res.degenerate is True
    assert res.market_beta is None and res.appraisal_ratio is None


def test_zero_residual_is_degenerate():
    """A strategy PERFECTLY explained by the market (zero residual) carries no measurable
    idiosyncratic alpha -> degenerate (fail closed), but beta/alpha are still surfaced for audit."""
    n = 100
    d = _dates(n)
    market = list(np.random.default_rng(4).normal(0.0005, 0.01, n))
    strat = [0.5 * m for m in market]      # exact linear function of market -> residual == 0
    res = information_ratio(strat, d, market, d)
    assert res.degenerate is True
    assert res.residual_vol_ann is None and res.appraisal_ratio is None
    assert res.market_beta is not None and abs(res.market_beta - 0.5) < 1e-6


def test_non_finite_market_is_degenerate():
    n = 100
    d = _dates(n)
    strat = list(np.random.default_rng(5).normal(0.001, 0.01, n))
    market = list(np.random.default_rng(6).normal(0.0, 0.01, n))
    market[10] = float("nan")
    res = information_ratio(strat, d, market, d)
    assert res.degenerate is True


def test_empty_overlap():
    strat = [0.01] * 30
    market = [0.005] * 30
    res = information_ratio(strat, _dates(30, datetime.date(2020, 1, 1)),
                            market, _dates(30, datetime.date(2021, 6, 1)))
    assert res.overlap_n == 0
    assert res.degenerate is False   # nothing to test -> the caller omits (insufficient_overlap)


def test_ragged_input_raises():
    """A ragged (returns, dates) leg is corrupt -> fail LOUD, never silently truncate to omit."""
    import pytest
    d = _dates(64)
    strat = [0.001] * 63          # one short of its dates
    market = list(np.random.default_rng(8).normal(0.0, 0.01, 64))
    with pytest.raises(ValueError):
        information_ratio(strat, d, market, d)


def test_duplicate_dates_raise():
    """Duplicate dates within a leg would dict-collapse to a silent subset -> fail LOUD instead."""
    import pytest
    d = _dates(64)
    d_dup = list(d)
    d_dup[5] = d_dup[4]           # inject a duplicate date -> 63 unique of 64
    rng = np.random.default_rng(9)
    strat = list(rng.normal(0.001, 0.01, 64))
    market = list(rng.normal(0.0, 0.01, 64))
    with pytest.raises(ValueError):
        information_ratio(strat, d_dup, market, d)


def test_appraisal_equals_alpha_over_residual_vol():
    """appraisal_ratio must equal alpha_ann / residual_vol_ann (annualization consistency)."""
    n = 200
    d = _dates(n)
    rng = np.random.default_rng(7)
    market = list(rng.normal(0.0003, 0.01, n))
    strat = list(0.0008 + 0.3 * np.array(market) + rng.normal(0.0, 0.006, n))
    res = information_ratio(strat, d, market, d)
    assert res.appraisal_ratio is not None
    assert res.alpha_ann is not None and res.residual_vol_ann is not None
    assert math.isclose(res.appraisal_ratio, res.alpha_ann / res.residual_vol_ann, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# evaluate_gate wiring
# ---------------------------------------------------------------------------

_HOLDOUT = {"sharpe": 7.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0}
_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _make_wf(*, holdout_returns=None, sharpe: float = 7.0) -> WalkForwardResult:
    return WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2021-01-01"},
        windows=4, holdout_frac=0.2, window_metrics=[],
        holdout_metrics={**_HOLDOUT, "sharpe": sharpe}, stability=dict(_STAB),
        holdout_returns=holdout_returns, market_returns=None,
    )


def _alpha_series(n: int = 252, seed: int = 10):
    d = _dates(n)
    rng = np.random.default_rng(seed)
    market = list(rng.normal(0.0004, 0.01, n))
    strat = list(0.0012 + rng.normal(0.0, 0.008, n))     # genuine market-neutral alpha
    return strat, market, d


def test_check_appended_and_binding_when_market_present():
    strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    names = [c["name"] for c in dec.checks]
    assert "idiosyncratic_alpha" in names
    assert dec.ir_binding is True
    assert dec.ir_method == "capm_appraisal"
    assert dec.ir_overlap_n == 252
    assert dec.appraisal_ratio is not None and dec.market_beta is not None


def test_genuine_alpha_passes_gate():
    strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    ir_check = next(c for c in dec.checks if c["name"] == "idiosyncratic_alpha")
    assert ir_check["passed"] is True
    assert dec.passed is True


def test_levered_beta_blocks_gate():
    """A high-raw-Sharpe levered-beta book (passes holdout_sharpe=7.0) is BLOCKED by the check."""
    n = 252
    d = _dates(n)
    rng = np.random.default_rng(11)
    market = list(rng.normal(0.0006, 0.01, n))
    raw_noise = rng.normal(0.0, 0.006, n)               # zero-mean idiosyncratic => no true alpha
    noise = raw_noise - raw_noise.mean()
    strat = [2.0 * m + z for m, z in zip(market, noise, strict=True)]
    wf = _make_wf(holdout_returns=(strat, d), sharpe=7.0)
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    ir_check = next(c for c in dec.checks if c["name"] == "idiosyncratic_alpha")
    assert ir_check["passed"] is False
    assert dec.passed is False


def test_degenerate_market_fails_closed_when_binding():
    """Armed (>=63 overlap) but constant market -> check present and FAILED (fail closed)."""
    n = 100
    d = _dates(n)
    strat = list(np.random.default_rng(12).normal(0.001, 0.01, n))
    market = [0.001] * n
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    ir_check = next(c for c in dec.checks if c["name"] == "idiosyncratic_alpha")
    assert ir_check["passed"] is False
    assert dec.ir_binding is True
    assert dec.passed is False


def test_no_market_returns_omits_check():
    strat, _market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    assert all(c["name"] != "idiosyncratic_alpha" for c in dec.checks)
    assert dec.ir_binding is False
    assert dec.ir_method == "unavailable"


def test_no_holdout_returns_omits_check():
    _strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=None)
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    assert all(c["name"] != "idiosyncratic_alpha" for c in dec.checks)
    assert dec.ir_method == "unavailable"


def test_insufficient_overlap_omits_check():
    n = 40   # < IR_MIN_OVERLAP_BARS
    d = _dates(n)
    strat = list(np.random.default_rng(13).normal(0.001, 0.01, n))
    market = list(np.random.default_rng(14).normal(0.0, 0.01, n))
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    assert all(c["name"] != "idiosyncratic_alpha" for c in dec.checks)
    assert dec.ir_method == "insufficient_overlap"
    assert dec.ir_binding is False
    assert dec.ir_overlap_n == n


def test_tighten_only_new_pass_implies_old_pass():
    """The IR check can only turn a PASS into a FAIL, never rescue a failing gate."""
    strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d))
    old = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=None)
    new = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    if new.passed:
        assert old.passed


def test_ir_check_cannot_rescue_failing_gate():
    strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d), sharpe=-5.0)   # holdout_sharpe FAILS
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    assert dec.passed is False


def test_to_dict_json_clean():
    strat, market, d = _alpha_series()
    wf = _make_wf(holdout_returns=(strat, d))
    dec = evaluate_gate(wf, GateCriteria(), pit_ok=True, market_returns=(market, d))
    dct = dec.to_dict()
    for key in ("ir_method", "ir_binding", "ir_overlap_n", "market_beta",
                "ir_alpha_ann", "ir_residual_vol_ann", "appraisal_ratio"):
        assert key in dct
    for key in ("market_beta", "ir_alpha_ann", "ir_residual_vol_ann", "appraisal_ratio"):
        v = dct[key]
        assert v is None or (isinstance(v, float) and math.isfinite(v))
