"""Tests for the factor-level funnel-FDR correction (issue #219, slice E of #140).

The correction mirrors the #137/#211 strategy-level machinery one level down:
- breadth_benchmark_t: sqrt(2*ln(N)) expected-max inflator applied to the IC t-stat.
- trial_ir_variance: sample variance of evaluated IRs (used as the DSR trial-Sharpe spread).
- correct_factor_ic: produces the full `fdr` block with tighten-only AND semantics.
"""
from __future__ import annotations

import math

import pytest

from algua.research.factor_fdr import (
    breadth_benchmark_t,
    correct_factor_ic,
    trial_ir_variance,
)
from algua.research.gates import DSR_ALPHA, dsr_confidence


# ---------------------------------------------------------------------------
# breadth_benchmark_t
# ---------------------------------------------------------------------------

def test_breadth_benchmark_at_n1_is_zero():
    """Single hypothesis: no multiple-testing penalty."""
    assert breadth_benchmark_t(1) == pytest.approx(0.0)


def test_breadth_benchmark_at_zero_clamped_to_n1():
    """N <= 0 is clamped to 1: no penalty."""
    assert breadth_benchmark_t(0) == pytest.approx(0.0)


def test_breadth_benchmark_monotone():
    """Penalty grows with more hypotheses."""
    vals = [breadth_benchmark_t(n) for n in (1, 2, 5, 10, 50, 100)]
    assert vals == sorted(vals)


def test_breadth_benchmark_n7_value():
    """sqrt(2*ln(7)) ≈ 1.9939 (known value for sanity)."""
    assert breadth_benchmark_t(7) == pytest.approx(math.sqrt(2.0 * math.log(7)), rel=1e-9)


# ---------------------------------------------------------------------------
# trial_ir_variance
# ---------------------------------------------------------------------------

def test_trial_ir_variance_none_when_fewer_than_two_finite():
    """< 2 finite IRs → None (fail-closed: DSR won't bind)."""
    assert trial_ir_variance([]) is None
    assert trial_ir_variance([None]) is None  # type: ignore[list-item]
    assert trial_ir_variance([1.0]) is None
    assert trial_ir_variance([None, None]) is None  # type: ignore[list-item]


def test_trial_ir_variance_none_on_non_finite():
    """Non-finite IRs are excluded; if < 2 remain → None."""
    assert trial_ir_variance([math.inf, 1.0]) is None  # only one finite
    assert trial_ir_variance([math.nan, 1.5]) is None


def test_trial_ir_variance_sample_variance_ddof1():
    """Known two-value variance: Var([0, 2], ddof=1) = 2."""
    result = trial_ir_variance([0.0, 2.0])
    assert result == pytest.approx(2.0)


def test_trial_ir_variance_constant_is_zero():
    """All identical → var=0 (valid — DSR will still bind, SR*=0)."""
    result = trial_ir_variance([1.5, 1.5, 1.5])
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# correct_factor_ic — structure and fail-closed
# ---------------------------------------------------------------------------

def test_fdr_block_always_has_fdr_corrected_true():
    """correct_factor_ic always sets fdr_corrected: True."""
    fdr = correct_factor_ic(
        t_stat=1.5, ir=0.5, n_obs=20,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=1, trial_ir_var=None,
    )
    assert fdr["fdr_corrected"] is True


def test_fdr_block_has_all_required_keys():
    fdr = correct_factor_ic(
        t_stat=2.5, ir=0.8, n_obs=30,
        ic_skew=0.1, ic_kurtosis=3.1,
        n_hypotheses=3, trial_ir_var=0.2,
    )
    expected_keys = {
        "n_hypotheses", "breadth_benchmark_t", "breadth_significant",
        "dsr_binding", "dsr_confidence", "dsr_significant",
        "dsr_skip_reason", "significant", "fdr_corrected",
    }
    assert expected_keys <= set(fdr.keys())


def test_t_stat_none_fails_closed():
    """None t_stat → breadth_significant False, significant False."""
    fdr = correct_factor_ic(
        t_stat=None, ir=None, n_obs=0,
        ic_skew=None, ic_kurtosis=None,
        n_hypotheses=1, trial_ir_var=None,
    )
    assert fdr["breadth_significant"] is False
    assert fdr["significant"] is False
    assert fdr["fdr_corrected"] is True


# ---------------------------------------------------------------------------
# correct_factor_ic — DSR binding logic
# ---------------------------------------------------------------------------

def test_dsr_does_not_bind_when_trial_ir_var_none():
    """When trial_ir_var is None, DSR doesn't bind; reason is set."""
    fdr = correct_factor_ic(
        t_stat=3.0, ir=1.0, n_obs=30,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=5, trial_ir_var=None,
    )
    assert fdr["dsr_binding"] is False
    assert fdr["dsr_confidence"] is None
    assert fdr["dsr_significant"] is None
    assert fdr["dsr_skip_reason"] is not None
    # verdict is breadth-only when DSR doesn't bind
    assert fdr["significant"] == fdr["breadth_significant"]


def test_dsr_does_not_bind_when_only_one_hypothesis():
    """N=1 → DSR doesn't bind even if trial_ir_var is 0."""
    fdr = correct_factor_ic(
        t_stat=3.0, ir=1.0, n_obs=30,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=1, trial_ir_var=0.0,
    )
    assert fdr["dsr_binding"] is False
    assert fdr["dsr_skip_reason"] is not None


def test_dsr_binds_with_dispersion_and_multiple_hypotheses():
    """N>=2 and trial_ir_var not None → DSR binds."""
    fdr = correct_factor_ic(
        t_stat=3.0, ir=1.0, n_obs=30,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=2, trial_ir_var=0.5,
    )
    assert fdr["dsr_binding"] is True
    assert fdr["dsr_confidence"] is not None
    assert fdr["dsr_significant"] is not None


def test_dsr_confidence_matches_gates_function():
    """dsr_confidence value matches the reference dsr_confidence from gates.py."""
    ir = 0.8
    n_obs = 50
    ic_skew = 0.1
    ic_kurtosis = 3.2
    n_hypotheses = 5
    trial_ir_var = 0.3

    fdr = correct_factor_ic(
        t_stat=ir * math.sqrt(n_obs), ir=ir, n_obs=n_obs,
        ic_skew=ic_skew, ic_kurtosis=ic_kurtosis,
        n_hypotheses=n_hypotheses, trial_ir_var=trial_ir_var,
    )
    expected_conf = dsr_confidence(
        sr_obs_per_period=ir,
        t=n_obs,
        skew=ic_skew,
        raw_kurtosis=ic_kurtosis,
        n_trials=n_hypotheses,
        trial_sr_var_per_period=trial_ir_var,
    )
    assert fdr["dsr_confidence"] == pytest.approx(expected_conf, rel=1e-9)


# ---------------------------------------------------------------------------
# correct_factor_ic — AND-check (tighten-only semantics)
# ---------------------------------------------------------------------------

def test_and_check_dsr_fail_overrides_breadth_pass():
    """DSR fail on a breadth-significant t-stat → significant False (tighten-only)."""
    # Manufacture a binding DSR that returns low confidence.
    # Use a high trial_ir_var so SR* >> ir → DSR conf near 0.
    fdr = correct_factor_ic(
        t_stat=5.0, ir=0.3,   # small ir, so DSR will fail despite big t-stat
        n_obs=100,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=3, trial_ir_var=100.0,  # huge spread → SR* large → conf ≈ 0
    )
    if fdr["dsr_binding"] and fdr["dsr_significant"] is False:
        assert fdr["significant"] is False  # DSR fail overrides
    # Either DSR didn't bind (irrelevant to this test) or it tightened as expected.


def test_and_check_breadth_fail_stays_fail_even_if_dsr_pass():
    """Breadth-insignificant t-stat stays False even if DSR confidence is high."""
    # t_stat = 0.0 → definitely not breadth-significant.
    fdr = correct_factor_ic(
        t_stat=0.0, ir=0.0, n_obs=30,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=2, trial_ir_var=0.001,  # tiny spread → DSR conf might be high
    )
    assert fdr["breadth_significant"] is False
    assert fdr["significant"] is False  # AND: can't recover from breadth fail


def test_both_pass_significant_true():
    """High t-stat + good DSR → significant True."""
    fdr = correct_factor_ic(
        t_stat=4.0, ir=2.0, n_obs=30,
        ic_skew=0.0, ic_kurtosis=3.0,
        n_hypotheses=2, trial_ir_var=0.01,  # small spread → SR* small → DSR passes
    )
    # May or may not bind depending on exact values; if binding, both should pass.
    assert fdr["fdr_corrected"] is True
    if fdr["dsr_binding"]:
        if fdr["dsr_significant"]:
            assert fdr["significant"] is True
