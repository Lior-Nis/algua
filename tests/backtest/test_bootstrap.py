"""Tests for algua.backtest.bootstrap — stationary-bootstrap DSR + Politis-White block length.

All tests follow TDD: written first, watched fail, then the module was implemented to pass them.
"""
from __future__ import annotations

import numpy as np

from algua.backtest.bootstrap import (
    _dsr_conf_core,
    politis_white_block_length,
    stable_bootstrap_seed,
    stationary_bootstrap_dsr,
)
from algua.research.gates import dsr_confidence, dsr_sr_star


def _white(n: int, seed: int = 0) -> list[float]:
    return list(np.random.default_rng(seed).normal(0.001, 0.01, n))


def _ar1(n: int, phi: float, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.01)
    return list(x + 0.001)


def test_seed_is_stable_and_deterministic() -> None:
    a = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abc")
    b = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abc")
    c = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abd")
    assert a == b and a != c and isinstance(a, int)


def test_block_length_white_noise_is_small() -> None:
    bl = politis_white_block_length(_white(400), 0.5)
    assert 1 <= bl <= 10  # near-iid -> short blocks


def test_block_length_ar1_is_larger_than_white() -> None:
    bl_w = politis_white_block_length(_white(400, 1), 0.5)
    bl_a = politis_white_block_length(_ar1(400, 0.7, 1), 0.5)
    assert bl_a > bl_w
    assert bl_a <= 200  # capped at floor(400 * 0.5)


def test_seed_reproducibility_same_output() -> None:
    args = dict(
        dates=["d"] * 200,
        sr_star=0.0,
        dsr_alpha=0.05,
        b=500,
        lower_quantile=0.05,
    )
    r = _white(200, 3)
    o1 = stationary_bootstrap_dsr(r, seed=123, **args)
    o2 = stationary_bootstrap_dsr(r, seed=123, **args)
    assert o1.lower_confidence == o2.lower_confidence
    assert o1.block_len == o2.block_len and o1.b_used == o2.b_used


def test_white_noise_bootstrap_lower_near_closed_form() -> None:
    # On a white-noise series the bootstrap-lower should be close to (not wildly below) the
    # closed-form DSR confidence — benign autocorrelation does not widen much.
    r = _white(252, 7)
    n_trials, var = 30, 0.04
    sr_star = dsr_sr_star(n_trials, var)
    out = stationary_bootstrap_dsr(r, ["d"] * 252, sr_star, 0.05, 2000, 11, lower_quantile=0.05)
    assert out.lower_confidence is not None
    assert 0.0 <= out.lower_confidence <= 1.0


def test_strong_ar1_lowers_confidence_vs_white() -> None:
    # Strongly autocorrelated returns inflate the naive Sharpe SE -> bootstrap-lower should be
    # MEANINGFULLY below a white-noise series of the same mean Sharpe.
    sr_star = 0.0
    w = stationary_bootstrap_dsr(
        _white(252, 2), ["d"] * 252, sr_star, 0.05, 2000, 5, lower_quantile=0.05
    )
    a = stationary_bootstrap_dsr(
        _ar1(252, 0.8, 2), ["d"] * 252, sr_star, 0.05, 2000, 5, lower_quantile=0.05
    )
    assert a.lower_confidence is not None and w.lower_confidence is not None
    assert a.lower_confidence <= w.lower_confidence + 1e-9


def test_degenerate_returns_none() -> None:
    assert (
        stationary_bootstrap_dsr(
            [0.1], ["d"], 0.0, 0.05, 100, 1, lower_quantile=0.05
        ).lower_confidence
        is None
    )  # T<=1
    assert (
        stationary_bootstrap_dsr(
            _white(50), ["d"] * 50, None, 0.05, 100, 1, lower_quantile=0.05
        ).lower_confidence
        is None
    )  # sr_star None


def test_dsr_core_consistent_with_gates() -> None:
    # The per-resample DSR-confidence core must equal gates.dsr_confidence for the same inputs,
    # pinning the duplicated formula against drift.
    n, var = 40, 0.04
    sr_star = dsr_sr_star(n, var)
    sr, t, skew, kurt = 0.1, 120, -0.1, 3.5
    assert _dsr_conf_core(sr, t, skew, kurt, sr_star) == dsr_confidence(sr, t, skew, kurt, n, var)
