"""Unit tests for the pure CSCV / PBO module (algua.research.cscv).

The module returns an aggregate-only ``PboResult`` (no raw logits, no per-split internals) and
FAILS CLOSED (``pbo=None`` + a warning, never a raise) on degenerate input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algua.research.cscv import CSCV_MAX_SUBPERIODS, PboResult, pbo


def _genuine_matrix(n: int = 6, t: int = 4) -> np.ndarray:
    """Row 0 dominates every column; all other rows are a small constant.

    The dominant trial is always both IS-best and OOS-best -> PBO == 0.
    """
    m = np.full((n, t), 0.1, dtype=float)
    m[0, :] = 10.0
    return m


def _overfit_matrix(n: int = 6) -> np.ndarray:
    """Rank-reversal construction: first two cols rank ascending, last two reversed.

    IS-best in the training half is always the OOS-worst -> PBO == 1.
    """
    base = np.arange(n, dtype=float)
    m = np.empty((n, 4), dtype=float)
    for j in range(4):
        m[:, j] = base if j < 2 else base[::-1]
    return m


def test_genuine_low_pbo() -> None:
    r = pbo(_genuine_matrix())
    assert isinstance(r, PboResult)
    assert r.pbo == 0.0
    assert r.warnings == []


def test_overfit_high_pbo() -> None:
    r = pbo(_overfit_matrix())
    assert r.pbo == 1.0
    assert r.pbo >= 0.5


def test_overfit_exceeds_genuine() -> None:
    assert pbo(_overfit_matrix()).pbo > pbo(_genuine_matrix()).pbo


def test_split_count_is_combinations_small_t() -> None:
    r = pbo(_overfit_matrix(n=8))
    # T == 4 (<= CSCV_MAX_SUBPERIODS) -> S == 4, split_count == C(4, 2) == 6.
    assert r.window_count == 4
    assert r.subperiod_count == 4
    assert r.split_count == math.comb(4, 2) == 6


def test_split_count_bounded_at_large_t() -> None:
    # T far larger than the cap: S is bounded to CSCV_MAX_SUBPERIODS and the split count with it.
    m = np.random.default_rng(0).normal(size=(3, 40))
    r = pbo(m)
    assert r.window_count == 40
    assert r.subperiod_count == CSCV_MAX_SUBPERIODS == 16
    assert r.split_count == math.comb(16, 8) == 12_870
    assert r.pbo is not None and 0.0 <= r.pbo <= 1.0


def test_odd_t_evened_down() -> None:
    # T == 5 (odd, >= CSCV_MIN_WINDOWS): S starts at 5 then is decremented to an even 4.
    m = np.random.default_rng(1).normal(size=(4, 5))
    r = pbo(m)
    assert r.window_count == 5
    assert r.subperiod_count == 4
    assert r.split_count == math.comb(4, 2)
    assert r.pbo is not None


# --- fail-closed paths: pbo=None + warning, NEVER a raise -------------------------------------


def test_fail_closed_single_trial() -> None:
    r = pbo(np.ones((1, 4)))
    assert r.pbo is None
    assert r.split_count == 0
    assert any("2 trials" in w for w in r.warnings)


def test_fail_closed_too_few_windows() -> None:
    r = pbo(np.ones((3, 3)))  # T == 3 < CSCV_MIN_WINDOWS (4)
    assert r.pbo is None
    assert any("windows" in w for w in r.warnings)


def test_fail_closed_non_2d() -> None:
    r = pbo(np.array([1.0, 2.0, 3.0, 4.0]))
    assert r.pbo is None
    assert any("2-D" in w for w in r.warnings)


def test_fail_closed_nan() -> None:
    m = _overfit_matrix()
    m[0, 0] = np.nan
    r = pbo(m)
    assert r.pbo is None
    assert any("non-finite" in w for w in r.warnings)


def test_fail_closed_inf() -> None:
    m = _overfit_matrix()
    m[0, 0] = np.inf
    r = pbo(m)
    assert r.pbo is None
    assert any("non-finite" in w for w in r.warnings)


def test_invalid_rank_by_raises() -> None:
    # An invalid rank_by is an API-contract violation (not data) -> raise, not fail-closed.
    with pytest.raises(ValueError, match="rank_by"):
        pbo(_genuine_matrix(), rank_by="sortino")


# --- rank_by: mean vs min, and the min-masking regression -------------------------------------


def _masking_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two 3x20 matrices identical EXCEPT for group-0's two windows, whose SUM is preserved.

    T == 20 -> S == 16, and group 0 bundles windows [0, 1] (base=1, rem=4 -> first 4 groups hold
    2 windows). Row 0 (trial A) is the pivotal top performer.

    * M1: trial A is a flat +2 everywhere.
    * M2: trial A's group-0 windows are (-50, +54) — same SUM (+4) as M1's (+2, +2), so any
      MEAN reduction over the pooled train/test windows is invariant; but the true per-window MIN
      now sees the catastrophic -50 that a group-mean cell would have masked.

    A group-mean implementation would give identical PBO for both under BOTH rank_by values; the
    true per-window implementation must diverge under ``min_sharpe`` while staying invariant under
    ``mean_sharpe``.
    """
    m1 = np.empty((3, 20), dtype=float)
    m1[0, :] = 2.0   # trial A: flat top performer
    m1[1, :] = 1.0   # trial B
    m1[2, :] = 1.5   # trial C
    m2 = m1.copy()
    m2[0, 0] = -50.0  # catastrophic buried window
    m2[0, 1] = 54.0   # partner compensates so the group-0 SUM (+4) is preserved
    return m1, m2


def test_min_reduction_sees_true_worst_window_not_group_mean() -> None:
    m1, m2 = _masking_pair()
    # MEAN is invariant to a sum-preserving reshuffle WITHIN a group (group 0 is always wholly in
    # train or wholly in test), so the buried window is hidden from mean — proving that.
    assert pbo(m1, rank_by="mean_sharpe").pbo == pbo(m2, rank_by="mean_sharpe").pbo
    # MIN sees the individual -50: M2's trial A is demoted out of IS-best on group-0-in-train
    # splits, so the PBO STRICTLY INCREASES. A group-mean bug would leave it unchanged.
    p1 = pbo(m1, rank_by="min_sharpe").pbo
    p2 = pbo(m2, rank_by="min_sharpe").pbo
    assert p1 is not None and p2 is not None
    assert p2 > p1


def test_mean_vs_min_rank_by_diverge() -> None:
    _m1, m2 = _masking_pair()
    # On M2 the two reductions disagree: under mean, trial A stays IS-best (its mean is unchanged)
    # so it generalizes (low PBO); under min, the -50 demotes it, raising PBO.
    assert pbo(m2, rank_by="mean_sharpe").pbo != pbo(m2, rank_by="min_sharpe").pbo


def test_pbo_in_unit_interval() -> None:
    for m in (_genuine_matrix(), _overfit_matrix()):
        for rank_by in ("mean_sharpe", "min_sharpe"):
            r = pbo(m, rank_by=rank_by)
            assert r.pbo is not None and 0.0 <= r.pbo <= 1.0
