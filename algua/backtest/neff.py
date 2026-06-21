"""Effective independent trials N_eff — shadow-only (#221, Phase 3 Slice 3 / Q2.2).

Kish average-pairwise-correlation effective trial count:
    N_eff = raw_n / (1 + (raw_n - 1) * rho_bar_lower)
where rho_bar_lower is a conservative (lower-bound) estimate of the mean off-diagonal pairwise
Pearson correlation of funnel siblings' date-aligned overlapping OOS return streams.

Q2.2 (this file): the ρ̄ standard-error is now computed via a Fisher-z CI using an effective
sample size M_eff = n_sib (sibling count, not pair count C(n,2)).

Rationale for M_eff = n_sib:
  Pairs sharing a strategy are dependent (each strategy appears in (n_sib-1) pairs). The
  independent information in the C(n_sib,2) pairwise z-scores scales with the number of
  STRATEGIES (n_sib), not the number of pairs. Using M = C(n,2) treats correlated pairs as
  independent, understating uncertainty → ρ̄_lower too high → N_eff too low (too lenient).
  Using M_eff = n_sib produces a larger SE_z → lower ρ̄_lower → higher, more conservative N_eff.

Must-tighten guarantee (spec Q2.2):
  The Fisher-z bound can in theory be LESS conservative than the linear σ_ρ/√M_eff bound
  in some regimes (e.g. near ρ̄→1 where arctanh compresses). To guarantee the spec's hard
  constraint (new bound ≤ old σ_ρ/√M bound), we take:
    rho_lower = min(fisher_rho_lower, linear_eff_lower)
  where linear_eff_lower = clamp(ρ̄ − k·σ_ρ/√M_eff, 0, 1) (already ≥-conservative vs σ_ρ/√M
  since M_eff = n_sib ≤ C(n_sib,2) = M).

Pure-maths leaf: no algua.research import (thresholds are passed in as parameters). Estimation
lives in algua/backtest per the Phase-1 architecture boundary; gates.py receives only the
pre-computed scalar in the audit payload. N_eff is SHADOW-ONLY in Phase 3 (never the binding
DSR trial count).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import NamedTuple

import numpy as np


class NEffResult(NamedTuple):
    n_eff: int | None        # None => no N_eff evidence; raw N stands (fail-open in shadow mode)
    rho_bar: float | None    # rho_bar_lower actually used (None when n_eff is None)
    n_siblings: int
    n_pairs: int


def _pair_correlation(a, b, min_overlap_bars) -> tuple[float, int] | None:
    """Inner-join two (returns, dates) streams on DATE; return (Pearson_corr, n_overlap) or None.

    Returns None if the overlap is too short, the correlation is non-finite (e.g. a zero-variance
    stream), or either stream contains duplicate dates (invalid data — fail-closed).

    The returned n_overlap (count of common dates) feeds the Fisher-z per-pair sampling variance
    v = 1/(n_overlap - 3) in estimate_n_eff.  n_overlap >= min_overlap_bars >= 21 > 3, so v is
    always well-defined on the success path.
    """
    ar, ad = a
    br, bd = b
    # Duplicate dates in either stream are invalid; fail-closed.
    if len(set(ad)) != len(ad) or len(set(bd)) != len(bd):
        return None
    amap = dict(zip(ad, ar, strict=True))
    bmap = dict(zip(bd, br, strict=True))
    common = sorted(set(amap) & set(bmap))
    n_overlap = len(common)
    if n_overlap < min_overlap_bars:
        return None
    av = np.array([amap[d] for d in common], dtype=float)
    bv = np.array([bmap[d] for d in common], dtype=float)
    if np.ptp(av) == 0.0 or np.ptp(bv) == 0.0:
        return None
    rho = float(np.corrcoef(av, bv)[0, 1])
    if not math.isfinite(rho):
        return None
    return rho, n_overlap


def estimate_n_eff(
    raw_n: int,
    sibling_streams: Sequence[tuple[list[float], list[str]]],
    *,
    min_siblings: int,
    min_overlap_bars: int,
    shrinkage_k: float,
) -> NEffResult:
    n_sib = len(sibling_streams)
    if raw_n < 1 or n_sib < min_siblings:
        return NEffResult(None, None, n_sib, 0)
    rhos: list[float] = []
    sampling_vars: list[float] = []  # v_i = 1/(n_overlap_i - 3) per pair
    for a, b in combinations(sibling_streams, 2):
        result = _pair_correlation(a, b, min_overlap_bars)
        if result is None:               # strict: any bad pair -> no estimate (raw N stands)
            return NEffResult(None, None, n_sib, len(rhos))
        rho, n_overlap = result
        rhos.append(rho)
        # n_overlap >= min_overlap_bars >= 21 > 3, so n_overlap - 3 >= 18 > 0 (always safe)
        sampling_vars.append(1.0 / (n_overlap - 3))
    m = len(rhos)
    if m == 0:  # n_sib >= min_siblings(>=2) guarantees m>=1, defensive
        return NEffResult(None, None, n_sib, 0)

    arr = np.asarray(rhos, dtype=float)
    rho_mean = float(arr.mean())
    sigma_rho = float(arr.std(ddof=1)) if m >= 2 else 0.0

    # ------------------------------------------------------------------
    # Fisher-z effective-N estimator (Q2.2)
    # ------------------------------------------------------------------
    # Step 1: Fisher-z transform each pair correlation (clamped to avoid ±inf at ±1).
    _EPS = 1e-12
    z_arr = np.arctanh(np.clip(arr, -1.0 + _EPS, 1.0 - _EPS))
    z_bar = float(z_arr.mean())

    # Step 2: per-pair sampling variance and mean.
    v_arr = np.asarray(sampling_vars, dtype=float)
    mean_v = float(v_arr.mean())

    # Step 3: M_eff = n_sib (sibling count, NOT pair count C(n_sib,2)).
    # Rationale: pairs sharing a strategy are dependent; independent information scales
    # with n_sib (strategies), not m = C(n_sib,2) (pairs).  See module docstring.
    m_eff = n_sib

    # Step 4: combined variance of z_bar (dispersion across pairs + within-pair sampling noise).
    dispersion_var = float(z_arr.var(ddof=1)) if m >= 2 else 0.0
    var_z_bar = (dispersion_var + mean_v) / m_eff
    se_z = math.sqrt(max(var_z_bar, 0.0))  # max guards against tiny floating-point negatives

    # Step 5: lower z-bound and back-transform to correlation space.
    z_lower = z_bar - shrinkage_k * se_z
    fisher_rho_lower = min(1.0, max(0.0, math.tanh(z_lower)))

    # Step 6: must-tighten guarantee — also compute the linear effective-N bound.
    # linear_eff_lower uses M_eff=n_sib (< pair count M), so it is already ≥-conservative
    # vs the old σ_ρ/√M bound.  Taking min(fisher, linear_eff) ensures we NEVER output a
    # bound that is less conservative than the old one, even near ρ̄→1 (tanh compression).
    linear_eff_lower = min(1.0, max(0.0, rho_mean - shrinkage_k * sigma_rho / math.sqrt(m_eff)))
    rho_lower = min(fisher_rho_lower, linear_eff_lower)

    # ------------------------------------------------------------------
    # Kish formula + conservative ceil rounding (unchanged from Slice 3)
    # ------------------------------------------------------------------
    n_eff = raw_n / (1.0 + (raw_n - 1) * rho_lower)
    if not math.isfinite(n_eff):
        return NEffResult(None, None, n_sib, m)
    # Round UP (conservative): a lower N_eff would lower the DSR benchmark SR* (more lenient),
    # so ceil is the anti-lenient direction. ceil(n_eff) <= raw_n always holds since n_eff <= raw_n.
    n_eff_int = max(1, min(raw_n, math.ceil(n_eff)))
    return NEffResult(n_eff_int, rho_lower, n_sib, m)
