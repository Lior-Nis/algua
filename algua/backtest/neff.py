"""Effective independent trials N_eff — shadow-only (#221, Phase 3 Slice 3).

Kish average-pairwise-correlation effective trial count:
    N_eff = raw_n / (1 + (raw_n - 1) * rho_bar_lower)
where rho_bar_lower is a conservative (lower-bound) estimate of the mean off-diagonal pairwise
Pearson correlation of funnel siblings' date-aligned overlapping OOS return streams.

Pure-maths leaf: no algua.research import (thresholds are passed in as parameters). Estimation lives
in algua/backtest per the Phase-1 architecture boundary; gates.py receives only the pre-computed
scalar in the audit payload. N_eff is SHADOW-ONLY in Phase 3 (never the binding DSR trial count).
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


def _pair_correlation(a, b, min_overlap_bars):
    """Inner-join two (returns, dates) streams on DATE, return Pearson corr or None if the overlap
    is too short or the correlation is non-finite (e.g. a zero-variance stream)."""
    ar, ad = a
    br, bd = b
    amap = dict(zip(ad, ar, strict=True))
    bmap = dict(zip(bd, br, strict=True))
    common = sorted(set(amap) & set(bmap))
    if len(common) < min_overlap_bars:
        return None
    av = np.array([amap[d] for d in common], dtype=float)
    bv = np.array([bmap[d] for d in common], dtype=float)
    if np.ptp(av) == 0.0 or np.ptp(bv) == 0.0:
        return None
    rho = float(np.corrcoef(av, bv)[0, 1])
    return rho if math.isfinite(rho) else None


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
    for a, b in combinations(sibling_streams, 2):
        rho = _pair_correlation(a, b, min_overlap_bars)
        if rho is None:                      # strict: any bad pair -> no estimate (raw N stands)
            return NEffResult(None, None, n_sib, len(rhos))
        rhos.append(rho)
    m = len(rhos)
    if m == 0:  # n_sib >= min_siblings(>=2) guarantees m>=1, defensive
        return NEffResult(None, None, n_sib, 0)
    arr = np.asarray(rhos, dtype=float)
    rho_mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(m)) if m >= 2 else 0.0
    rho_lower = min(1.0, max(0.0, rho_mean - shrinkage_k * se))
    n_eff = raw_n / (1.0 + (raw_n - 1) * rho_lower)
    if not math.isfinite(n_eff):
        return NEffResult(None, None, n_sib, m)
    n_eff_int = max(1, min(raw_n, int(round(n_eff))))
    return NEffResult(n_eff_int, rho_lower, n_sib, m)
