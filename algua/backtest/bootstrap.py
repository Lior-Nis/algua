"""Stationary-bootstrap DSR calibration — serial-dependence bootstrap (#221, Phase 3 Slice 2).

Provides:
- ``stable_bootstrap_seed``   — deterministic cross-process seed from strategy identity.
- ``politis_white_block_length`` — automatic Politis & White (2004) / Patton et al. (2009)
  optimal block length for the stationary bootstrap (Politis & Romano 1994).
- ``_dsr_conf_core``          — per-resample DSR confidence (mirrors gates.dsr_confidence).
- ``stationary_bootstrap_dsr`` — vectorised stationary-bootstrap distribution of DSR confidence;
  returns the lower-tail quantile as ``lower_confidence``.

This module is a pure-maths leaf: no algua.research import (blocked by import-linter).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import norm as _norm
from scipy.stats import skew as _scipy_skew


class BootstrapResult(NamedTuple):
    """Result from ``stationary_bootstrap_dsr``.

    lower_confidence: lower-quantile bootstrap DSR confidence, or None on degenerate input.
    seed_used:        the RNG seed that produced this result (for audit / reproducibility).
    b_used:           number of resamples requested (``b`` argument; some may be non-finite).
    block_len:        block length actually used (Politis-White auto or override).
    """

    lower_confidence: float | None
    seed_used: int
    b_used: int
    block_len: int


# ---------------------------------------------------------------------------
# Stable seed
# ---------------------------------------------------------------------------


def stable_bootstrap_seed(
    strategy_name: str,
    holdout_start: str,
    holdout_end: str,
    config_hash: str,
) -> int:
    """Deterministic cross-process seed from strategy identity.

    Uses SHA-256 (never Python's built-in ``hash()``, which is salted per-process and produces
    different values across runs). Takes the first 8 bytes of the digest as a big-endian integer.
    """
    payload = "\x00".join([strategy_name, holdout_start, holdout_end, config_hash])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Politis & White (2004) / Patton, Politis & White (2009) block-length selector
# ---------------------------------------------------------------------------


def politis_white_block_length(
    returns: Sequence[float],
    max_fraction: float,
) -> int:
    """Automatic stationary-bootstrap block length.

    Algorithm: Politis & White (2004) "Automatic Block-Length Selection for the Dependent
    Bootstrap", Econometric Theory 20(6), 1;  Patton, Politis & White (2009) correction,
    Econometric Reviews 28(4), 372–375.

    Parameters
    ----------
    returns:
        Per-period return series (list or array).
    max_fraction:
        Cap block length at ``max(1, floor(n * max_fraction))``.

    Returns
    -------
    int
        Block length in [1, max(1, floor(n * max_fraction))].  Returns 1 on any degenerate
        input (n < 3, zero variance, non-finite autocovariances).
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    max_bl = max(1, int(math.floor(n * max_fraction)))

    # --- Degenerate guard: need at least 3 observations ---
    # Politis & White (2004), §2: the estimator requires meaningful autocorrelation structure.
    if n < 3:
        return 1

    xbar = x.mean()
    xc = x - xbar  # centred

    # --- Step 1: Sample autocovariances R(k) and autocorrelations rho(k) ---
    # R(k) = (1/n) * sum_{i=0}^{n-k-1} xc[i] * xc[i+k]
    # rho(k) = R(k) / R(0)
    # We need lags up to M + K_N (determined below); start with an upper bound.
    # K_N = max(5, ceil(sqrt(log10(n)))) as in Patton et al. (2009).
    K_N = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    # Significance bound: c * sqrt(log10(n) / n) with c=2.0 (Politis & White 2004, eq. 16).
    c = 2.0
    bound = c * math.sqrt(math.log10(n) / n)

    # We do not know M yet; compute up to a principled upper limit first.
    # Dynamic horizon: grows with n to detect long-memory series (high-order AR) that need
    # lags well beyond the old fixed ~30-lag cap. The n//3 term ensures we always scan at
    # least one-third of the series, while the 4*K_N+10 baseline keeps short series fast.
    max_lag = min(n - 1, max(4 * K_N + 10, n // 3))

    def _acov(k: int) -> float:
        """Biased autocovariance R(k) = (1/n) * sum xc[i]*xc[i+k]."""
        if k >= n:
            return 0.0
        return float(np.dot(xc[: n - k], xc[k:])) / n

    r0 = _acov(0)

    # --- Degenerate guard: zero variance ---
    if r0 <= 0 or not math.isfinite(r0):
        return 1

    # Precompute normalised autocorrelations up to max_lag.
    rho = np.array([_acov(k) / r0 for k in range(max_lag + 1)])

    # --- Step 2: Bandwidth M via Politis & White (2004) §3 / Patton et al. (2009) ---
    # Scan k=1,2,... for the smallest m_hat such that |rho(m_hat+j)| < bound
    # for all j=1..K_N (K_N consecutive insignificant lags after m_hat).
    # If no such run exists, m_hat = last significant lag (or 1 if none significant).
    # Scanning the full dynamic horizon (vs. the old ~30-lag cap) ensures long-memory series
    # (e.g., AR(1) with phi≈0.95) are not undercounted when significant lags extend beyond 30.
    m_hat: int | None = None
    last_sig = 0  # largest k with |rho(k)| >= bound

    for k in range(1, max_lag - K_N + 1):
        if abs(rho[k]) >= bound:
            last_sig = k
        # Check if K_N consecutive lags after k are all insignificant.
        if k >= 1 and all(abs(rho[k + j]) < bound for j in range(1, K_N + 1) if k + j <= max_lag):
            m_hat = k
            break

    if m_hat is None:
        # No clean run found; use the last significant lag (minimum 1 to avoid M=0).
        m_hat = max(last_sig, 1)

    M = min(2 * m_hat, n - 1)
    if M < 1:
        M = 1

    # Ensure rho is computed to at least M + K_N.
    need = M + K_N
    if need > max_lag:
        new_max = min(n - 1, need)
        extra = np.array([_acov(k) / r0 for k in range(max_lag + 1, new_max + 1)])
        rho = np.concatenate([rho, extra])
        max_lag = new_max

    # --- Step 3: Flat-top lag window lambda(s) ---
    # Politis & White (2004), eq. (5):
    # lam(s) = 1                 if |s| <= 0.5
    #         = 2*(1-|s|)        if 0.5 < |s| <= 1.0
    #         = 0                otherwise
    def _lam(s: float) -> float:
        a = abs(s)
        if a <= 0.5:
            return 1.0
        if a <= 1.0:
            return 2.0 * (1.0 - a)
        return 0.0

    # --- Step 4: Long-run variance sums G0 and G ---
    # G0 = sum_{k=-M..M} lam(k/M) * R(|k|)
    # G  = sum_{k=-M..M} lam(k/M) * |k| * R(|k|)
    # Politis & White (2004), eq. (14) and (15).
    # Use symmetry: k=0 contributes once; k=1..M contributes twice (k and -k).
    G0 = _lam(0.0) * r0  # k=0 term
    G = 0.0
    for k in range(1, M + 1):
        lw = _lam(k / M)
        if lw == 0.0:
            continue
        rk = _acov(k) if k <= max_lag else 0.0
        G0 += 2.0 * lw * rk
        G += 2.0 * lw * k * rk

    # --- Degenerate guard: G0 must be positive ---
    if not math.isfinite(G0) or G0 <= 0:
        return 1
    if not math.isfinite(G):
        return 1

    # --- Step 5: Optimal block length for the stationary bootstrap ---
    # Politis & White (2004), eq. (9):
    # b_opt_SB = ( (G/G0)^2 )^(1/3) * n^(1/3) = (G/G0)^(2/3) * n^(1/3)
    # (Equivalent to (2*G^2 / D_SB)^(1/3) * n^(1/3) with D_SB = 2*G0^2.)
    ratio = G / G0
    b_opt = (ratio**2) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))

    # --- Step 6: Degenerate guards and clamping ---
    if not math.isfinite(b_opt) or b_opt < 1.0:
        return 1
    return max(1, min(int(round(b_opt)), max_bl))


# ---------------------------------------------------------------------------
# Per-resample DSR confidence core
# ---------------------------------------------------------------------------


def _dsr_conf_core(
    sr_pp: float,
    t: int,
    skew: float,
    kurt: float,
    sr_star: float,
) -> float | None:
    """DSR confidence for one resample using pre-computed SR* (per-period).

    MUST stay in lock-step with gates.dsr_confidence — pinned by
    test_dsr_core_consistent_with_gates.

    Formula (Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014):
        var_term = 1 - skew*sr + (kurt-1)/4 * sr^2
        z = (sr - sr_star) * sqrt(t-1) / sqrt(var_term)
        confidence = Phi(z)

    ``kurt`` is Pearson kurtosis (=3 for Gaussian), matching scipy's ``kurtosis(fisher=False)``
    and ``metrics_from_returns``.  Returns None on any non-finite intermediate value.
    """
    if not math.isfinite(sr_pp) or not math.isfinite(skew) or not math.isfinite(kurt):
        return None
    if t <= 1:
        return None
    sr = sr_pp
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    conf = float(_norm.cdf(z))
    return conf if math.isfinite(conf) else None


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------


def stationary_bootstrap_dsr(
    returns: Sequence[float],
    dates: Sequence[str],
    sr_star: float | None,
    dsr_alpha: float,
    b: int,
    seed: int,
    *,
    block_len_auto: bool = True,
    block_len_override: int | None = None,
    lower_quantile: float,
    max_block_fraction: float = 0.5,
) -> BootstrapResult:
    """Stationary-bootstrap distribution of DSR confidence; return lower-tail quantile.

    Resampling: Politis & Romano (1994) "The stationary bootstrap", JASA 89(428), 1303–1313.
    Block length: ``politis_white_block_length`` (Politis & White 2004; Patton et al. 2009)
    unless ``block_len_auto=False`` and ``block_len_override`` is set.

    Parameters
    ----------
    returns:          Per-period OOS return series.
    dates:            Corresponding date labels (same length as returns; used for audit only).
    sr_star:          Pre-computed DSR benchmark SR* (per-period).  None → lower_confidence=None.
    dsr_alpha:        Not used in the computation; carried for caller audit only.
    b:                Number of stationary-bootstrap resamples.
    seed:             RNG seed (use ``stable_bootstrap_seed`` for reproducibility).
    block_len_auto:   If True, use Politis-White auto block length (default).
    block_len_override: Manual block length override; only used when block_len_auto=False.
    lower_quantile:   Quantile of the bootstrap distribution to return as lower_confidence.
    max_block_fraction: Cap for Politis-White block length = max(1, floor(T * fraction)).

    Returns
    -------
    BootstrapResult
        lower_confidence: None on degenerate input or too few finite resamples (< b/2).
    """
    arr = np.asarray(returns, dtype=float)
    T = len(arr)

    # Guard max_block_fraction to a sane range; treat out-of-range as the default 0.5.
    if not (0 < max_block_fraction <= 1):
        max_block_fraction = 0.5

    # Compute block-length cap once; applied to BOTH the auto and override paths so the
    # contract is consistent regardless of which path is taken.
    max_bl = max(1, int(math.floor(T * max_block_fraction))) if T > 0 else 1

    # --- Compute block length first (always, for audit; uses T=1 floor safely) ---
    if not block_len_auto and block_len_override is not None and block_len_override >= 1:
        # Clamp manual override to the same cap that the auto path enforces.
        block_len = min(int(block_len_override), max_bl)
    else:
        block_len = politis_white_block_length(returns, max_block_fraction) if T > 1 else 1

    # --- Top guards: return None early ---
    if T <= 1 or sr_star is None or not np.all(np.isfinite(arr)) or b < 1:
        return BootstrapResult(None, seed, b, block_len)

    # --- Vectorised stationary resampling ---
    # Politis & Romano (1994): each position either starts a new block (prob p=1/block_len)
    # or continues the previous index circularly.
    rng = np.random.default_rng(seed)
    p = 1.0 / block_len

    # Shape: (b, T). Build index paths in vectorised fashion.
    # For each resample r and position t:
    #   restart[r,t] ~ Bernoulli(p)  (position 0 always restarts)
    #   when restart: idx[r,t] = uniform start in [0, T)
    #   else:         idx[r,t] = (idx[r,t-1] + 1) % T
    restart = rng.random((b, T)) < p  # (b, T) bool
    restart[:, 0] = True  # always start fresh at position 0
    fresh_starts = rng.integers(0, T, size=(b, T))  # (b, T) random starts

    # Fill forward sequentially along T; numpy does not have a native scan,
    # but T ≤ 500 so a Python loop over T (not over b) is fast.
    idx = np.empty((b, T), dtype=np.intp)
    idx[:, 0] = fresh_starts[:, 0]
    for t in range(1, T):
        prev = idx[:, t - 1]
        cont = (prev + 1) % T  # circular continuation
        idx[:, t] = np.where(restart[:, t], fresh_starts[:, t], cont)

    # Gather resamples: shape (b, T)
    samples = arr[idx]  # (b, T)

    # --- Per-resample moments and DSR confidence ---
    means = samples.mean(axis=1)               # (b,)
    stds = samples.std(axis=1, ddof=1)         # (b,)

    # Compute skewness and kurtosis via scipy (matches metrics_from_returns).
    # scipy.stats.skew/kurtosis accept 2-D arrays with axis argument.
    skews = _scipy_skew(samples, axis=1)                          # (b,)
    kurts = _scipy_kurtosis(samples, axis=1, fisher=False)        # (b,) Pearson

    finite_confidences: list[float] = []
    for i in range(b):
        std_i = float(stds[i])
        if std_i <= 0 or not math.isfinite(std_i):
            continue
        sr_i = float(means[i]) / std_i
        conf = _dsr_conf_core(sr_i, T, float(skews[i]), float(kurts[i]), sr_star)
        if conf is not None:
            finite_confidences.append(conf)

    if len(finite_confidences) < b / 2:
        return BootstrapResult(None, seed, b, block_len)

    lower_conf = float(np.quantile(np.asarray(finite_confidences), lower_quantile))
    return BootstrapResult(lower_conf, seed, b, block_len)
