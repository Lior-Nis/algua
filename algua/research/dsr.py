"""Deflated-Sharpe-Ratio evidence layer — SR*, DSR confidence, dispersion floor, effective breadth
(extracted from gates.py, #335).

Pure-maths leaf: imports the backtest annualization constant and scipy's normal distribution; no
research-internal imports.
"""
from __future__ import annotations

import math

from scipy.stats import norm as _norm

from algua.backtest._constants import ANN

# DSR evidence layer (#211, Phase 1). Protected constants — relaxing them weakens the gate.
DSR_ALPHA = 0.05  # require >= 95% confidence the true Sharpe beats the selection-inflated benchmark
EULER_MASCHERONI = 0.5772156649015329  # gamma_E, the DSR expected-max weight (NOT e^-1)

# Serial-dependence bootstrap (#221, Phase 3 Slice 2). Protected — relaxing weakens the gate.
DSR_BOOTSTRAP_RESAMPLES = 2000            # stationary-bootstrap resample count (B)
DSR_BOOTSTRAP_LOWER_QUANTILE = 0.05       # lower quantile of the bootstrap DSR-confidence dist.
MAX_BOOTSTRAP_BLOCK_LEN_FRACTION = 0.5    # cap block length at max(1, floor(T * FRACTION))

# Effective independent trials N_eff (#221, Phase 3 Slice 3). SHADOW-ONLY in Phase 3 — recorded in
# the audit payload, never the binding DSR trial count (a lower N_eff would loosen the gate; it goes
# binding only at Slice 5, bundled with haircut retirement). Protected — load-bearing from Slice 5.
MIN_N_EFF_SIBLINGS = 5          # min overlapping-OOS sibling streams to attempt an N_eff estimate
MIN_CORR_OVERLAP_BARS = 21      # min date-aligned shared bars per sibling pair to estimate a corr
RHO_BAR_SHRINKAGE_K = 1.0       # SE multiplier for the conservative (lower-bound) rho_bar

# Funnel-wide dispersion floor (#221, Phase 3 Slice 0). Min finite per-strategy trial-Sharpe
# variances needed to form a meaningful cross-strategy floor. Below this, the floor is unavailable
# and the DSR falls back to own-sweep variance (Phase-1 behavior). Protected — raising it weakens
# the floor's availability; the floor can only ever TIGHTEN the gate, so its absence is
# conservative.
MIN_FUNNEL_FLOOR_STRATEGIES = 5


def floored_trial_var_per_period(
    own_var_pp: float, floor_var_pp: float | None
) -> float | None:
    """Own-variance-first dispersion floor (#221 Slice 0): validate own (finite & >=0 else None),
    then max(own, floor) only when floor is finite and strictly greater. Returns the floored
    per-period trial-Sharpe variance, or None when own is degenerate (the floor must never rescue a
    degenerate own variance into a pass)."""
    if not math.isfinite(own_var_pp) or own_var_pp < 0.0:
        return None
    var_used = own_var_pp
    if floor_var_pp is not None and math.isfinite(floor_var_pp) and floor_var_pp > var_used:
        var_used = floor_var_pp
    return var_used


def dsr_sr_star(n_trials: int, trial_sr_var_per_period: float) -> float | None:
    """Selection-inflated benchmark SR* (per-period). 0.0 for n<=1; else the López de Prado E[max]
    of n trial Sharpes scaled by sqrt(var). None on degenerate input."""
    n = int(n_trials)
    if n < 1:
        return None
    if not math.isfinite(trial_sr_var_per_period) or trial_sr_var_per_period < 0.0:
        return None
    if n <= 1:
        return 0.0
    sr_star = math.sqrt(trial_sr_var_per_period) * (
        (1.0 - EULER_MASCHERONI) * float(_norm.ppf(1.0 - 1.0 / n))
        + EULER_MASCHERONI * float(_norm.ppf(1.0 - 1.0 / (n * math.e)))
    )
    return sr_star if math.isfinite(sr_star) else None


def dsr_sr_star_annualized(
    n_trials: int, trial_var_ann: float | None, floor_var_ann: float | None
) -> float | None:
    """SR* (per-period) from ANNUALIZED inputs — the SR* the binding dsr_evidence uses. Converts
    /ANN, applies the funnel floor, then dsr_sr_star. None if no own variance or degenerate."""
    if trial_var_ann is None or not math.isfinite(trial_var_ann):
        return None
    own_pp = trial_var_ann / ANN
    floor_pp = (
        floor_var_ann / ANN
        if floor_var_ann is not None and math.isfinite(floor_var_ann)
        else None
    )
    var_used = floored_trial_var_per_period(own_pp, floor_pp)
    if var_used is None:
        return None
    return dsr_sr_star(n_trials, var_used)


def dsr_confidence(
    sr_obs_per_period: float,
    t: int,
    skew: float,
    raw_kurtosis: float,
    n_trials: int,
    trial_sr_var_per_period: float,
    funnel_floor_var_per_period: float | None = None,
) -> float | None:
    """Deflated-Sharpe-Ratio confidence (Bailey & López de Prado): the probability — in [0,1],
    NOT a p-value — that the true (per-period) Sharpe exceeds the expected maximum Sharpe of
    ``n_trials`` selections.

        SR* = sqrt(var) * [ (1-gamma_E)*Z^-1(1-1/N) + gamma_E*Z^-1(1-1/(N*e)) ]   for N > 1
        SR* = 0                                                                    for N <= 1
        DSR = Phi( (SR_obs - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_obs + (kurt-1)/4 * SR_obs^2) )

    ``raw_kurtosis`` is Pearson kurtosis (=3 for Gaussian), so the variance term reduces to the
    Lo/Mertens 1 + SR^2/2 for a normal series. Inputs are PER-PERIOD; the caller converts from the
    system's annualized Sharpes. Returns None (fail closed) on any degenerate input.

    ``funnel_floor_var_per_period`` is the optional funnel-wide dispersion floor (#221 Slice 0).
    When finite and > own variance, ``max(own, floor)`` is used — a tighten-only operation (SR*
    can only rise, DSR confidence can only fall). A None or non-finite floor falls back to own
    variance (Phase-1 behavior). The own variance is validated FIRST (fail-closed on
    non-finite/negative): the floor must never rescue a degenerate own variance into a pass —
    that would be a FAIL->PASS tighten-only violation."""
    n = int(n_trials)
    if n < 1:                      # invalid breadth
        return None
    if t <= 1:                     # PSR needs sqrt(T-1) > 0; underpowered holdout
        return None
    if not math.isfinite(sr_obs_per_period) or not math.isfinite(skew) \
            or not math.isfinite(raw_kurtosis):
        return None
    var_used = floored_trial_var_per_period(trial_sr_var_per_period, funnel_floor_var_per_period)
    if var_used is None:
        return None
    sr_star = dsr_sr_star(n, var_used)
    if sr_star is None:
        return None
    sr = sr_obs_per_period

    var_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    conf = float(_norm.cdf(z))
    return conf if math.isfinite(conf) else None


def effective_funnel_breadth(
    own_lifetime: int,
    windowed_total: int,
    family_lifetime_effective: int = 0,
) -> int:
    """3-way max (tighten-only): own lifetime, funnel-wide windowed, family+ancestor lifetime.

    Effective funnel breadth fed to the haircut (Wall A): ``max`` of this strategy's LIFETIME
    recorded breadth, the funnel-wide breadth recorded in the rolling window (``windowed_total``
    INCLUDES this strategy's own windowed sweeps, so no double-count, no name-exclusion subtlety),
    and the family+ancestor lifetime combos (``family_lifetime_effective``).
    An *effective funnel-breadth policy*, NOT a literal independent-trial count. A lone hypothesis
    with no siblings has ``windowed_total <= own_lifetime`` ⇒ returns ``own_lifetime`` ⇒ identical
    to the prior per-strategy behavior (no regression).
    ``family_lifetime_effective=0`` (default) is byte-identical to the prior 2-arg behavior:
    ``max(own, windowed, 0) == max(own, windowed)`` when family=0."""
    return max(int(own_lifetime), int(windowed_total), int(family_lifetime_effective))
