from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from algua.backtest._constants import ANN
from algua.backtest.metrics import metrics_from_returns
from algua.backtest.walkforward import WalkForwardResult

# Funnel-level multiple-testing window (Wall A). Protected constant, not an agent-tunable knob
# (relaxing it would weaken the gate). Rolling window keeps the bar bounded; the wait-out-the-window
# trade-off is accepted and auditable via search_trials.created_at.
FUNNEL_WINDOW_DAYS = 90

# Minimum holdout sample (Wall C). A holdout with fewer observations is underpowered and fails
# closed — complements the 1/sqrt(T) haircut, which is ZERO at N=1. ~one trading quarter. Protected.
MIN_HOLDOUT_OBSERVATIONS = 63

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

# Multi-regime robustness (#221, Phase 3 Slice 4). Protected — relaxing weakens the gate.
N_REGIMES = 3                  # market-volatility tertiles (low/medium/high)
MIN_REGIME_OBSERVATIONS = 21   # per-regime power floor (underpowered regimes are dropped)
MIN_REGIME_SHARPE = 0.0        # relaxed per-regime Sharpe bar (paired with zero-vol drop)
MIN_REGIME_OVERLAP_BARS = 63   # min holdout dates with a valid trailing market-vol for the check
VOL_ROLLING_WINDOW = 21        # trailing bars for the benchmark realized-vol estimate

# Dominance-audit predeclaration (#221, Phase 3 Slice 4). CODEOWNERS-protected constants — these
# thresholds are committed HERE (before any audit data accumulates) so the Slice 5
# haircut-retirement audit cannot select them post-hoc. The retirement audit (Slice 5) filters
# decision_json rows where `phase3_component_mask` has all required bits set, then checks that no
# row where `haircut_would_have_blocked=True AND dsr_raw_N_pass=True` exceeds
# DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS over >= DOMINANCE_AUDIT_MIN_PROMOTIONS promotions
# across >= DOMINANCE_AUDIT_MIN_WINDOW_DAYS of real-traffic wall time. SHADOW/AUDIT ONLY in
# Slice 4 — these constants are read only by the Slice 5 audit, not by any gate pass/fail logic.
DOMINANCE_AUDIT_MIN_PROMOTIONS = 30        # min promotions in the audit window for a verdict
DOMINANCE_AUDIT_MIN_WINDOW_DAYS = 90       # min calendar days of traffic for the audit to bind
DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS = 0  # haircut-blocked-but-DSR-passed cases allowed: none

# Phase 3 component mask — bitmask recording which Phase 3 slices are active on a gate evaluation.
# Bits 0-4 = Slices 0-4; all five set = 0b11111 = 31. Stored as `phase3_component_mask` in
# decision_json so the Slice 5 audit can filter rows where all Phase 3 components are active.
# SHADOW/AUDIT ONLY — does not affect passed.
PHASE3_COMPONENT_MASK = 0b11111  # bits 0-4 = Phase 3 slices 0-4, all five active


class RegimeSlice(NamedTuple):
    """One volatility-tertile regime slice of the holdout period.

    ``dropped_reason`` is None when the slice is usable; ``"too_short"`` when it has fewer
    than the observation floor; ``"zero_vol"`` when ann_volatility==0.0 in the robustness
    check (set by ``regime_robustness_check``, not by ``regime_splits``).
    """

    regime_index: int
    returns: list[float]
    n_bars: int
    dropped_reason: str | None  # None | "too_short" | "zero_vol"


class RegimeRobustnessResult(NamedTuple):
    """Outcome of the per-regime robustness check."""

    passed: bool
    n_attempted: int
    n_surviving: int
    per_regime_sharpes: list[float | None]  # None for dropped regimes, float for survivors


def regime_splits(
    strategy_returns: list[float],
    strategy_dates: list[str],
    market_returns: list[float],
    market_dates: list[str],
    *,
    n_regimes: int,
    vol_window: int,
) -> tuple[list[RegimeSlice], int]:
    """Partition holdout dates into ``n_regimes`` volatility-tertile buckets.

    Algorithm:
    1. Build ``market_date -> market_return`` and ``strategy_date -> strategy_return`` dicts.
    2. Compute trailing-``vol_window`` realized vol of the MARKET series.  For each market
       date index ``i`` with ``i >= vol_window - 1``, the window is the ``vol_window`` returns
       ending at ``i`` (inclusive): ``market_returns[i - vol_window + 1 : i + 1]``.
       Vol = annualized std (ddof=1) of log(1+r) for r in the window.  Guard: if any
       ``1 + r <= 0`` the entire date is skipped (no valid log).  Dates with ``i < vol_window - 1``
       have no vol label and are excluded from the join.
    3. Inner-join: market dates that HAVE a valid vol label AND appear in ``strategy_dates``.
       ``overlap_n = len(joined_dates)``.  Empty join returns ``([], 0)``.
    4. Sort joined dates by ``(vol, date_string)`` ascending (deterministic tie-break on equal
       vols by date string → reproducible rank assignment).  Split the sorted list into
       ``n_regimes`` contiguous equal-sized groups using ``np.array_split``.
       Group 0 = lowest-vol tertile, …, group n_regimes-1 = highest.
    5. For each regime, collect STRATEGY returns for that regime's dates (in date order).
       ``RegimeSlice.dropped_reason`` is always ``None`` here; dropping is done by
       ``regime_robustness_check``.

    Returns ``(slices, overlap_n)``.  If ``overlap_n == 0`` returns ``([], 0)``.
    ``slices`` always has exactly ``n_regimes`` entries when ``overlap_n > 0``.
    """
    # Step 1: build lookup dicts
    strategy_map: dict[str, float] = dict(zip(strategy_dates, strategy_returns, strict=False))

    # Step 2: compute trailing-vol_window realized vol for each market date
    m_dates_list = list(market_dates)
    m_returns_list = list(market_returns)
    vol_labels: dict[str, float] = {}
    for i in range(len(m_dates_list)):
        if i < vol_window - 1:
            continue  # insufficient lookback
        window = m_returns_list[i - vol_window + 1 : i + 1]
        # Guard: all 1+r must be > 0 to take log
        if any(1.0 + r <= 0.0 for r in window):
            continue
        log_rets = [math.log(1.0 + r) for r in window]
        std = float(np.std(log_rets, ddof=1))
        vol_labels[m_dates_list[i]] = std * math.sqrt(ANN)

    # Step 3: inner-join with strategy_dates
    joined: list[tuple[str, float]] = [
        (d, vol_labels[d])
        for d in vol_labels
        if d in strategy_map
    ]
    overlap_n = len(joined)
    if overlap_n == 0:
        return ([], 0)

    # Step 4: assign tertiles by market-vol VALUE (quantile thresholds), not by equal-count rank.
    # This ensures a degenerate vol distribution collapses: if all vols are equal,
    # t1 == t2 == that value, ALL dates satisfy vol <= t1 (regime 0), and regimes 1 & 2 are
    # EMPTY (n_bars=0). Empty regimes are dropped by regime_robustness_check (too_short) ->
    # n_surviving < 2 -> passed=False. That is the intended fail-closed behavior for constant vol.
    #
    # For a genuine low/mid/high spread, dates distribute across all 3 tertiles normally.
    # Boundaries: regime 0: vol <= t1; regime 1: t1 < vol <= t2; regime 2: vol > t2.
    # Deterministic: equal vols always resolve to the same regime (<=/>).
    # The joined list is sorted by (vol, date) for order-independence before assignment.
    joined_sorted = sorted(joined, key=lambda x: (x[1], x[0]))  # sort by (vol, date) asc
    vols_array = np.array([x[1] for x in joined_sorted])
    t1 = float(np.quantile(vols_array, 1.0 / n_regimes))
    t2 = float(np.quantile(vols_array, 2.0 / n_regimes))

    # Build per-regime date lists (in vol-sorted order, which matches joined_sorted)
    regime_date_sets: list[list[str]] = [[] for _ in range(n_regimes)]
    for date_str, vol in joined_sorted:
        if vol <= t1:
            regime_date_sets[0].append(date_str)
        elif vol > t2:
            regime_date_sets[n_regimes - 1].append(date_str)
        else:
            regime_date_sets[1].append(date_str)

    # Step 5: build RegimeSlice for each regime — collect strategy returns in DATE order
    slices: list[RegimeSlice] = []
    for regime_idx, regime_dates_unordered in enumerate(regime_date_sets):
        # ISO date strings sort lexicographically = chronologically
        regime_dates = sorted(regime_dates_unordered)
        regime_returns = [strategy_map[d] for d in regime_dates]
        slices.append(RegimeSlice(
            regime_index=regime_idx,
            returns=regime_returns,
            n_bars=len(regime_returns),
            dropped_reason=None,
        ))
    return (slices, overlap_n)


def regime_robustness_check(
    slices: list[RegimeSlice],
    *,
    min_obs: int,
    min_sharpe: float,
) -> RegimeRobustnessResult:
    """Check per-regime Sharpe robustness across volatility-tertile slices.

    Drop rules (in order):
    - ``n_bars < min_obs`` → ``dropped_reason = "too_short"``; ``per_regime_sharpe = None``.
    - ``ann_volatility == 0.0`` → ``dropped_reason = "zero_vol"``; ``per_regime_sharpe = None``.
      A zero-vol regime gives sharpe=0.0 which would falsely pass ``MIN_REGIME_SHARPE=0.0``,
      so it must be dropped and NOT counted as a surviving pass.

    Passing rule:
    - ``n_surviving < 2`` → ``passed = False`` (cannot establish multi-regime evidence).
    - Else ``passed = all(sharpe >= min_sharpe for surviving regimes)``.

    ``per_regime_sharpes`` is aligned to the input ``slices`` list (None for dropped).
    """
    n_attempted = len(slices)
    per_regime_sharpes: list[float | None] = []
    surviving_sharpes: list[float] = []

    for s in slices:
        if s.n_bars < min_obs:
            per_regime_sharpes.append(None)
            continue
        m = metrics_from_returns(pd.Series(s.returns))
        if m["ann_volatility"] == 0.0:
            per_regime_sharpes.append(None)
            continue
        sharpe = m["sharpe"]
        per_regime_sharpes.append(sharpe)
        surviving_sharpes.append(sharpe)

    n_surviving = len(surviving_sharpes)
    if n_surviving < 2:
        passed = False
    else:
        passed = all(sh >= min_sharpe for sh in surviving_sharpes)

    return RegimeRobustnessResult(
        passed=passed,
        n_attempted=n_attempted,
        n_surviving=n_surviving,
        per_regime_sharpes=per_regime_sharpes,
    )


# LORD++ FDR accounting layer (#220, Phase 2). Protected constants — relaxing them weakens the gate.
# FDR here is an operating target (shared-holdout dependence breaks the formal guarantee); Phase 3
# (#221) adds dependence-aware calibration. The p-value fed here is 1 − dsr_confidence, which is
# P(SR_true ≤ SR*) — i.e. the FDR guarantee governs discoveries relative to the DSR null (SR > SR*),
# a STRONGER criterion than the simple null (SR > 0).
FDR_ALPHA = 0.05   # target FDR level
FDR_W0 = FDR_ALPHA / 2   # initial alpha-wealth (standard choice, Ramdas et al. 2017)
# γ-sequence truncation for normalization. 10 000 terms captures >99.99% of tail mass; α_t at
# positions > 10 000 uses γ_j=0 for j>10 000 (negligible, conservative).
FDR_GAMMA_TRUNCATION = 10_000


def _compute_lord_gamma(n: int) -> list[float]:
    """Normalized LORD++ γ weights for j=1..n.

    Raw: γ_j ∝ log(max(j, 2)) / (j · exp(√(log(max(j, 2)))))
    max(j, 2) is the standard practical variant per Ramdas et al. 2017 / the onlineFDR R package,
    ensuring γ_j > 0 for all j (log(max(1,2))=log(2)>0 handles j=1). Dividing by the truncated
    sum normalizes so Σγ_j ≤ 1.0 + machine-epsilon over the truncation window.
    """
    raw = [
        math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(max(j, 2)))))
        for j in range(1, n + 1)
    ]
    total = sum(raw)
    return [w / total for w in raw]


_LORD_GAMMA: list[float] = _compute_lord_gamma(FDR_GAMMA_TRUNCATION)


def lord_plus_plus_level(
    t: int,
    discovery_indices: Sequence[int],
    *,
    alpha: float = FDR_ALPHA,
    w0: float = FDR_W0,
) -> float:
    """LORD++ test level α_t (Ramdas et al. 2017 Biometrika 104:1).

    α_t = γ_t · w0 + (α − w0) · γ_{t−τ_1} + α · Σ_{j≥2} γ_{t−τ_j}

    where τ_1 < τ_2 < … are the 1-indexed positions of past discoveries (all strictly < t).
    α_t depends ONLY on past decisions — no circularity. Wealth is computed from the ledger
    rows on every call (not cached), mirroring pooled_trial_sharpe_var's fail-closed philosophy.

    The p-value fed here must be 1 − dsr_confidence (conversion at the caller), which equals
    P(SR_true ≤ SR*) — the DSR selection-inflated null. The FDR guarantee is over that null.

    Dry-spell behavior: with no discoveries, α_t = γ_t · w0 → 0 as t grows (expected
    LORD++ "alpha-death" in a long null streak; no floor, by design).

    Returns a CONSERVATIVE 0.0 on any degenerate input (t<1, non-finite alpha/w0, any
    discovery index ≥ t or < 1). 0.0 means p_t ≤ α_t can never be satisfied — only tightens.
    """
    if t < 1 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(w0) or w0 <= 0:
        return 0.0
    taus = sorted(int(tau) for tau in discovery_indices)
    if any(tau < 1 or tau >= t for tau in taus):
        return 0.0

    def _gamma(j: int) -> float:
        if j < 1 or j > len(_LORD_GAMMA):
            return 0.0
        return _LORD_GAMMA[j - 1]

    level = _gamma(t) * w0
    if taus:
        level += (alpha - w0) * _gamma(t - taus[0])
        for tau in taus[1:]:
            level += alpha * _gamma(t - tau)
    return level


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> candidate. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this
    min_holdout_observations: int = MIN_HOLDOUT_OBSERVATIONS  # Wall C: power floor, fails closed


def sharpe_haircut(n_combos: int, n_bars: int) -> float:
    """Deflated-Sharpe haircut: how many Sharpe units to add to the holdout-Sharpe bar after
    searching ``n_combos`` parameter combinations over a holdout of ``n_bars`` observations.

    Rationale (Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014). Selecting the best
    of N independent trials inflates the winner's Sharpe: the expected maximum of N standard
    normals grows like ``sqrt(2 * ln(N))`` standard errors. The standard error of a *per-period*
    Sharpe estimate over T observations is ``≈ 1/sqrt(T)``. So the per-period inflation is
    ``sqrt(2 * ln(N)) / sqrt(T)``.

    UNIT MATCH (critical): the holdout Sharpe in ``algua.backtest.metrics`` is ANNUALIZED —
    ``sharpe = (mean * ANN) / (std * sqrt(ANN)) = (mean/std) * sqrt(ANN)`` — i.e. the per-period
    Sharpe scaled by ``sqrt(ANN)``. The haircut must live in the same units as the threshold it
    raises, so the per-period standard-error term is scaled by ``sqrt(ANN)`` identically:

        haircut = sqrt(2 * ln(max(N, 1))) * sqrt(ANN) / sqrt(T)

    Invariants: 0 at N=1 (``ln(1) == 0`` — no penalty for a single pre-registered hypothesis),
    monotonically non-decreasing in N, and uses the holdout sample size T (not a constant).

    DEGENERATE HOLDOUT (T <= 0) FAILS CLOSED: a zero-length holdout carries no out-of-sample
    evidence, so the multiple-testing penalty is UNDEFINED, not zero. Returning ``inf`` lifts the
    effective holdout-Sharpe bar out of reach so the gate cannot pass on an empty holdout — the
    opposite of waiving the penalty (which returning 0.0 would silently do).

    NOTE: N is the RAW combo count with no deduplication. Correlated combos (e.g. neighboring
    parameter values that produce near-duplicate strategies) make the effective number of
    independent trials smaller, so ``sqrt(2*ln N)`` is an upper bound — the haircut errs on the
    strict side, which is intentional.
    """
    n = max(int(n_combos), 1)
    if n_bars <= 0:
        return math.inf
    return math.sqrt(2.0 * math.log(n)) * math.sqrt(ANN) / math.sqrt(n_bars)


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


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None
    breadth_provenance: str | None = None
    base_min_holdout_sharpe: float | None = None
    effective_min_holdout_sharpe: float | None = None
    own_lifetime_combos: int | None = None
    windowed_total_combos: int | None = None
    funnel_window_days: int | None = None
    pit_ok: bool | None = None
    pit_override: bool = False
    dsr_binding: bool = False
    dsr_confidence: float | None = None
    dsr_skip_reason: str | None = None
    dsr_n_trials: int | None = None
    dsr_trial_sr_var_ann: float | None = None
    dsr_t: int | None = None
    dsr_skew: float | None = None
    dsr_raw_kurtosis: float | None = None
    # Funnel-wide dispersion floor audit (#221, Phase 3 Slice 0). Populated when dsr_binding.
    dsr_funnel_floor_var_ann: float | None = None
    dsr_funnel_floor_n_strategies: int | None = None
    dsr_funnel_floor_n_total_rows: int | None = None
    # LORD++ FDR accounting (#220, Phase 2). Populated by run_gate() AFTER the atomic store
    # call (α_t is only known inside the write transaction). Non-binding rows leave all fdr_*
    # fields at their defaults (False/None); fdr_skip_reason explains why FDR was omitted.
    fdr_binding: bool = False
    fdr_p_value: float | None = None
    fdr_alpha_level: float | None = None
    fdr_test_index: int | None = None
    fdr_rejected: bool | None = None
    fdr_skip_reason: str | None = None
    # Returns-persistence audit (#221 Slice 1). Non-binding: True iff run_gate wrote a
    # holdout_returns row for this evaluation. False for pre-Slice-1 promotions (omit-not-fail).
    returns_available: bool = False
    # Serial-dependence bootstrap audit (#221 Slice 2). Binding only when dsr_binding AND
    # bootstrap_binding are both True (gates computes effective_bootstrap_binding = dsr_binding
    # AND bootstrap_binding). The caller (promotion.py) sets bootstrap_binding only when measured
    # breadth AND the OOS vector is present; gates binds on dsr_binding AND bootstrap_binding.
    # Non-binding: all dsr_bootstrap_* fields remain False/None (omit-not-fail).
    dsr_bootstrap_binding: bool = False
    dsr_bootstrap_lower: float | None = None
    dsr_bootstrap_seed: int | None = None
    dsr_bootstrap_b: int | None = None
    dsr_bootstrap_block_len: int | None = None
    # Effective independent trials audit (#221 Slice 3). SHADOW-ONLY: recorded, never fed into the
    # binding dsr_confidence (which keeps n_trials = raw N). Populated by promotion.run_gate.
    dsr_n_eff: int | None = None
    dsr_rho_bar: float | None = None
    dsr_n_siblings: int | None = None
    # Multi-regime robustness audit (#221 Slice 4). Populated by evaluate_gate when the check
    # binds (holdout_returns + market_returns present, overlap >= MIN_REGIME_OVERLAP_BARS).
    # When omitted (unavailable / insufficient overlap): regime_method explains why.
    regime_method: str | None = None          # "vol_tertile"|"unavailable"|"insufficient_overlap"
    n_regimes_attempted: int | None = None    # total regime slices fed to robustness_check
    n_regimes_surviving: int | None = None    # regimes that cleared power + vol floors
    per_regime_sharpes: list[float | None] | None = None  # per-slice Sharpe; None for dropped
    regime_robustness_binding: bool = False   # True iff the check was appended to checks
    # Dominance-audit shadow fields (#221 Slice 4). SHADOW/AUDIT ONLY — not in checks, not in
    # passed. Recorded on every evaluate_gate call so the Slice 5 retirement audit can accumulate
    # real-traffic statistics with pre-committed thresholds.
    # haircut_would_have_blocked: True iff the holdout Sharpe cleared the BASE bar but failed the
    #   haircut-inflated bar (the haircut is what blocked an otherwise-passing holdout). When the
    #   effective bar is +inf (degenerate zero-length holdout), True iff the base bar passed.
    # phase3_component_mask: PHASE3_COMPONENT_MASK (0b11111 = 31) — all five Slice 0-4 components
    #   active. Allows the audit to filter rows where the full Phase 3 stack was live.
    haircut_would_have_blocked: bool = False
    phase3_component_mask: int | None = None

    def to_dict(self) -> dict[str, Any]:
        # A degenerate holdout drives the effective bar to inf (fail-closed); null it so the
        # payload stays JSON-clean, mirroring how non-finite check values are nulled.
        eff = self.effective_min_holdout_sharpe

        def _f(x: float | None) -> float | None:
            return x if x is None or math.isfinite(x) else None

        return {
            "passed": self.passed,
            "checks": self.checks,
            "n_combos": self.n_combos,
            "breadth_provenance": self.breadth_provenance,
            "base_min_holdout_sharpe": self.base_min_holdout_sharpe,
            "effective_min_holdout_sharpe": (
                eff if eff is None or math.isfinite(eff) else None
            ),
            "own_lifetime_combos": self.own_lifetime_combos,
            "windowed_total_combos": self.windowed_total_combos,
            "funnel_window_days": self.funnel_window_days,
            "pit_ok": self.pit_ok,
            "pit_override": self.pit_override,
            "dsr_binding": self.dsr_binding,
            "dsr_confidence": _f(self.dsr_confidence),
            "dsr_skip_reason": self.dsr_skip_reason,
            "dsr_n_trials": self.dsr_n_trials,
            "dsr_trial_sr_var_ann": _f(self.dsr_trial_sr_var_ann),
            "dsr_t": self.dsr_t,
            "dsr_skew": _f(self.dsr_skew),
            "dsr_raw_kurtosis": _f(self.dsr_raw_kurtosis),
            "dsr_funnel_floor_var_ann": _f(self.dsr_funnel_floor_var_ann),
            "dsr_funnel_floor_n_strategies": self.dsr_funnel_floor_n_strategies,
            "dsr_funnel_floor_n_total_rows": self.dsr_funnel_floor_n_total_rows,
            "fdr_binding": self.fdr_binding,
            "fdr_p_value": _f(self.fdr_p_value),
            "fdr_alpha_level": _f(self.fdr_alpha_level),
            "fdr_test_index": self.fdr_test_index,
            "fdr_rejected": self.fdr_rejected,
            "fdr_skip_reason": self.fdr_skip_reason,
            "returns_available": self.returns_available,
            "dsr_bootstrap_binding": self.dsr_bootstrap_binding,
            "dsr_bootstrap_lower": _f(self.dsr_bootstrap_lower),
            "dsr_bootstrap_seed": self.dsr_bootstrap_seed,
            "dsr_bootstrap_b": self.dsr_bootstrap_b,
            "dsr_bootstrap_block_len": self.dsr_bootstrap_block_len,
            "dsr_n_eff": self.dsr_n_eff,
            "dsr_rho_bar": _f(self.dsr_rho_bar),
            "dsr_n_siblings": self.dsr_n_siblings,
            "regime_method": self.regime_method,
            "n_regimes_attempted": self.n_regimes_attempted,
            "n_regimes_surviving": self.n_regimes_surviving,
            "per_regime_sharpes": (
                [None if (x is None or not math.isfinite(x)) else x
                 for x in self.per_regime_sharpes]
                if self.per_regime_sharpes is not None else None
            ),
            "regime_robustness_binding": self.regime_robustness_binding,
            # Dominance-audit shadow fields (#221 Slice 4)
            "haircut_would_have_blocked": self.haircut_would_have_blocked,
            "phase3_component_mask": self.phase3_component_mask,
        }


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
}


@dataclass(frozen=True)
class GateSpec:
    """Declarative description of one gate check (#40).

    Adding a gate is appending a spec here — no edits to the evaluation loop. `source` and
    `metric_key` locate the value on the WalkForwardResult; `threshold_attr` names the
    GateCriteria field; `op` is the comparison operator.
    """

    name: str
    source: str  # "holdout" -> wf.holdout_metrics, "stability" -> wf.stability
    metric_key: str
    threshold_attr: str
    op: str


GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("holdout_sharpe", "holdout", "sharpe", "min_holdout_sharpe", ">="),
    GateSpec("holdout_return", "holdout", "total_return", "min_holdout_return", ">"),
    GateSpec("pct_positive_windows", "stability", "pct_positive_windows",
             "min_pct_positive_windows", ">="),
    GateSpec("min_window_sharpe", "stability", "min_sharpe", "min_window_sharpe", ">="),
    GateSpec("min_holdout_observations", "holdout", "n_bars", "min_holdout_observations", ">="),
)


_HOLDOUT_SHARPE_SPEC = "holdout_sharpe"


def evaluate_gate(
    wf: WalkForwardResult,
    criteria: GateCriteria,
    *,
    n_combos: int | None = None,
    breadth_provenance: str | None = None,
    pit_ok: bool,
    allow_non_pit: bool = False,
    own_lifetime_combos: int | None = None,
    windowed_total_combos: int | None = None,
    funnel_window_days: int | None = None,
    dsr_binding: bool = False,
    dsr_trial_var_ann: float | None = None,
    dsr_funnel_floor_var_ann: float | None = None,
    dsr_funnel_floor_n_strategies: int | None = None,
    dsr_funnel_floor_n_total_rows: int | None = None,
    bootstrap_binding: bool = False,
    bootstrap_lower_confidence: float | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_b: int | None = None,
    bootstrap_block_len: int | None = None,
    market_returns: tuple[list[float], list[str]] | None = None,
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects.

    Driven by GATE_SPECS so the metric dict and the gate stay in sync without hand editing.

    The holdout-Sharpe threshold is DEFLATED by ``sharpe_haircut(n_combos, T)`` where T is the
    holdout sample size (``wf.holdout_metrics["n_bars"]``). This is the multiple-testing defense:
    the more combos the agent searched, the higher the bar the selected strategy must clear on the
    untouched holdout. At N=1 (or no breadth) the haircut is 0 and the effective bar equals the
    base. The other gate checks are left untouched. ``breadth_provenance`` ("measured"/"declared")
    is carried into the decision for the audit trail; it does not change the math.
    """
    sources = {"holdout": wf.holdout_metrics, "stability": wf.stability}
    base_holdout_sharpe = float(criteria.min_holdout_sharpe)
    haircut = (
        sharpe_haircut(n_combos, int(wf.holdout_metrics["n_bars"]))
        if n_combos is not None
        else 0.0
    )
    effective_holdout_sharpe = base_holdout_sharpe + haircut

    checks: list[dict[str, Any]] = []
    for spec in GATE_SPECS:
        value = float(sources[spec.source][spec.metric_key])
        threshold = float(getattr(criteria, spec.threshold_attr))
        if spec.name == _HOLDOUT_SHARPE_SPEC:
            threshold = effective_holdout_sharpe
        # A non-finite EFFECTIVE THRESHOLD means a degenerate (zero-length) holdout drove the
        # haircut to inf: there is no out-of-sample evidence, so the check FAILS CLOSED — never a
        # pass — and the inf threshold is nulled out of the payload to keep it JSON-clean.
        if not math.isfinite(threshold):
            checks.append({"name": spec.name, "value": None, "threshold": None,
                           "op": spec.op, "passed": False})
            continue
        # A non-finite metric (inf trivially clears >=/>, NaN is never a real result) is a
        # gate failure, never a pass — and is never recorded as a NaN/inf in the payload.
        if not math.isfinite(value):
            checks.append({"name": spec.name, "value": None, "threshold": threshold,
                           "op": spec.op, "passed": False})
            continue
        passed = _OPS[spec.op](value, threshold)
        checks.append({"name": spec.name, "value": value, "threshold": threshold,
                       "op": spec.op, "passed": bool(passed)})
    # PIT precondition (Wall B): boolean, not a metric comparison. pit_ok is computed by the
    # protected promotion orchestrator (presence + coverage). Non-PIT fails closed unless a human
    # passed allow_non_pit (recorded as an audited override).
    pit_passed = bool(pit_ok or allow_non_pit)
    pit_override = bool((not pit_ok) and allow_non_pit)
    checks.append({"name": "pit_required", "passed": pit_passed,
                   "pit_ok": bool(pit_ok), "override": "non_pit" if pit_override else None})
    # DSR evidence (#211): a tighten-only AND-check, appended ONLY when binding (measured trial
    # dispersion is available). When not binding it is omitted entirely so `passed` is unchanged.
    # Unit conversion lives here: holdout Sharpe and trial variance are ANNUALIZED; DSR per-period.
    dsr_conf: float | None = None
    dsr_skip_reason: str | None = None
    n_for_dsr = n_combos if n_combos is not None else 1
    t_hold = int(wf.holdout_metrics["n_bars"])
    skew = float(wf.holdout_metrics.get("skewness", 0.0))
    raw_kurt = float(wf.holdout_metrics.get("kurtosis", 3.0))
    sr_obs_ann = float(wf.holdout_metrics["sharpe"])
    # The bootstrap check is only armed when BOTH dsr_binding AND bootstrap_binding are True.
    # The caller (promotion.py) sets bootstrap_binding only when measured breadth AND the OOS
    # vector is present; gates binds on dsr_binding AND bootstrap_binding.
    # Using effective_bootstrap_binding throughout keeps the audit state self-consistent: if
    # dsr_binding=False the bootstrap check is never appended and all bootstrap audit fields
    # remain None/False, regardless of what the caller passed for bootstrap_binding.
    effective_bootstrap_binding = bool(dsr_binding and bootstrap_binding)
    if dsr_binding:
        var_pp = (dsr_trial_var_ann / ANN) if dsr_trial_var_ann is not None else None
        floor_pp = (
            dsr_funnel_floor_var_ann / ANN
            if dsr_funnel_floor_var_ann is not None and math.isfinite(dsr_funnel_floor_var_ann)
            else None
        )
        if var_pp is not None and math.isfinite(var_pp):
            dsr_conf = dsr_confidence(
                sr_obs_ann / math.sqrt(ANN), t_hold, skew, raw_kurt, n_for_dsr, var_pp,
                funnel_floor_var_per_period=floor_pp)
        passed_dsr = dsr_conf is not None and dsr_conf >= (1.0 - DSR_ALPHA)
        dsr_value = dsr_conf if (dsr_conf is not None and math.isfinite(dsr_conf)) else None
        checks.append({"name": "dsr_evidence", "value": dsr_value,
                       "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(passed_dsr)})
        if dsr_conf is None:
            # measured sweep exists but stats missing -> fail closed
            dsr_skip_reason = "no_dispersion"
        if effective_bootstrap_binding:
            boot_pass = (bootstrap_lower_confidence is not None
                         and bootstrap_lower_confidence >= (1.0 - DSR_ALPHA))
            boot_value = (bootstrap_lower_confidence
                          if (bootstrap_lower_confidence is not None
                              and math.isfinite(bootstrap_lower_confidence)) else None)
            checks.append({"name": "dsr_bootstrap", "value": boot_value,
                           "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(boot_pass)})
    else:
        dsr_skip_reason = "no_measured_dispersion"

    # Multi-regime robustness (#221 Slice 4): a tighten-only AND-check, appended ONLY when the
    # market series is available AND wf.holdout_returns is present AND overlap is sufficient.
    # When omitted (unavailable / insufficient_overlap), checks is byte-identical to today so
    # every existing promotion test (no market_returns) is unchanged.
    # Binding is INDEPENDENT of dsr_binding — purely a holdout-robustness check.
    regime_method: str | None = None
    n_regimes_attempted: int | None = None
    n_regimes_surviving: int | None = None
    per_regime_sharpes_out: list[float | None] | None = None
    regime_robustness_binding = False

    holdout_ret_vec = wf.holdout_returns  # tuple[list[float], list[str]] | None
    if holdout_ret_vec is None or market_returns is None:
        regime_method = "unavailable"
    else:
        strat_rets_list, strat_dates_list = holdout_ret_vec
        mkt_rets_list, mkt_dates_list = market_returns
        slices, overlap_n = regime_splits(
            strat_rets_list,
            strat_dates_list,
            mkt_rets_list,
            mkt_dates_list,
            n_regimes=N_REGIMES,
            vol_window=VOL_ROLLING_WINDOW,
        )
        if overlap_n < MIN_REGIME_OVERLAP_BARS:
            regime_method = "insufficient_overlap"
        else:
            res = regime_robustness_check(slices, min_obs=MIN_REGIME_OBSERVATIONS,
                                          min_sharpe=MIN_REGIME_SHARPE)
            checks.append({"name": "regime_robustness", "value": None,
                           "threshold": MIN_REGIME_SHARPE, "op": ">=",
                           "passed": bool(res.passed)})
            regime_method = "vol_tertile"
            regime_robustness_binding = True
            n_regimes_attempted = res.n_attempted
            n_regimes_surviving = res.n_surviving
            per_regime_sharpes_out = res.per_regime_sharpes

    # Dominance-audit shadow field (#221 Slice 4): did the haircut alone block an otherwise-passing
    # holdout? True iff the holdout Sharpe clears the BASE bar but fails the haircut-inflated bar.
    # The haircut is still BINDING (this is shadow recording only). When the effective bar is +inf
    # (degenerate zero-length holdout drove the haircut to inf), the haircut blocks everything the
    # base bar passed, so True iff the base bar passed. SHADOW/AUDIT ONLY — does not affect passed.
    haircut_would_have_blocked = bool(
        sr_obs_ann >= base_holdout_sharpe
        and (not math.isfinite(effective_holdout_sharpe) or sr_obs_ann < effective_holdout_sharpe)
    )

    return GateDecision(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        n_combos=n_combos,
        breadth_provenance=breadth_provenance,
        base_min_holdout_sharpe=base_holdout_sharpe,
        effective_min_holdout_sharpe=effective_holdout_sharpe,
        own_lifetime_combos=own_lifetime_combos,
        windowed_total_combos=windowed_total_combos,
        funnel_window_days=funnel_window_days,
        pit_ok=bool(pit_ok),
        pit_override=pit_override,
        dsr_binding=bool(dsr_binding),
        dsr_confidence=dsr_conf,
        dsr_skip_reason=dsr_skip_reason,
        dsr_n_trials=(n_for_dsr if dsr_binding else None),
        dsr_trial_sr_var_ann=(dsr_trial_var_ann if dsr_binding else None),
        dsr_t=(t_hold if dsr_binding else None),
        dsr_skew=(skew if dsr_binding else None),
        dsr_raw_kurtosis=(raw_kurt if dsr_binding else None),
        dsr_funnel_floor_var_ann=(dsr_funnel_floor_var_ann if dsr_binding else None),
        dsr_funnel_floor_n_strategies=(dsr_funnel_floor_n_strategies if dsr_binding else None),
        dsr_funnel_floor_n_total_rows=(dsr_funnel_floor_n_total_rows if dsr_binding else None),
        dsr_bootstrap_binding=effective_bootstrap_binding,
        dsr_bootstrap_lower=(bootstrap_lower_confidence if effective_bootstrap_binding else None),
        dsr_bootstrap_seed=(bootstrap_seed if effective_bootstrap_binding else None),
        dsr_bootstrap_b=(bootstrap_b if effective_bootstrap_binding else None),
        dsr_bootstrap_block_len=(bootstrap_block_len if effective_bootstrap_binding else None),
        regime_method=regime_method,
        n_regimes_attempted=n_regimes_attempted,
        n_regimes_surviving=n_regimes_surviving,
        per_regime_sharpes=per_regime_sharpes_out,
        regime_robustness_binding=regime_robustness_binding,
        haircut_would_have_blocked=haircut_would_have_blocked,
        phase3_component_mask=PHASE3_COMPONENT_MASK,
    )
