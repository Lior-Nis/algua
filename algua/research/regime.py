"""Volatility-tertile regime robustness + single-factor CAPM idiosyncratic-alpha screen
(extracted from gates.py, #335).

Pure-maths module: imports the backtest annualization constant, the return-metrics helper, and the
shared holdout-power floor from ``algua.research._constants``. No import of ``gates`` (which
composes this module), so the dependency graph stays acyclic.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from algua.backtest._constants import ANN
from algua.backtest.metrics import metrics_from_returns
from algua.research._constants import MIN_HOLDOUT_OBSERVATIONS

# Multi-regime robustness (#221, Phase 3 Slice 4). Protected — relaxing weakens the gate.
N_REGIMES = 3                  # market-volatility tertiles (low/medium/high)
MIN_REGIME_OBSERVATIONS = 21   # per-regime power floor (underpowered regimes are dropped)
MIN_REGIME_SHARPE = 0.0        # relaxed per-regime Sharpe bar (paired with the vol-floor drop)
# Annualized-vol floor: a regime at/below this is "effectively constant" and is DROPPED (not
# counted as a surviving pass). An exact `== 0.0` test failed open (#248): a constant-but-NONZERO
# return series produces a catastrophic-cancellation residual vol (~1e-18 — not exactly 0.0) and a
# Sharpe ~1e16 that trivially clears MIN_REGIME_SHARPE=0.0. The floor sits orders of magnitude above
# float64 cancellation noise and far below any real strategy vol (>=~1e-2). Protected — raising it
# weakens the gate. `not (vol > MIN_REGIME_VOL)` also drops NaN vol (another degenerate case).
MIN_REGIME_VOL = 1e-9
MIN_REGIME_OVERLAP_BARS = 63   # min holdout dates with a valid trailing market-vol for the check
VOL_ROLLING_WINDOW = 21        # trailing bars for the benchmark realized-vol estimate

# Market-beta / idiosyncratic-alpha screen (#328). Protected — relaxing weakens the gate.
# All gate Sharpes are RAW (risk_free=0, no benchmark subtraction), so a persistently net-long or
# LEVERED-market-beta book in a bull market posts a HIGH raw Sharpe with ~ZERO true alpha and is
# promoted. This tighten-only AND-check regresses the strategy's OOS holdout returns on the SAME
# PIT equal-weighted cross-sectional `market_returns` benchmark the regime check already reuses
# (single-factor CAPM: r_strat = alpha + beta*r_mkt + eps) and requires the ANNUALIZED
# idiosyncratic APPRAISAL RATIO (residual alpha / residual vol) to clear a floor. A pure/levered
# beta book has alpha~=0 and residual~=0 -> appraisal~=0 -> FAILS; a market-neutral genuine-alpha
# book has beta~=0 and appraisal ~= its raw Sharpe -> PASSES. Beta is estimated ONLY over the
# date-joined holdout window (no look-ahead). SCOPE: this nets out MARKET beta only, NOT style
# factors (size/value/momentum) — the platform has no factor-return series (deferred follow-up).
IR_MIN_OVERLAP_BARS = MIN_HOLDOUT_OBSERVATIONS   # 63; min date-joined holdout bars to bind
# Pragmatic annualized idiosyncratic-IR floor, NOT a significance test (statistical significance +
# multiple-testing are owned by the orthogonal DSR + Sharpe-haircut AND-checks). 0.3 matches the
# platform's existing 0.3 Sharpe-floor family (the forward gate uses max(0.5*holdout, 0.3)); it sits
# BELOW the 0.5 raw holdout-Sharpe bar so a genuine low-beta alpha still clears it, but far above
# the ~0 idiosyncratic appraisal of pure/levered market beta. Over a 63-bar holdout the estimate
# carries material sampling error — a structural beta-dominance screen, not an alpha significance
# test. Protected — lowering it weakens the gate.
IR_MIN_APPRAISAL_RATIO = 0.3
# Annualized-vol floor for BOTH the market series (beta denominator) and the residual (appraisal
# denominator). Identical to MIN_REGIME_VOL: ~10 orders of magnitude below any real strategy vol, so
# it only trips on a numerically-degenerate (near-constant market, or perfectly-explained residual)
# series — which then fails the check CLOSED (an armed-but-unusable regression is not a pass).
IR_MIN_VOL = MIN_REGIME_VOL


class RegimeSlice(NamedTuple):
    """One volatility-tertile regime slice of the holdout period.

    ``dropped_reason`` is None when the slice is usable; ``"too_short"`` when it has fewer
    than the observation floor; ``"zero_vol"`` when ann_volatility <= MIN_REGIME_VOL (effectively
    constant, or NaN) in the robustness check (set by ``regime_robustness_check``, not
    ``regime_splits``).
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
    4. Assign tertiles by market-vol VALUE: thresholds ``t1 = quantile(vols, 1/3)`` and
       ``t2 = quantile(vols, 2/3)``; regime 0 if ``vol <= t1``, regime 2 if ``vol > t2``, else
       regime 1.  Value-based (not equal-count rank) so a degenerate/constant vol distribution
       COLLAPSES — all dates fall into one tertile and the others are empty → dropped → fail-closed.
       Group 0 = lowest-vol tertile, …, group n_regimes-1 = highest. Deterministic (no RNG).
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
        # Guard: every return must be FINITE and have 1+r > 0 to take log. A non-finite return
        # (NaN/inf) must NOT produce a vol label — `1+r <= 0` is False for NaN, so check finiteness
        # explicitly, else a NaN vol label would count toward overlap and poison np.quantile.
        if any((not math.isfinite(r)) or (1.0 + r <= 0.0) for r in window):
            continue
        log_rets = [math.log(1.0 + r) for r in window]
        std = float(np.std(log_rets, ddof=1))
        vol = std * math.sqrt(ANN)
        if not math.isfinite(vol):
            continue  # fail-closed: a non-finite vol label is no label at all
        vol_labels[m_dates_list[i]] = vol

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
    - ``ann_volatility <= MIN_REGIME_VOL`` (effectively constant) or NaN → ``per_regime_sharpe =
      None`` (the Sharpe is meaningless — a near-constant series' Sharpe explodes to ~±1e16 — and is
      NEVER recorded; #248/#268). Such a regime is NOT a surviving pass. BUT its return direction is
      still real: a degenerate regime that LOST money (``sum(returns) < 0``) forces ``passed=False``
      (``degenerate_loss``) — dropping it would loosen the gate vs the old ``== 0.0`` code, whose
      huge-negative Sharpe failed the AND-check. A flat/favorable degenerate regime is just dropped.

    Passing rule:
    - any degenerate LOSING regime → ``passed = False`` (a real per-regime loss, fail closed).
    - ``n_surviving < 2`` → ``passed = False`` (cannot establish multi-regime evidence).
    - Else ``passed = all(sharpe >= min_sharpe for surviving regimes)``.

    ``per_regime_sharpes`` is aligned to the input ``slices`` list (None for dropped).
    """
    n_attempted = len(slices)
    per_regime_sharpes: list[float | None] = []
    surviving_sharpes: list[float] = []
    degenerate_loss = False

    for s in slices:
        if s.n_bars < min_obs:
            per_regime_sharpes.append(None)
            continue
        returns = pd.Series(s.returns)
        m = metrics_from_returns(returns)
        # An effectively-constant (or NaN-vol) regime has no reliable Sharpe — a catastrophic-
        # cancellation residual vol (~1e-18 != 0.0, where `== 0.0` failed open, #248) yields a
        # ~±1e16 Sharpe. `not (vol > floor)` catches that and NaN. We never record that explosive
        # Sharpe (it would feed #268) — per_regime_sharpe is None. But the regime's RETURN direction
        # is still real: a degenerate regime that LOST money is a true robustness failure and must
        # force a fail — dropping it would LOOSEN the gate vs the old code (whose huge-negative
        # Sharpe failed the AND-check). The loss proxy uses the SAME NaN-cleaned series metrics does
        # (a raw sum() would be NaN for a NaN-bearing slice and silently skip the check, #248 r2). A
        # flat/favorable degenerate regime carries no signal and is simply dropped.
        if not (m["ann_volatility"] > MIN_REGIME_VOL):
            if returns.dropna().sum() < 0.0:
                degenerate_loss = True
            per_regime_sharpes.append(None)
            continue
        sharpe = m["sharpe"]
        per_regime_sharpes.append(sharpe)
        surviving_sharpes.append(sharpe)

    n_surviving = len(surviving_sharpes)
    if n_surviving < 2 or degenerate_loss:
        passed = False
    else:
        passed = all(sh >= min_sharpe for sh in surviving_sharpes)

    return RegimeRobustnessResult(
        passed=passed,
        n_attempted=n_attempted,
        n_surviving=n_surviving,
        per_regime_sharpes=per_regime_sharpes,
    )


class InformationRatioResult(NamedTuple):
    """Outcome of the single-factor CAPM idiosyncratic-alpha screen (#328).

    ``degenerate`` True means the regression was armed (enough overlap) but UNUSABLE (constant
    market, zero residual, or non-finite) — the caller fails the check CLOSED. Numeric fields are
    None when unavailable/degenerate; ``market_beta`` / ``alpha_ann`` are still populated when
    computable (audit visibility) even if a later stage is degenerate.
    """

    overlap_n: int
    market_beta: float | None
    alpha_ann: float | None          # annualized intercept (residual alpha)
    residual_vol_ann: float | None   # annualized idiosyncratic (residual) volatility
    appraisal_ratio: float | None    # annualized alpha_ann / residual_vol_ann
    degenerate: bool


def information_ratio(
    strategy_returns: list[float],
    strategy_dates: list[str],
    market_returns: list[float],
    market_dates: list[str],
) -> InformationRatioResult:
    """Single-factor CAPM idiosyncratic appraisal ratio of the strategy vs the market benchmark.

    Inner-joins the two series by ISO date (PIT: both are period returns keyed by the SAME trading
    calendar; the strategy leg is the OOS holdout and the market leg is the as-of-member-masked PIT
    benchmark, so beta is estimated ONLY over the joined holdout window — no look-ahead), then OLS-
    regresses strategy on market:  ``r_strat_t = alpha + beta * r_mkt_t + eps_t``.

        market_beta     = cov(strat, mkt) / var(mkt)                    (systematic exposure)
        alpha_pp        = mean(strat) - beta * mean(mkt)                (per-period intercept)
        resid_var_pp    = SSR / (n - 2)                                 (OLS residual df, 2 params)
        appraisal_ratio = (alpha_pp / sqrt(resid_var_pp)) * sqrt(ANN)   (annualized; = alpha_ann /
                                                                         residual_vol_ann)

    Both legs are RAW returns (risk_free=0, the platform-wide convention #44) with no modeled cash/
    carry leg, so the intercept is idiosyncratic alpha, not risk-free yield.

    ``degenerate=True`` (caller fails the check closed) when the regression is UNUSABLE: fewer than
    3 joined bars, a (near-)constant market (annualized market vol <= IR_MIN_VOL — beta undefined),
    a (near-)zero residual (annualized residual vol <= IR_MIN_VOL — a perfectly-explained series
    carries no measurable idiosyncratic alpha, appraisal explodes), or any non-finite intermediate.
    Both vol floors are ANNUALIZED and identical to MIN_REGIME_VOL — far below any real strategy
    vol, so only numerically-degenerate series trip them.

    ``overlap_n`` is the number of date-joined bars (0 on empty join). Pure function; no I/O.

    RAISES ``ValueError`` on a CORRUPT armed input — a ragged ``(returns, dates)`` leg or duplicate
    dates within a leg. Upstream always pairs returns/dates equal-length with unique trading-day
    dates (walk_forward; promotion.py asserts it), so this is a regression backstop: silently
    truncating (``zip`` short) or collapsing duplicates (``dict``) could shrink ``overlap_n`` below
    the bind floor and DOWNGRADE a binding check to a silent "omit" — the opposite of fail-closed.
    """
    # Ragged (returns, dates) -> fail LOUD, never truncate to a smaller (silently-omitted) overlap.
    strat_map: dict[str, float] = dict(zip(strategy_dates, strategy_returns, strict=True))
    mkt_map: dict[str, float] = dict(zip(market_dates, market_returns, strict=True))
    # Duplicate dates within a leg are dict-collapsed above (trading-day dates are unique by
    # construction); detect the collapse and fail loud rather than join on a silent subset.
    if len(strat_map) != len(strategy_dates) or len(mkt_map) != len(market_dates):
        raise ValueError(
            "information_ratio: duplicate dates in strategy/market series (corrupt gate input)"
        )
    joined_dates = sorted(set(strat_map) & set(mkt_map))
    overlap_n = len(joined_dates)
    # Fewer than 3 joined bars cannot fit a 2-parameter OLS (n - 2 <= 0). Degenerate only when the
    # caller would otherwise bind; the caller's IR_MIN_OVERLAP_BARS (63) floor makes this defensive.
    if overlap_n < 3:
        return InformationRatioResult(overlap_n, None, None, None, None, degenerate=overlap_n > 0)

    strat = np.array([strat_map[d] for d in joined_dates], dtype=float)
    mkt = np.array([mkt_map[d] for d in joined_dates], dtype=float)
    if not (bool(np.all(np.isfinite(strat))) and bool(np.all(np.isfinite(mkt)))):
        return InformationRatioResult(overlap_n, None, None, None, None, degenerate=True)

    n = overlap_n
    mkt_mean = float(mkt.mean())
    strat_mean = float(strat.mean())
    mkt_centered = mkt - mkt_mean
    mkt_ss = float(np.dot(mkt_centered, mkt_centered))          # sum of squares (var * (n-1))
    mkt_var_pp = mkt_ss / (n - 1)
    mkt_vol_ann = math.sqrt(mkt_var_pp) * math.sqrt(ANN) if mkt_var_pp > 0.0 else 0.0
    # Constant / near-constant market -> beta is undefined -> fail closed.
    if not math.isfinite(mkt_var_pp) or mkt_var_pp <= 0.0 or not (mkt_vol_ann > IR_MIN_VOL):
        return InformationRatioResult(overlap_n, None, None, None, None, degenerate=True)

    beta = float(np.dot(mkt_centered, strat - strat_mean) / mkt_ss)
    alpha_pp = strat_mean - beta * mkt_mean
    alpha_ann = alpha_pp * ANN
    resid = strat - (alpha_pp + beta * mkt)
    ssr = float(np.dot(resid, resid))
    resid_var_pp = ssr / (n - 2)
    resid_vol_ann = math.sqrt(resid_var_pp) * math.sqrt(ANN) if resid_var_pp > 0.0 else 0.0

    beta_out = beta if math.isfinite(beta) else None
    alpha_out = alpha_ann if math.isfinite(alpha_ann) else None
    # Perfectly-explained / degenerate residual -> no measurable idiosyncratic alpha -> fail closed.
    # Beta and alpha are still surfaced for the audit trail.
    if not math.isfinite(resid_var_pp) or resid_var_pp <= 0.0 or not (resid_vol_ann > IR_MIN_VOL):
        return InformationRatioResult(overlap_n, beta_out, alpha_out, None, None, degenerate=True)

    appraisal = (alpha_pp / math.sqrt(resid_var_pp)) * math.sqrt(ANN)
    if beta_out is None or alpha_out is None or not math.isfinite(appraisal):
        return InformationRatioResult(overlap_n, beta_out, alpha_out, resid_vol_ann, None,
                                      degenerate=True)
    return InformationRatioResult(
        overlap_n, beta_out, alpha_out, resid_vol_ann, float(appraisal), degenerate=False
    )
