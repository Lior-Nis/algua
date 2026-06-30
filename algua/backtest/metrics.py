"""The ONE canonical metrics module for the backtest core (#37).

Both the single-pass backtest (`engine.run`) and walk-forward (`walkforward.walk_forward`)
compute their return-based metrics here, so a "Sharpe" or "max drawdown" means the same
thing everywhere. Metrics are a registry of named pure functions over a return series
(#40), so adding a metric is a registration — not an edit to a core loop.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats as _stats

from algua.backtest._constants import ANN, REBALANCE_EPS

# A metric is a pure function from a (already-dropna'd) return series to a float.
MetricFn = Callable[[pd.Series], float]


def _total_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def _ann_return(r: pd.Series) -> float:
    return float(r.mean() * ANN)


def _ann_volatility(r: pd.Series) -> float:
    # pd.Series.std() defaults to ddof=1, which is NaN for a one-sample series. Coerce a
    # non-finite std to 0.0 so the "safe on ... zero-vol input" contract holds and no
    # non-finite vol leaks into Sharpe or JSON. Multi-bar values are unchanged (ddof=1 kept).
    vol = r.std() * np.sqrt(ANN)
    return float(vol) if np.isfinite(vol) else 0.0


def _max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve.

    The running peak is floored at starting capital (1.0): a loss on the first bar is a
    real drawdown, not absorbed by seeding the peak at the already-depressed first equity
    value. This floored-peak definition is the conservative one and is used everywhere.
    """
    equity = (1.0 + r).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float((equity / peak - 1.0).min())


def _hit_rate(r: pd.Series) -> float:
    """Fraction of strictly-positive periods (STRICT: a flat/zero period counts as a miss).

    In [0, 1]. Part of the Golden-Rule-6 dashboard — never read in isolation.
    """
    return float((r > 0.0).mean())


def _tail_ratio(r: pd.Series) -> float:
    """Right-tail magnitude over left-tail magnitude: |p95| / |p5|.

    Uses NumPy's default (linear-interpolation) percentile. >1 means upside tails dominate
    downside tails. Returns the finite sentinel 0.0 when the left tail is empty (|p5| == 0,
    e.g. a non-negative series) or the ratio is otherwise non-finite — 0.0 here means
    "undefined", NOT "worst possible", so read it only alongside the rest of the dashboard.
    """
    p95, p5 = np.percentile(r, [95, 5])
    denom = abs(float(p5))
    if denom == 0.0:
        return 0.0
    ratio = float(p95) / denom
    return ratio if math.isfinite(ratio) else 0.0


# Registry of named, pure, single-series metrics. Order defines dict key order.
METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "total_return": _total_return,
    "ann_return": _ann_return,
    "ann_volatility": _ann_volatility,
    "max_drawdown": _max_drawdown,
    "hit_rate": _hit_rate,
    "tail_ratio": _tail_ratio,
}


def metrics_from_returns(returns: pd.Series, *, risk_free: float = 0.0) -> dict[str, float]:
    """Canonical return-based metrics for one return series.

    Safe on empty / zero-vol input (-> zeros). Sharpe is annualized excess return over
    annualized volatility. `risk_free` is the *annualized* risk-free rate; it defaults to
    0.0, which is the historical assumption the promotion gates (e.g. min_holdout_sharpe)
    are calibrated against (#44). Pass a non-zero rate only if every downstream threshold
    is recalibrated to match.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {name: 0.0 for name in METRIC_FUNCTIONS} | {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "calmar": 0.0,
            "skewness": 0.0, "kurtosis": 0.0,
        }

    out = {name: fn(r) for name, fn in METRIC_FUNCTIONS.items()}
    ann_vol = out["ann_volatility"]
    excess = out["ann_return"] - risk_free
    out["sharpe"] = float(excess / ann_vol) if ann_vol > 0 else 0.0

    # Sortino: arithmetic annualized excess return (same numerator as Sharpe) over the
    # annualized downside deviation. Downside deviation uses target/MAR = 0 with the FULL
    # sample in the denominator: dd = sqrt(mean(min(r, 0)**2)). dd == 0 (no down periods) ->
    # finite sentinel 0.0 (undefined, NOT "good"); read only with the rest of the dashboard.
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2))) * np.sqrt(ANN)
    out["sortino"] = float(excess / dd) if dd > 0 and math.isfinite(dd) else 0.0

    # CAGR = compounded annual growth over the realized number of periods. Guard a
    # compounding base <= 0 (a cumulative loss of >= 100%, possible if any period return
    # <= -1) which would make the fractional power complex/non-finite -> sentinel 0.0.
    base = 1.0 + out["total_return"]
    cagr = float(base ** (ANN / len(r)) - 1.0) if base > 0.0 else 0.0
    out["cagr"] = cagr if math.isfinite(cagr) else 0.0

    # Calmar = CAGR / |max drawdown| (note: max_drawdown is the NEGATIVE convention, e.g.
    # -0.2). Numerator is CAGR (compounded), unlike Sortino/Sharpe which use the arithmetic
    # annualized return. |maxDD| == 0 -> finite sentinel 0.0.
    mdd = abs(out["max_drawdown"])
    out["calmar"] = float(out["cagr"] / mdd) if mdd > 0 else 0.0
    # Moments for the DSR non-normality adjustment (#211). RAW (Pearson) kurtosis (fisher=False):
    # a Gaussian series gives ~3, so the gate's (kurtosis-1)/4 term reduces to 0.5. scipy returns
    # NaN for a single-element or zero-variance series; coerce any non-finite moment to 0.0 so no
    # NaN leaks into holdout_metrics / the JSON gate payload. dsr_confidence's T<=1 guard and the
    # MIN_HOLDOUT_OBSERVATIONS=63 floor ensure these placeholders are never consumed.
    skew = float(_stats.skew(r))
    kurt = float(_stats.kurtosis(r, fisher=False))
    out["skewness"] = skew if math.isfinite(skew) else 0.0
    out["kurtosis"] = kurt if math.isfinite(kurt) else 0.0
    return out


def weights_turnover(weights: pd.DataFrame) -> float:
    """Mean per-rebalance one-way turnover: 0.5 * sum |w_t - w_{t-1}| summed over the path.

    For a single full rotation (100% A -> 100% B) this returns 1.0.
    """
    diffs = weights.diff().abs().sum(axis=1).iloc[1:]  # drop first (NaN) row
    return float(diffs.sum() / 2.0)


def avg_gross_exposure(weights: pd.DataFrame) -> float:
    return float(weights.abs().sum(axis=1).mean())


def portfolio_metrics(pf: vbt.Portfolio, weights_eff: pd.DataFrame) -> dict[str, float]:
    """Full backtest metric dict for a simulated portfolio.

    Builds the canonical return-based metrics from `pf.returns()` (so backtest and
    walk-forward report identical definitions, including floored-peak drawdown), then adds
    the path/accounting metrics derived from the effective weights. `cagr` is the
    compounded annual growth rate over the realized number of periods (distinct from the
    arithmetic `ann_return`).
    """
    returns = pf.returns()
    base = metrics_from_returns(returns)
    n_rebalances = int((weights_eff.diff().abs().sum(axis=1) > REBALANCE_EPS).sum())
    return {
        "total_return": base["total_return"],
        "cagr": base["cagr"],
        "ann_volatility": base["ann_volatility"],
        "sharpe": base["sharpe"],
        "sortino": base["sortino"],
        "calmar": base["calmar"],
        "max_drawdown": base["max_drawdown"],
        "hit_rate": base["hit_rate"],
        "tail_ratio": base["tail_ratio"],
        "turnover": weights_turnover(weights_eff),
        "avg_gross_exposure": avg_gross_exposure(weights_eff),
        "n_rebalances": n_rebalances,
    }
