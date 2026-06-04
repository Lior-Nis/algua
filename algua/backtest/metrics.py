"""The ONE canonical metrics module for the backtest core (#37).

Both the single-pass backtest (`engine.run`) and walk-forward (`walkforward.walk_forward`)
compute their return-based metrics here, so a "Sharpe" or "max drawdown" means the same
thing everywhere. Metrics are a registry of named pure functions over a return series
(#40), so adding a metric is a registration — not an edit to a core loop.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import vectorbt as vbt

from algua.backtest._constants import ANN, REBALANCE_EPS

# A metric is a pure function from a (already-dropna'd) return series to a float.
MetricFn = Callable[[pd.Series], float]


def _total_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def _ann_return(r: pd.Series) -> float:
    return float(r.mean() * ANN)


def _ann_volatility(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(ANN))


def _max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve.

    The running peak is floored at starting capital (1.0): a loss on the first bar is a
    real drawdown, not absorbed by seeding the peak at the already-depressed first equity
    value. This floored-peak definition is the conservative one and is used everywhere.
    """
    equity = (1.0 + r).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float((equity / peak - 1.0).min())


# Registry of named, pure, single-series metrics. Order defines dict key order.
METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "total_return": _total_return,
    "ann_return": _ann_return,
    "ann_volatility": _ann_volatility,
    "max_drawdown": _max_drawdown,
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
        return {name: 0.0 for name in METRIC_FUNCTIONS} | {"sharpe": 0.0}

    out = {name: fn(r) for name, fn in METRIC_FUNCTIONS.items()}
    ann_vol = out["ann_volatility"]
    excess = out["ann_return"] - risk_free
    out["sharpe"] = float(excess / ann_vol) if ann_vol > 0 else 0.0
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
    n_periods = len(returns.dropna())
    total_return = base["total_return"]
    cagr = (
        float((1.0 + total_return) ** (ANN / n_periods) - 1.0) if n_periods > 0 else 0.0
    )
    n_rebalances = int((weights_eff.diff().abs().sum(axis=1) > REBALANCE_EPS).sum())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": base["ann_volatility"],
        "sharpe": base["sharpe"],
        "max_drawdown": base["max_drawdown"],
        "turnover": weights_turnover(weights_eff),
        "avg_gross_exposure": avg_gross_exposure(weights_eff),
        "n_rebalances": n_rebalances,
    }
