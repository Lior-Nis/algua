from __future__ import annotations

import numpy as np
import pandas as pd

from algua.backtest.engine import BacktestError

_ANN = 252  # trading days/year
_MIN_WINDOW_BARS = 5


def _segment_bounds(
    n: int, windows: int, holdout_frac: float
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Partition n bars (by index) into K equal windows + a final holdout, as half-open ranges.

    Holdout = the last int(n*holdout_frac) bars. The remaining bars split into `windows` equal
    pieces; any integer-division remainder goes to the LAST window.
    """
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1)")
    holdout_n = int(n * holdout_frac)
    train_n = n - holdout_n
    base = train_n // windows
    if base < _MIN_WINDOW_BARS:
        raise BacktestError(
            f"not enough bars: {train_n} train bars / {windows} windows is "
            f"< {_MIN_WINDOW_BARS} bars/window; widen the period or lower --windows"
        )
    bounds: list[tuple[int, int]] = []
    s = 0
    for i in range(windows):
        e = train_n if i == windows - 1 else s + base
        bounds.append((s, e))
        s = e
    return bounds, (train_n, n)


def metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    """Return-based metrics for one segment. Safe on empty / zero-vol input (-> zeros)."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "total_return": 0.0, "ann_return": 0.0, "ann_volatility": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
        }
    total_return = float((1.0 + r).prod() - 1.0)
    ann_return = float(r.mean() * _ANN)
    ann_vol = float(r.std() * np.sqrt(_ANN))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    equity = (1.0 + r).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_return, "ann_return": ann_return, "ann_volatility": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_drawdown,
    }
