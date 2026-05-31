from __future__ import annotations

import numpy as np
import pandas as pd

_ANN = 252  # trading days/year


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
