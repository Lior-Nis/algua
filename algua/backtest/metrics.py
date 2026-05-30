from __future__ import annotations

import pandas as pd


def weights_turnover(weights: pd.DataFrame) -> float:
    """Mean per-rebalance one-way turnover: 0.5 * sum |w_t - w_{t-1}| summed over the path.

    For a single full rotation (100% A -> 100% B) this returns 1.0.
    """
    diffs = weights.diff().abs().sum(axis=1).iloc[1:]  # drop first (NaN) row
    return float(diffs.sum() / 2.0)


def avg_gross_exposure(weights: pd.DataFrame) -> float:
    return float(weights.abs().sum(axis=1).mean())
