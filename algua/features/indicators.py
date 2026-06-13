from __future__ import annotations

import pandas as pd

from algua.features.catalogue import FactorKind, factor


@factor(
    summary="Trailing simple return per symbol over `lookback` periods.",
    kind=FactorKind.MOMENTUM,
    tags=["momentum", "cross-sectional"],
)
def momentum[PandasObj: (pd.Series, pd.DataFrame)](prices: PandasObj, lookback: int) -> PandasObj:
    """Trailing simple return over `lookback` periods: price_t / price_{t-lookback} - 1.

    Works on a Series (one symbol) or a wide DataFrame (symbols in columns); the type
    parameter preserves the caller's type (Series -> Series, DataFrame -> DataFrame).
    """
    return prices / prices.shift(lookback) - 1.0


@factor(
    summary="Cross-sectional z-score (population std; all-NaN on a degenerate cross-section).",
    kind=FactorKind.OTHER,
    tags=["normalization", "cross-sectional"],
)
def zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional/sample z-score. Std uses population (ddof=0) to stay defined for n>=1.

    A zero-variance (degenerate/flat) cross-section has an *undefined* z-score, so this
    returns all-NaN rather than all-zero. Zero would read as a legitimate "no signal" tie
    and survive a downstream `.dropna()`; NaN makes the degeneracy explicit and lets
    rankers/filters drop the cross-section instead of treating it as a flat tie.
    """
    std = values.std(ddof=0)
    if std == 0:
        return values * float("nan")
    return (values - values.mean()) / std
