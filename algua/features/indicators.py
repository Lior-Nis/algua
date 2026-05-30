from __future__ import annotations

from typing import TypeVar

import pandas as pd

# momentum works element-wise on a Series (one symbol) or a wide DataFrame (symbols in
# columns); the TypeVar preserves the caller's type (Series -> Series, DataFrame -> DataFrame).
PandasObj = TypeVar("PandasObj", pd.Series, pd.DataFrame)


def momentum(prices: PandasObj, lookback: int) -> PandasObj:
    """Trailing simple return over `lookback` periods: price_t / price_{t-lookback} - 1."""
    return prices / prices.shift(lookback) - 1.0


def zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional/sample z-score. Std uses population (ddof=0) to stay defined for n>=1."""
    std = values.std(ddof=0)
    if std == 0:
        return values * 0.0
    return (values - values.mean()) / std
