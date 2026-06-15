"""Standalone-evaluable alpha factors: signal-shaped (view, params) -> cross-sectional scores.

Distinct from indicators.py (arbitrary-signature building blocks): an alpha here IS reusable as a
strategy `signal`, and can be evaluated on its own (algua factor eval). Pure layer — pandas only.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.features.catalogue import FactorKind, factor
from algua.features.indicators import momentum


@factor(
    standalone=True,
    summary="Cross-sectional trailing return per symbol over `lookback` bars (the momentum alpha).",
    kind=FactorKind.MOMENTUM,
    tags=["momentum", "cross-sectional", "alpha"],
)
def xs_trailing_return(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol — the cross_sectional_momentum alpha as a
    reusable, individually-evaluable factor. Empty until `lookback`+1 bars of history exist."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return momentum(wide.iloc[-1 - lookback :], lookback).iloc[-1].dropna()
