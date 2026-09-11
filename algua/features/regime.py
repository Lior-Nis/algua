"""Universe-derived market-state features for the overlay stage (overlays spec §Features).

Pure (no I/O, no clock, no global state), `adj_close`-only. `view` is the long bar-schema frame the
signal saw (tz-aware `timestamp` index; `symbol`, `adj_close` columns). Everything here is
computable from the strategy's own universe, so a regime overlay needs no reference symbols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algua.features.catalogue import FactorKind, factor

# Consistency constant: 1 / Phi^{-1}(3/4). Scales a MAD to a normal-distribution sigma.
_MAD_TO_SIGMA = 1.4826


def wide_adj_close(view: pd.DataFrame) -> pd.DataFrame:
    """timestamp x symbol matrix of `adj_close`, sorted by timestamp (never raw `close`, #521)."""
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide.sort_index()


@factor(
    summary="Equal-weight index level (base 1.0) of the universe from adj_close simple returns.",
    kind=FactorKind.OTHER,
    tags=["regime", "universe", "index"],
)
def equal_weight_index(view: pd.DataFrame) -> pd.Series:
    """Each bar's index return is the mean of the member simple returns available that bar (a
    member missing a bar contributes nothing that bar); the level compounds from 1.0. A bar where
    no member has a return (the first bar) has index return 0."""
    rets = wide_adj_close(view).pct_change(fill_method=None)
    idx_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + idx_ret).cumprod()


@factor(
    summary="Drawdown of a level series from its trailing `window`-bar high.",
    kind=FactorKind.VOLATILITY,
    tags=["regime", "drawdown"],
)
def rolling_drawdown(level: pd.Series, window: int) -> pd.Series:
    """`level / rolling_max(window) - 1`; NaN until `window` observations exist."""
    return level / level.rolling(window, min_periods=window).max() - 1.0


@factor(
    summary="Robust z-score: (x - rolling median) / (1.4826 * rolling MAD).",
    kind=FactorKind.OTHER,
    tags=["normalization", "robust"],
)
def robust_zscore(x: pd.Series, window: int) -> pd.Series:
    """For each bar t: (x_t - med) / (1.4826 * MAD) over the window ending at t, where med is the
    window median and MAD the median of |x_i - med| over that SAME window (the exact windowed MAD).
    NaN until `window` observations exist, NaN for any window containing NaN, and NaN (never inf)
    when the MAD is zero (a constant window)."""
    v = x.to_numpy(dtype="float64")
    out = np.full(len(v), np.nan)
    if len(v) >= window:
        win = np.lib.stride_tricks.sliding_window_view(v, window)  # (n - window + 1, window)
        med = np.median(win, axis=1)
        scale = _MAD_TO_SIGMA * np.median(np.abs(win - med[:, None]), axis=1)
        ok = np.isfinite(scale) & (scale > 0.0)
        z = np.full(len(win), np.nan)
        z[ok] = (win[ok, -1] - med[ok]) / scale[ok]
        out[window - 1 :] = z
    return pd.Series(out, index=x.index, dtype="float64")


@factor(
    summary=(
        "Cross-sectional turbulence: Mahalanobis distance of a bar's return vector vs the "
        "trailing covariance."
    ),
    kind=FactorKind.VOLATILITY,
    tags=["regime", "turbulence", "cross-sectional"],
)
def turbulence(view: pd.DataFrame, window: int, *, last: int | None = None) -> pd.Series:
    """For bar t: d' * pinv(cov) * d where d = r_t - mean(r over the `window` bars BEFORE t) and cov
    is that trailing window's covariance (pseudo-inverse, so a singular covariance is finite). A
    symbol enters bar t's vector only if it has a return on t AND on every one of the `window`
    prior bars. NaN until `window` prior returns exist (i.e. the first `window + 1` bars) or when no
    symbol qualifies. `last=k` computes only the final k bars (the rest NaN) — an overlay evaluated
    per decision bar only needs the tail, and this keeps that O(window) instead of O(history).
    NaN also when the trailing covariance is not full rank (e.g. perfectly correlated members, or
    more qualifying symbols than `window` bars) — a Mahalanobis distance is undefined there and a
    pseudo-inverse would produce arbitrarily large values; a downstream gate reads NaN as
    "not stressed"."""
    rets = wide_adj_close(view).pct_change(fill_method=None)
    n = len(rets)
    out = pd.Series(np.nan, index=rets.index, dtype="float64")
    start = window + 1
    if last is not None:
        start = max(start, n - last)
    values = rets.to_numpy(dtype="float64")
    for i in range(start, n):
        hist = values[i - window : i]
        ok = np.isfinite(hist).all(axis=0) & np.isfinite(values[i])
        if not ok.any():
            continue
        h = hist[:, ok]
        d = values[i, ok] - h.mean(axis=0)
        cov = np.atleast_2d(np.cov(h, rowvar=False))
        if np.linalg.matrix_rank(cov) < cov.shape[0]:
            continue
        out.iloc[i] = float(d @ np.linalg.pinv(cov) @ d)
    return out
