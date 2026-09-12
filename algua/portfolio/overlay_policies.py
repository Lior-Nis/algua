"""The two bundled overlay policies — `trailing_stop` and `regime_gate` — plus their per-policy
validators and lookback functions. The seam (`algua/portfolio/overlays.py`) registers them."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from algua.features.regime import (
    equal_weight_index,
    robust_zscore,
    rolling_drawdown,
    turbulence,
    wide_adj_close,
)
from algua.portfolio.overlay_validation import OverlayError, _exact_keys, _float_in, _positive_int


def _trailing_bars(view: pd.DataFrame, n: int) -> pd.DataFrame:
    """The last `n` distinct timestamps of a long bar-schema view (all rows on those bars). A
    policy reads only its own declared window; slicing before the pivot keeps each call
    O(window x symbols) instead of O(history x symbols), and makes the backtest (full expanding
    view) and the lane (feature_lookback-sized view) feed the policy byte-identical inputs."""
    stamps = view.index.unique().sort_values()
    if len(stamps) <= n:
        return view
    return view[view.index >= stamps[-n]]


# --- trailing_stop ------------------------------------------------------------------------------


def trailing_stop(weights: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Zero a name whose adj_close sits more than `stop_pct` below its `lookback`-bar rolling high
    (incl. the current bar), and keep it at zero while that breach fired within the last
    `cooldown_bars` bars — all read from the view, no position state. A name with fewer than
    `lookback` bars uses the bars it has; a name absent from `view` passes through unchanged.
    A short is stopped the same way (weight -> 0, never flipped)."""
    lookback = int(params["lookback"])
    stop_pct = float(params["stop_pct"])
    cooldown = int(params["cooldown_bars"])
    view = _trailing_bars(view, _trailing_stop_lookback(params))
    wide = wide_adj_close(view)
    present = [s for s in weights.index if s in wide.columns]
    if not present:
        return weights
    px = wide[present]
    high = px.rolling(lookback, min_periods=1).max()
    breached = px < (1.0 - stop_pct) * high  # NaN price -> False (no breach on a missing bar)
    stopped = breached.iloc[-(cooldown + 1):].any(axis=0)
    out = weights.astype("float64").copy()
    out[stopped.index[stopped.to_numpy()]] = 0.0
    return out


def _validate_trailing_stop(params: dict[str, Any]) -> None:
    _exact_keys(params, {"lookback", "stop_pct", "cooldown_bars"})
    _positive_int(params, "lookback")
    _float_in(params, "stop_pct", 0.0, 1.0, lo_open=True, hi_open=True)
    _positive_int(params, "cooldown_bars", minimum=0)


def _trailing_stop_lookback(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + int(params["cooldown_bars"])


# --- regime_gate --------------------------------------------------------------------------------

_REGIME_INT_KEYS = (
    "trend_window", "dd_window", "turb_window", "z_window", "shock_window", "fast_lookback",
    "persistence",
)
_REGIME_KEYS = {
    *_REGIME_INT_KEYS, "dd_threshold", "shock_return", "turb_z", "fast_turb_z",
    "neutral_exposure", "risk_off_exposure", "fast_exposure",
}

# One quarter of trading sessions. Bounds how far back the persistence rule may look for a
# same-state run: `tail` (below) sizes the turbulence/z-score tail to cover this whole horizon, so
# every leg the horizon can select from is actually defined over it — a run outside it can never
# be picked, closing the "ffill reaches an arbitrarily distant, differently-legged state" gap.
REGIME_SEARCH_BARS = 63


def regime_gate(weights: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Two-speed exposure multiplier from the strategy's OWN universe (no reference symbols). The
    regime controls the risk budget, not asset selection: every weight is scaled by one factor.

    Precondition: `turb_window` must exceed the number of distinct symbols in `view` so the
    trailing turbulence covariance can be full rank — a universe at least as wide as `turb_window`
    raises `OverlayError` rather than silently leaving the volatility leg off.

    SLOW gate, per bar: count three stresses on the equal-weight universe index —
      trend       level < its `trend_window`-bar mean
      drawdown    rolling_drawdown(level, dd_window) < -dd_threshold
      volatility  robust_zscore(turbulence(view, turb_window), z_window) > turb_z
    score 0 -> risk-on (1.0), 1 -> neutral (`neutral_exposure`), >= 2 -> risk-off
    (`risk_off_exposure`). The state IN EFFECT is that of the most recent run of `persistence`
    consecutive bars sharing one state within the last `REGIME_SEARCH_BARS` bars; none -> risk-on.

    FAST overlay: if within the last `fast_lookback` bars the index's `shock_window`-bar return
    was below -`shock_return` OR the turbulence robust-z exceeded `fast_turb_z`, `fast_exposure`
    applies.

    Effective multiplier = min(slow, fast). A component without enough history is NOT stressed (a
    gate cannot fire on data it does not have), so on a short view this is a no-op."""
    p = params
    turb_window = int(p["turb_window"])
    n_symbols = view["symbol"].nunique()
    if n_symbols >= turb_window:
        raise OverlayError(
            f"regime_gate: turb_window ({turb_window}) must exceed the number of symbols in the "
            f"view ({n_symbols}) so the turbulence covariance is full rank; widen turb_window or "
            "narrow the universe"
        )
    # Slice AFTER the symbol-count precondition (which judges the universe as a WHOLE, so a
    # churned universe cannot slip past it) and before any pivot. Every leg is a ratio or a
    # return — trend (level / rolling mean), drawdown, pct_change shock, turbulence on returns —
    # so re-basing `equal_weight_index` to 1.0 at the slice start leaves all of them unchanged.
    view = _trailing_bars(view, _regime_gate_lookback(params))
    persistence = int(p["persistence"])
    level = equal_weight_index(view)
    if level.empty:
        return weights
    tail = int(p["z_window"]) + REGIME_SEARCH_BARS
    turb_z = robust_zscore(turbulence(view, turb_window, last=tail), int(p["z_window"]))

    trend = level < level.rolling(int(p["trend_window"]), min_periods=int(p["trend_window"])).mean()
    dd = rolling_drawdown(level, int(p["dd_window"])) < -float(p["dd_threshold"])
    vol = turb_z > float(p["turb_z"])  # NaN -> False
    score = trend.astype(int) + dd.astype(int) + vol.astype(int)
    state = score.clip(upper=2)
    horizon = state.iloc[-REGIME_SEARCH_BARS:]
    run_ok = horizon.rolling(persistence, min_periods=persistence).max() == horizon.rolling(
        persistence, min_periods=persistence
    ).min()
    in_effect = horizon.where(run_ok).ffill().fillna(0).iloc[-1]
    slow_by_state = {0: 1.0, 1: float(p["neutral_exposure"]), 2: float(p["risk_off_exposure"])}
    slow = slow_by_state[int(in_effect)]

    shock = level.pct_change(int(p["shock_window"]), fill_method=None) < -float(p["shock_return"])
    fast_signal = shock | (turb_z > float(p["fast_turb_z"]))
    fast_hit = bool(fast_signal.iloc[-int(p["fast_lookback"]):].any())
    fast = float(p["fast_exposure"]) if fast_hit else 1.0

    return weights.astype("float64") * min(slow, fast)


def _validate_regime_gate(params: dict[str, Any]) -> None:
    _exact_keys(params, _REGIME_KEYS)
    for key in _REGIME_INT_KEYS:
        _positive_int(params, key)
    if params["persistence"] > params["dd_window"]:
        raise OverlayError("persistence must be <= dd_window")
    if params["persistence"] > REGIME_SEARCH_BARS:
        raise OverlayError(f"persistence must be <= REGIME_SEARCH_BARS ({REGIME_SEARCH_BARS})")
    if params["fast_lookback"] > REGIME_SEARCH_BARS:
        raise OverlayError(f"fast_lookback must be <= REGIME_SEARCH_BARS ({REGIME_SEARCH_BARS})")
    for key in ("dd_threshold", "shock_return"):
        _float_in(params, key, 0.0, 1.0, lo_open=True, hi_open=True)
    for key in ("turb_z", "fast_turb_z"):
        _float_in(params, key, 0.0, math.inf, lo_open=True, hi_open=True)
    for key in ("neutral_exposure", "risk_off_exposure", "fast_exposure"):
        _float_in(params, key, 0.0, 1.0, lo_open=False, hi_open=False)
    if float(params["risk_off_exposure"]) > float(params["neutral_exposure"]):
        raise OverlayError("risk_off_exposure must be <= neutral_exposure")


def _regime_gate_lookback(params: dict[str, Any]) -> int:
    """The declared window must cover the WHOLE persistence search horizon, not just one run:
    the search scans the last `REGIME_SEARCH_BARS` bars, so every leg must be DEFINED over all of
    them or a `feature_lookback`-sized lane view selects a different state than the backtest's
    expanding view. `+ REGIME_SEARCH_BARS` is a strict superset of `+ persistence` (the validator
    caps `persistence <= REGIME_SEARCH_BARS`)."""
    p = params
    return max(
        int(p["trend_window"]), int(p["dd_window"]),
        int(p["turb_window"]) + int(p["z_window"]),
        int(p["shock_window"]) + int(p["fast_lookback"]),
    ) + REGIME_SEARCH_BARS
