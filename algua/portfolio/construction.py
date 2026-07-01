from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from algua.contracts.types import CapacityLimit

# A construction policy maps cross-sectional scores -> target weights under a risk convention.
# `view` is the same PIT bar-schema frame the signal saw (passed so a future vol-targeting policy
# can estimate vol from prices with no contract change); the starter policies ignore it.
ConstructFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]


class ConstructionError(ValueError):
    """An invalid construction policy id, params, or score series. Subclasses ValueError so the
    CLI's json error contract still renders it."""


def _finite_scores(scores: pd.Series) -> pd.Series:
    """Fail closed on a non-numeric score series, then DROP missing/non-finite scores.

    A missing or NaN/inf score means 'no opinion - not selectable' and is removed; it is NEVER
    coerced to 0.0 (a real 0.0 score must stay distinct from 'no score'). Mirrors the fail-closed
    philosophy of risk.limits.check_finite_weights at the construction seam.
    """
    if pd.api.types.is_bool_dtype(scores) or not pd.api.types.is_numeric_dtype(scores):
        raise ConstructionError("signal returned a non-numeric score series")
    if scores.index.isnull().any():
        raise ConstructionError("signal returned a null symbol label")
    if scores.index.has_duplicates:
        raise ConstructionError("signal returned duplicate symbol score(s)")
    return scores[np.isfinite(scores.to_numpy())]


def _ranked(scores: pd.Series) -> pd.Series:
    """Finite scores ordered by (score descending, symbol ascending). Sorting by symbol first
    (stable) then by score makes ties resolve to the lexicographically-smaller symbol regardless of
    input order, so a per-bar `signal` Series and a `signal_panel` matrix row select IDENTICALLY."""
    finite = _finite_scores(scores)
    by_symbol = finite.sort_index(kind="stable")
    return by_symbol.sort_values(ascending=False, kind="stable")


def top_k_equal_weight(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Hold the top-`top_k` names by score, equal weight 1/k."""
    top_k = int(params["top_k"])
    winners = _ranked(scores).head(top_k).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)


def equal_weight_positive(
    scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]
) -> pd.Series:
    """Equal-weight every name with a strictly positive score."""
    finite = _finite_scores(scores)
    winners = sorted(finite[finite > 0.0].index)
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)


def score_proportional_long(
    scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]
) -> pd.Series:
    """Clip negatives to zero; weight the positives proportionally, normalized to gross 1.0."""
    finite = _finite_scores(scores)
    positive = finite[finite > 0.0]
    total = float(positive.sum())
    if total <= 0.0:
        return pd.Series(dtype="float64")
    return (positive / total).sort_index()


def _dollar_adv(view: pd.DataFrame, window: int) -> pd.Series:
    """Per-symbol trailing dollar-ADV over the last `window` bars of `view`.

    `view` is the PIT bar-schema frame the signal saw — history up to and INCLUDING the fully-closed
    decision bar t, never t+1 — so a trailing window over it carries NO look-ahead. Dollar volume is
    raw `close * volume` (the actual historical dollars that traded), not adj_close.

    Fail-closed shape guard: `view` must carry `symbol`, `close`, and `volume` (a live/paper view
    that dropped `volume` or supplied no history must be a loud failure, not a silent flatten).

    A symbol is OMITTED from the result (no established ADV -> the caller treats it as un-sized)
    unless its trailing window has EXACTLY `window` finite dollar-volume observations. That fails
    closed on all three degenerate cases at once: fewer than `window` bars (newly-listed / sparse),
    and any NaN/inf dollar value in the window (a NaN would otherwise be silently skipped by the
    mean and produce a positive ADV from a partial sample). Returns a Series indexed by the
    surviving symbols mapping to their mean dollar volume (may be 0.0 when volume=0)."""
    for col in ("symbol", "close", "volume"):
        if col not in view.columns:
            raise ConstructionError(
                f"capacity cap requires a '{col}' column in the bar view (fail closed)"
            )
    if view.empty:
        return pd.Series(dtype="float64")
    dollar = view["close"].to_numpy(dtype="float64") * view["volume"].to_numpy(dtype="float64")
    tmp = pd.DataFrame({"symbol": view["symbol"].to_numpy(), "dollar": dollar}, index=view.index)
    # Sort by timestamp so `tail(window)` is the most-recent `window` bars per symbol regardless of
    # input row order; stable so equal timestamps keep their relative order.
    tmp = tmp.sort_index(kind="stable")
    tail = tmp.groupby("symbol", sort=False).tail(window).copy()
    tail["_finite"] = np.isfinite(tail["dollar"].to_numpy())
    grouped = tail.groupby("symbol", sort=False)
    # A clean full window == `window` FINITE observations in the trailing slice (tail holds at most
    # `window` rows per symbol, so this is both the min-history AND the all-finite gate).
    finite_counts = grouped["_finite"].sum()
    full = finite_counts[finite_counts >= window].index
    if len(full) == 0:
        return pd.Series(dtype="float64")
    # mean() over a fully-finite window is exact (no NaN to skip); keep only surviving symbols.
    return grouped["dollar"].mean().reindex(full)


def apply_capacity_cap(
    weights: pd.Series, view: pd.DataFrame, capacity: CapacityLimit
) -> pd.Series:
    """Cap each target weight at `max_participation_rate` of the name's trailing dollar-ADV,
    evaluated at the declared `reference_aum` (issue #344). SCALES down (never rejects): the freed
    weight is left as cash, and the vector is NOT renormalized (renormalizing could push notional
    back onto neighbours and re-breach — cap-and-hold-cash is the KISS-correct behaviour).

    Per symbol s: max_weight_s = (max_participation_rate * dollar_adv_s) / reference_aum, then
    capped_w_s = copysign(min(|w_s|, max_weight_s), w_s). Scaling only ever REDUCES |weight|, so a
    weight vector that already passed the gross / per-symbol rails still passes them.

    Fail-closed illiquidity: a symbol with no established ADV (zero/NaN/inf dollar volume, fewer
    than `adv_window_bars` observations, or simply absent from `view`) gets max_weight_s = 0 and is
    forced flat. There is no division by ADV anywhere (we divide by the validated-positive
    reference_aum and multiply by ADV), so no divide-by-zero is possible."""
    if len(weights) == 0:
        return weights
    adv = _dollar_adv(view, capacity.adv_window_bars)  # only full-window symbols present
    max_weight = (capacity.max_participation_rate * adv) / capacity.reference_aum
    # Align to the weight index; a symbol missing an established ADV -> NaN -> treated as 0 budget.
    budget = max_weight.reindex(weights.index).to_numpy(dtype="float64")
    # Non-finite (NaN from missing symbol, or a stray inf) -> 0 budget (fail closed).
    budget = np.where(np.isfinite(budget) & (budget > 0.0), budget, 0.0)
    w = weights.to_numpy(dtype="float64")
    capped = np.copysign(np.minimum(np.abs(w), budget), w)
    return pd.Series(capped, index=weights.index, dtype="float64")


def _require_no_unknown_keys(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ConstructionError(f"unknown construction param(s): {sorted(unknown)}")


def _validate_top_k(params: dict[str, Any]) -> None:
    _require_no_unknown_keys(params, {"top_k"})
    if "top_k" not in params:
        raise ConstructionError("top_k_equal_weight requires 'top_k'")
    top_k = params["top_k"]
    # bool is an int subtype; reject it so True can't masquerade as 1.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ConstructionError(f"top_k must be a positive int, got {top_k!r}")


def _validate_no_params(params: dict[str, Any]) -> None:
    _require_no_unknown_keys(params, set())


def _assert_finite_json(value: Any, path: str = "construction_params") -> None:
    """Recursively reject non-finite floats, non-string dict keys, and non-JSON value types so
    config_hash (serialized with allow_nan=False) is canonical and meaningful."""
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConstructionError(f"{path}: non-string key {k!r}")
            _assert_finite_json(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_finite_json(v, f"{path}[{i}]")
    elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ConstructionError(f"{path}: non-finite float {value!r}")
    else:
        raise ConstructionError(f"{path}: non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True)
class _Policy:
    fn: ConstructFn
    validate: Callable[[dict[str, Any]], None]


_POLICIES: dict[str, _Policy] = {
    "top_k_equal_weight": _Policy(top_k_equal_weight, _validate_top_k),
    "equal_weight_positive": _Policy(equal_weight_positive, _validate_no_params),
    "score_proportional_long": _Policy(score_proportional_long, _validate_no_params),
}
# Read-only public dispatch view. Identity rests on this module's STATIC source (approvals.py hashes
# the whole module), not on runtime dispatch state — there is no dynamic registration.
CONSTRUCTION_POLICIES = MappingProxyType(_POLICIES)


def get_construction_policy(policy_id: str) -> ConstructFn:
    try:
        return _POLICIES[policy_id].fn
    except KeyError:
        raise ConstructionError(
            f"unknown construction policy {policy_id!r}; available: {sorted(_POLICIES)}"
        ) from None


def validate_construction_params(policy_id: str, params: dict[str, Any]) -> None:
    """Per-policy load-time validation: unknown id, then finite/JSON values, then the policy's own
    type+domain checks (e.g. top_k positive int). Raises ConstructionError on any violation."""
    try:
        policy = _POLICIES[policy_id]
    except KeyError:
        raise ConstructionError(
            f"unknown construction policy {policy_id!r}; available: {sorted(_POLICIES)}"
        ) from None
    _assert_finite_json(params)
    policy.validate(params)
