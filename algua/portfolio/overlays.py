"""Portfolio overlays: an ordered, stateless, TIGHTEN-ONLY stage applied to the construction
output before the capacity cap (spec:
docs/superpowers/specs/2026-09-10-portfolio-overlays-design.md).

An overlay maps (weights, view, params) -> weights, reading only the PIT `view` the signal saw. It
may zero, drop, or scale DOWN a weight; it may never add a symbol, scale up, flip a side, or emit
a non-finite value — `apply_overlays` enforces that after every policy, so a vector that passed
the gross/per-symbol rails inside construction still passes after any chain. Freed weight is cash
(no renormalisation), the same cap-and-hold-cash rule as `apply_capacity_cap`.

Identity rests on this module's STATIC source (approvals hash the module); there is no dynamic
registration. Policies are pure: no I/O, no clock, no global state. The bundled policies live in
`algua.portfolio.overlay_policies`; the shared exception + param-domain helpers live in
`algua.portfolio.overlay_validation` — both are re-exported from here as the seam's public API.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from algua.portfolio.overlay_policies import (
    _regime_gate_lookback,
    _trailing_stop_lookback,
    _validate_regime_gate,
    _validate_trailing_stop,
    regime_gate,
    trailing_stop,
)
from algua.portfolio.overlay_validation import OverlayError

__all__ = [
    "OverlayError",
    "OverlayFn",
    "OverlaySpec",
    "OVERLAY_POLICIES",
    "apply_overlays",
    "get_overlay_policy",
    "overlay_lookback",
    "regime_gate",
    "resolve_overlays",
    "trailing_stop",
    "validate_overlay_params",
]

OverlayFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]

# Float slack for |out| <= |in|: a multiply by a factor <= 1 cannot exceed the input, but a policy
# that recomputes a weight through a different arithmetic path may differ by an ulp.
_TOL = 1e-12


class OverlaySpec(BaseModel):
    """One declared overlay: a policy id + its params. Validated per-policy at load."""

    model_config = {"frozen": True}
    policy: str
    params: dict[str, Any] = {}


# --- invariants ---------------------------------------------------------------------------------


def _checked(index: int, policy: str, before: pd.Series, after: object) -> pd.Series:
    tag = f"overlay[{index}] {policy!r}"
    if not isinstance(after, pd.Series):
        raise OverlayError(f"{tag} returned {type(after).__name__}, not a pd.Series")
    if after.index.has_duplicates:
        raise OverlayError(f"{tag} returned duplicate symbol(s)")
    extra = after.index.difference(before.index)
    if len(extra):
        raise OverlayError(f"{tag} added symbol(s) {sorted(map(str, extra))}")
    try:
        a = after.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise OverlayError(f"{tag} returned non-numeric weights: {exc}") from None
    if not np.isfinite(a).all():
        raise OverlayError(f"{tag} returned non-finite weight(s)")
    b = before.reindex(after.index).to_numpy(dtype="float64")
    if np.any(np.abs(a) > np.abs(b) + _TOL):
        raise OverlayError(f"{tag} increased |weight| — overlays are tighten-only")
    if np.any((a != 0.0) & (np.sign(a) != np.sign(b))):
        raise OverlayError(f"{tag} flipped a weight's sign — overlays are tighten-only")
    return pd.Series(a, index=after.index, dtype="float64")


def apply_overlays(
    weights: pd.Series,
    view: pd.DataFrame,
    specs: Sequence[OverlaySpec],
    fns: Sequence[OverlayFn],
) -> pd.Series:
    """Run the declared chain in order, enforcing the tighten-only invariants after each policy.
    Empty weights short-circuit (nothing to tighten). `fns` must be the resolved callables for
    `specs`, one per spec, in order (see `resolve_overlays`)."""
    if len(specs) != len(fns):
        raise OverlayError(
            f"apply_overlays needs one resolved fn per spec; got {len(specs)} spec(s) and "
            f"{len(fns)} fn(s)"
        )
    if len(weights) == 0:
        return weights
    for i, (spec, fn) in enumerate(zip(specs, fns, strict=True)):
        weights = _checked(i, spec.policy, weights, fn(weights, view, spec.params))
    return weights


# --- registry -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Overlay:
    fn: OverlayFn
    validate: Callable[[dict[str, Any]], None]
    lookback: Callable[[dict[str, Any]], int]


_OVERLAYS: dict[str, _Overlay] = {
    "trailing_stop": _Overlay(trailing_stop, _validate_trailing_stop, _trailing_stop_lookback),
    "regime_gate": _Overlay(regime_gate, _validate_regime_gate, _regime_gate_lookback),
}
# Read-only public dispatch view (see module docstring: static source is the identity).
OVERLAY_POLICIES = MappingProxyType(_OVERLAYS)


def _policy(policy_id: str) -> _Overlay:
    try:
        return _OVERLAYS[policy_id]
    except KeyError:
        raise OverlayError(
            f"unknown overlay policy {policy_id!r}; available: {sorted(_OVERLAYS)}"
        ) from None


def get_overlay_policy(policy_id: str) -> OverlayFn:
    return _policy(policy_id).fn


def validate_overlay_params(policy_id: str, params: dict[str, Any]) -> None:
    """Per-policy load-time validation: unknown id, then the policy's own exact-key + type +
    domain checks (non-finite floats are rejected inside those checks). Raises OverlayError."""
    _policy(policy_id).validate(params)


def overlay_lookback(spec: OverlaySpec) -> int:
    """The longest trailing window (in bars) the policy reads for these params."""
    return _policy(spec.policy).lookback(spec.params)


def resolve_overlays(
    specs: Sequence[OverlaySpec], *, feature_lookback: int | None
) -> tuple[OverlayFn, ...]:
    """Resolve + validate a declared chain: each policy id and its params, then the #345 cross-
    check that a DECLARED `feature_lookback` covers the longest overlay window (an under-declared
    lookback would size the walk-forward embargo too small). Returns the callables in order."""
    fns: list[OverlayFn] = []
    for i, spec in enumerate(specs):
        try:
            validate_overlay_params(spec.policy, spec.params)
        except OverlayError as exc:
            raise OverlayError(f"overlay[{i}] {spec.policy!r}: {exc}") from None
        fns.append(get_overlay_policy(spec.policy))
    if feature_lookback is not None and specs:
        need = max(overlay_lookback(s) for s in specs)
        if feature_lookback < need:
            raise OverlayError(
                f"feature_lookback {feature_lookback} is smaller than the longest overlay window "
                f"{need}; declare feature_lookback >= {need}"
            )
    return tuple(fns)
