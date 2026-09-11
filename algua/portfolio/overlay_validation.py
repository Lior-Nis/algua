"""Shared exception + param-domain-validation helpers for overlay policies (a leaf so both the
overlay policies and the overlay seam can import it without a cycle)."""
from __future__ import annotations

import math
from typing import Any


class OverlayError(ValueError):
    """An invalid overlay policy id, params, or output. Subclasses ValueError so the CLI's json
    error contract still renders it."""


def _exact_keys(params: dict[str, Any], required: set[str]) -> None:
    missing = required - set(params)
    if missing:
        raise OverlayError(f"missing param(s): {sorted(missing)}")
    unknown = set(params) - required
    if unknown:
        raise OverlayError(f"unknown param(s): {sorted(unknown)}")


def _positive_int(params: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    v = params[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise OverlayError(f"{key} must be an int >= {minimum}, got {v!r}")
    return v


def _float_in(
    params: dict[str, Any], key: str, lo: float, hi: float, *, lo_open: bool, hi_open: bool
) -> float:
    v = params[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise OverlayError(f"{key} must be a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise OverlayError(f"{key} is non-finite: {v!r}")
    below = f <= lo if lo_open else f < lo
    above = f >= hi if hi_open else f > hi
    if below or above:
        lb, rb = ("(" if lo_open else "["), (")" if hi_open else "]")
        raise OverlayError(f"{key} must be in {lb}{lo}, {hi}{rb}, got {v!r}")
    return f
