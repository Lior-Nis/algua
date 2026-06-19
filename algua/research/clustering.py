"""Pure clustering similarity module — no I/O, no DB access.

Analogous to algua/research/factor_fdr.py. Computes strategy-to-family similarity
across three axes and returns a SimVerdict.

This file is CODEOWNERS-protected alongside gates.py. Constants here are load-bearing;
changes affect the clustering_version() digest and invalidate stored similarity records.
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum


class SimVerdict(Enum):
    MERGE = "merge"
    PARENTAGE = "parentage"
    NOVEL = "novel"


# Protected constants (this file is in CODEOWNERS alongside gates.py)
MERGE_THRESHOLD = 0.85  # similarity >= this → MERGE
PARENTAGE_THRESHOLD = 0.50  # similarity >= this (but < MERGE) → PARENTAGE
# Below PARENTAGE_THRESHOLD → NOVEL

WEIGHT_CODE_ANCESTRY = 0.50
WEIGHT_FACTOR_LINEAGE = 0.30
WEIGHT_RETURN_CORRELATION = 0.20  # return-correlation axis (Task 7; requires returns_lookup)

_AXIS_AVAILABILITY = {
    "code_ancestry": True,
    "factor_lineage": True,
    "return_correlation": True,  # activated in Task 7 (#222)
}

# Minimum shared trading dates for the return-correlation axis to be computed.
# Below this threshold the axis is omitted (contributes 0.0).
_RETURN_CORRELATION_MIN_OVERLAP = 63


def clustering_version() -> str:
    """Hash of the configuration that determines clustering behaviour.

    Changes if thresholds, weights, or axis availability change.
    Returns a 32-character lowercase hex string.
    """
    config = {
        "merge_threshold": MERGE_THRESHOLD,
        "parentage_threshold": PARENTAGE_THRESHOLD,
        "weight_code_ancestry": WEIGHT_CODE_ANCESTRY,
        "weight_factor_lineage": WEIGHT_FACTOR_LINEAGE,
        "weight_return_correlation": WEIGHT_RETURN_CORRELATION,
        "axis_availability": _AXIS_AVAILABILITY,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return digest[:32]


def _return_correlation_axis(
    strategy_returns: object | None,
    member_returns: object | None,
    *,
    min_overlap: int = _RETURN_CORRELATION_MIN_OVERLAP,
) -> float | None:
    """Max correlation between strategy and family member returns over shared dates.

    Returns None if either series is None or shared dates < min_overlap (axis omitted).
    Negative correlation is clamped to 0.0 (anticorrelated strategies are not similar).

    Parameters are typed ``object | None`` so the module avoids a top-level ``pandas`` import;
    callers pass ``pd.Series`` instances.
    """
    if strategy_returns is None or member_returns is None:
        return None
    shared_idx = strategy_returns.index.intersection(member_returns.index)  # type: ignore[attr-defined]
    if len(shared_idx) < min_overlap:
        return None
    corr = strategy_returns.loc[shared_idx].corr(member_returns.loc[shared_idx])  # type: ignore[attr-defined]
    if not math.isfinite(corr):
        return None
    # Round to 14 decimal places to avoid floating-point noise (e.g. self-correlation != 1.0),
    # then clamp to [0.0, 1.0]. float() cast drops the numpy dtype.
    return max(0.0, min(1.0, round(float(corr), 14)))


def pairwise_axes(
    code_a: str,
    factors_a: set[str],
    returns_a: object | None,
    code_b: str,
    factors_b: set[str],
    returns_b: object | None,
) -> tuple[float, dict]:
    """Per-axis similarity between two strategies + the weighted blend.

    Single source of the axis math (family_similarity routes through this). The "return"
    axis is None when not evaluable (< MIN_OVERLAP shared dates or a series missing);
    blended uses 0.0 for an unevaluable return axis. Mirrors family_similarity's prior
    inner loop exactly: empty hash -> 0.0, empty factor sets -> 0.0, negative corr -> 0.0.
    """
    if not code_a or not code_b:
        code_score = 0.0
    else:
        code_score = 1.0 if code_a == code_b else 0.0

    union = factors_a | factors_b
    factor_score = (len(factors_a & factors_b) / len(union)) if union else 0.0

    return_axis = _return_correlation_axis(returns_a, returns_b)
    return_score = return_axis if return_axis is not None else 0.0

    blended = (
        WEIGHT_CODE_ANCESTRY * code_score
        + WEIGHT_FACTOR_LINEAGE * factor_score
        + WEIGHT_RETURN_CORRELATION * return_score
    )
    return blended, {"code": code_score, "factor": factor_score, "return": return_axis}


def family_similarity(
    strategy_code_hash: str,
    strategy_factors: set[str],
    family_members: list[dict],
    *,
    returns_lookup: dict | None = None,
) -> tuple[SimVerdict, float]:
    """Compute similarity between a strategy and a family, returning a verdict and score.

    Args:
        strategy_code_hash: The strategy's code hash.
        strategy_factors: Set of factor names used by the strategy.
        family_members: Each dict has keys ``"code_hash": str``, ``"factors": set[str]``,
            and optionally ``"name": str`` (strategy name, used for return-correlation lookup).
        returns_lookup: Maps names to ``pd.Series`` return series. Use the sentinel
            key ``"__strategy__"`` to supply the current strategy's own returns (a
            name that would otherwise collide with a strategy literally registered
            as ``"__strategy__"``). If None, the return-correlation axis is 0.0.

    Returns:
        ``(SimVerdict, float)`` — verdict and similarity score (0.0–1.0, or 0.0 on failure).

    Fail-closed: any non-finite intermediate value yields ``(SimVerdict.NOVEL, 0.0)``.
    """
    if not family_members:
        return (SimVerdict.NOVEL, 0.0)

    strategy_returns = returns_lookup.get("__strategy__") if returns_lookup is not None else None
    best_score = 0.0
    for member in family_members:
        member_returns = None
        if returns_lookup is not None:
            member_name = member.get("name")
            member_returns = returns_lookup.get(member_name) if member_name else None
        score, _axes = pairwise_axes(
            strategy_code_hash, strategy_factors, strategy_returns,
            member["code_hash"], member["factors"], member_returns,
        )
        if not math.isfinite(score):
            return (SimVerdict.NOVEL, 0.0)
        if score > best_score:
            best_score = score

    if not math.isfinite(best_score):
        return (SimVerdict.NOVEL, 0.0)
    if best_score >= MERGE_THRESHOLD:
        return (SimVerdict.MERGE, best_score)
    if best_score >= PARENTAGE_THRESHOLD:
        return (SimVerdict.PARENTAGE, best_score)
    return (SimVerdict.NOVEL, best_score)
