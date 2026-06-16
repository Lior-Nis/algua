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
WEIGHT_RETURN_CORRELATION = 0.20  # Stubbed 0.0 until Task 7 activates return axis

_AXIS_AVAILABILITY = {
    "code_ancestry": True,
    "factor_lineage": True,
    "return_correlation": False,  # becomes True in Task 7
}


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
        family_members: Each dict has keys ``"code_hash": str`` and ``"factors": set[str]``.
        returns_lookup: Stub — ignored until Task 7 activates the return_correlation axis.

    Returns:
        ``(SimVerdict, float)`` — verdict and similarity score (0.0–1.0, or 0.0 on failure).

    Fail-closed: any non-finite intermediate value yields ``(SimVerdict.NOVEL, 0.0)``.
    """
    if not family_members:
        return (SimVerdict.NOVEL, 0.0)

    best_score = 0.0

    for member in family_members:
        # --- code_ancestry axis ---
        code_score = 1.0 if strategy_code_hash == member["code_hash"] else 0.0

        # --- factor_lineage axis (Jaccard) ---
        a: set[str] = strategy_factors
        b: set[str] = member["factors"]
        union = a | b
        if not union:
            # Both sets empty → 0.0 (no overlap, not 1.0)
            factor_score = 0.0
        else:
            factor_score = len(a & b) / len(union)

        # --- return_correlation axis (stubbed until Task 7) ---
        return_score = 0.0

        # Weighted sum
        score = (
            WEIGHT_CODE_ANCESTRY * code_score
            + WEIGHT_FACTOR_LINEAGE * factor_score
            + WEIGHT_RETURN_CORRELATION * return_score
        )

        # Fail-closed: non-finite intermediate → NOVEL
        if not math.isfinite(score):
            return (SimVerdict.NOVEL, 0.0)

        if score > best_score:
            best_score = score

    # Final non-finite check on best_score (e.g. if weights were patched to NaN)
    if not math.isfinite(best_score):
        return (SimVerdict.NOVEL, 0.0)

    # Apply verdict thresholds
    if best_score >= MERGE_THRESHOLD:
        return (SimVerdict.MERGE, best_score)
    if best_score >= PARENTAGE_THRESHOLD:
        return (SimVerdict.PARENTAGE, best_score)
    return (SimVerdict.NOVEL, best_score)
