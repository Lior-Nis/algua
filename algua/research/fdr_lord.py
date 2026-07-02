"""LORD++ online-FDR accounting with count-triggered cohort restarts (from gates.py, #335).

Pure-maths leaf: stdlib only, no numpy/scipy, no research-internal imports. Provides the
γ-sequence, the per-cohort position map, and the LORD++ test level α_t.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# LORD++ FDR accounting layer (#220, Phase 2). Protected constants — relaxing them weakens the gate.
# FDR here is an operating target (shared-holdout dependence breaks the formal guarantee); Phase 3
# (#221) adds dependence-aware calibration. The p-value fed here is 1 − dsr_confidence, which is
# P(SR_true ≤ SR*) — i.e. the FDR guarantee governs discoveries relative to the DSR null (SR > SR*),
# a STRONGER criterion than the simple null (SR > 0).
FDR_ALPHA = 0.05   # target FDR level
FDR_W0 = FDR_ALPHA / 2   # initial alpha-wealth (standard choice, Ramdas et al. 2017)
# γ-sequence truncation for normalization. 10 000 terms captures >99.99% of tail mass; α_t at
# positions > 10 000 uses γ_j=0 for j>10 000 (negligible, conservative).
FDR_GAMMA_TRUNCATION = 10_000

# Count-triggered cohort restarts (#324). The LORD++ stream is partitioned into consecutive,
# non-overlapping COHORTS of exactly FDR_COHORT_SIZE binding tests, assigned by ARRIVAL ORDER;
# each cohort runs an INDEPENDENT LORD++ stream (fresh W0, in-cohort t and rejection positions).
#
# WHY (protected constant — raising it weakens the fix). A SINGLE lifetime-global LORD++ stream is
# anti-scaling: every measured test (mostly clear-null flops, p≈1) advances position t, so in a
# dry spell α_t = γ_t·W0 → 0 as t grows — testing MORE garbage monotonically lowers everyone's
# future bar. This is INTRINSIC to a *lifetime* target on a garbage-dominated funnel: any valid
# lifetime online-FDR procedure must drive the per-test level to 0 over an unbounded null stream.
# Bounding the count (a deterministic LORD-with-restarts, Ramdas et al. 2017; Zrnic et al.) caps
# the worst-case dry-spell level at γ_{FDR_COHORT_SIZE}·W0 INDEPENDENT of throughput — 1000
# tests/day or 1/day yield identical within-cohort statistics. The FDR guarantee is RE-SCOPED and
# EXPLICIT: FDR is controlled PER COHORT of FDR_COHORT_SIZE binding tests at FDR_ALPHA, NOT per
# lifetime. Cumulative exposure over K completed cohorts is bounded by FDR_ALPHA·K (surfaced as an
# audit-only fdr_exposure block; see promotion.py). "Only bind passing rows" is REJECTED — it hides
# non-rejections from the multiplicity process (covert loosening). SAFFRON is insufficient here: it
# indexes γ by non-candidate count, so clear-null garbage still alpha-deaths.
#
# WHY 64 (power calibration, not aesthetics). Real normalized-γ dry-spell floors: α_1 = 0.00165
# (dsr ≥ 0.99835) — the first test is already strict, inherent to W0 + γ-normalization
# (pre-existing, not a regression); α_64 = 4.6e-5 (dsr ≥ 0.99995). 64 keeps α_N within ~35× of α_1
# (same order as the pre-existing α_1 strictness) and caps decay; larger N approaches lifetime-like
# decay, smaller N means more independent 5% cohorts (weaker multiplicity control).
FDR_COHORT_SIZE = 64


def fdr_cohort_position(k: int) -> tuple[int, int]:
    """Map a 1-based GLOBAL binding-test ordinal ``k`` to its ``(cohort_index, within_cohort_t)``.

    ``cohort_index = (k − 1) // FDR_COHORT_SIZE`` (0-based); ``within_cohort_t`` runs 1..
    FDR_COHORT_SIZE and is the position fed to :func:`lord_plus_plus_level` for that cohort's
    independent LORD++ stream. Fails closed (``ValueError``) on ``k < 1`` — a binding ordinal is
    always ≥ 1 by construction, so a non-positive value is a caller bug, not a silent-0 default.
    """
    if k < 1:
        raise ValueError(f"binding-test ordinal k must be >= 1, got {k}")
    return (k - 1) // FDR_COHORT_SIZE, (k - 1) % FDR_COHORT_SIZE + 1


def _compute_lord_gamma(n: int) -> list[float]:
    """Normalized LORD++ γ weights for j=1..n.

    Raw: γ_j ∝ log(max(j, 2)) / (j · exp(√(log(max(j, 2)))))
    max(j, 2) is the standard practical variant per Ramdas et al. 2017 / the onlineFDR R package,
    ensuring γ_j > 0 for all j (log(max(1,2))=log(2)>0 handles j=1). Dividing by the truncated
    sum normalizes so Σγ_j ≤ 1.0 + machine-epsilon over the truncation window.
    """
    raw = [
        math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(max(j, 2)))))
        for j in range(1, n + 1)
    ]
    total = sum(raw)
    return [w / total for w in raw]


_LORD_GAMMA: list[float] = _compute_lord_gamma(FDR_GAMMA_TRUNCATION)


def lord_plus_plus_level(
    t: int,
    discovery_indices: Sequence[int],
    *,
    alpha: float = FDR_ALPHA,
    w0: float = FDR_W0,
) -> float:
    """LORD++ test level α_t (Ramdas et al. 2017 Biometrika 104:1).

    α_t = γ_t · w0 + (α − w0) · γ_{t−τ_1} + α · Σ_{j≥2} γ_{t−τ_j}

    where τ_1 < τ_2 < … are the 1-indexed positions of past discoveries (all strictly < t).
    α_t depends ONLY on past decisions — no circularity. Wealth is computed from the ledger
    rows on every call (not cached), mirroring pooled_trial_sharpe_var's fail-closed philosophy.

    COHORT SCOPING (#324): ``t`` and ``discovery_indices`` are WITHIN-COHORT — the caller supplies
    the current cohort's position (1..FDR_COHORT_SIZE via :func:`fdr_cohort_position`) and that
    cohort's in-cohort rejection positions. Each cohort of FDR_COHORT_SIZE binding tests is an
    independent LORD++ stream (fresh w0). This math is unchanged; only its scoping moved from a
    single lifetime stream to per-cohort restarts to defeat throughput-driven alpha-death.

    The p-value fed here must be 1 − dsr_confidence (conversion at the caller), which equals
    P(SR_true ≤ SR*) — the DSR selection-inflated null. The FDR guarantee is over that null, and is
    controlled PER COHORT of FDR_COHORT_SIZE binding tests, NOT per lifetime (see FDR_COHORT_SIZE).

    Dry-spell behavior: with no in-cohort discoveries, α_t = γ_t · w0. Because t is bounded to
    1..FDR_COHORT_SIZE, α_t is floored at γ_{FDR_COHORT_SIZE} · w0 (never collapses toward 0 from
    throughput) and restarts fresh (α_1 = γ_1 · w0) at each cohort boundary.

    Returns a CONSERVATIVE 0.0 on any degenerate input (t<1, non-finite alpha/w0, any
    discovery index ≥ t or < 1). 0.0 means p_t ≤ α_t can never be satisfied — only tightens.
    """
    if t < 1 or not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(w0) or w0 <= 0:
        return 0.0
    taus = sorted(int(tau) for tau in discovery_indices)
    if any(tau < 1 or tau >= t for tau in taus):
        return 0.0

    def _gamma(j: int) -> float:
        if j < 1 or j > len(_LORD_GAMMA):
            return 0.0
        return _LORD_GAMMA[j - 1]

    level = _gamma(t) * w0
    if taus:
        level += (alpha - w0) * _gamma(t - taus[0])
        for tau in taus[1:]:
            level += alpha * _gamma(t - tau)
    return level
