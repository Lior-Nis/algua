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

# Count-triggered cohort restarts (#324) + budget-derived recalibration (#529). The LORD++ stream
# is partitioned into consecutive, non-overlapping COHORTS of exactly FDR_COHORT_SIZE binding
# tests, assigned by ARRIVAL ORDER; each cohort runs an INDEPENDENT LORD++ stream (fresh W0,
# in-cohort t and rejection positions). This (c) check is HARD — the per-strategy DSR p-value
# (p = 1 − dsr_confidence) must be ≤ α_t; it is the across-strategy funnel-wide FDR control, ANDed
# with (a) the breadth-deflated holdout-Sharpe bar and (b) the DSR ≥ 0.95 confidence bar.
#
# WHY cohort restarts (protected constant — raising it weakens the fix). A SINGLE lifetime-global
# LORD++ stream is anti-scaling: every measured test (mostly clear-null flops, p≈1) advances
# position t, so in a dry spell α_t = γ_t·W0 → 0 as t grows — testing MORE garbage monotonically
# lowers everyone's future bar. This is INTRINSIC to a *lifetime* target on a garbage-dominated
# funnel: any valid lifetime online-FDR procedure must drive the per-test level to 0 over an
# unbounded null stream (a genuine lifetime FDR guarantee is provably incompatible with a passable
# gate here — §3.4). Bounding the count (a deterministic LORD-with-restarts, Ramdas et al. 2017;
# Zrnic et al.) floors the worst-case dry-spell level at γ_{FDR_COHORT_SIZE}·W0 INDEPENDENT of
# throughput. FDR is RE-SCOPED and EXPLICIT: controlled PER COHORT of FDR_COHORT_SIZE binding
# tests at FDR_ALPHA, NOT per lifetime — an OPERATING TARGET (shared-holdout dependence breaks the
# formal guarantee), resting on the standard LORD++ null assumptions (null p-values super-uniform;
# the LORD++ predictability/independence conditions). Cumulative exposure over K completed cohorts
# is ≈ FDR_ALPHA·K (audit-only fdr_exposure block). "Only bind passing rows" is REJECTED — it hides
# non-rejections from the multiplicity process (covert loosening). SAFFRON is insufficient here: it
# indexes γ by non-candidate count, so clear-null garbage still alpha-deaths.
#
# WHY N=8 (round-4 recalibration, #529 — budget-derived, NOT "lands α in a band"). 64 was
# calibrated for a HIGH-throughput funnel; the real funnel is LOW-throughput and 64 spread even the
# full W0 budget across 64 tests (average per-test α ≈ 0.025/64 ≈ 4e-4 — an effectively unpassable
# ~99.8%-confidence wall for a genuinely strong strategy that already clears (a) and (b)). Two
# recalibrations, both INSIDE valid LORD++:
#   1. γ-NORMALIZATION FIX. A cohort is a self-contained LORD++ stream of length ≤ FDR_COHORT_SIZE,
#      so γ MUST sum to 1 over its own FDR_COHORT_SIZE terms (was normalized over 10 000, wasting
#      ~half a 64-cohort's budget). _LORD_GAMMA is now normalized over FDR_COHORT_SIZE terms so
#      Σ_{t≤N} α_t^dry = W0 exactly.
#   2. N DERIVED FROM AN EXPLICIT NEAR-TERM CUMULATIVE-EXPOSURE BUDGET (§3.1), not from the issue's
#      [0.01,0.05] band (rejected — no N reaching it keeps the retry surface acceptable). With
#      near-term horizon H_near = 16 binding tests (≈1yr at this funnel) and target all-null
#      cumulative false-discovery S(16, N) = 1 − (1 − p_cohort)^{16/N} ≤ 0.05 (one cohort-budget's
#      worth), N=8 is the SMALLEST cohort (⇒ largest per-test α ⇒ most passable) with S(16)=0.049 ≤
#      budget. It lifts α_1 from 0.00165 → 0.00764 (a ~4.6× improvement; dsr bar 0.99835 → 0.99236).
# The budget is not merely derived — it is ENFORCED by the hard windowed PROMOTION-ELIGIBILITY
# throttle below (FDR_NEAR_TERM_BINDING_BUDGET / FDR_THROTTLE_WINDOW_DAYS). W0 stays at FDR_ALPHA/2
# (round-2's doubling to FDR_ALPHA is reverted — it would double p_cohort for a marginal early-α
# gain). Smaller N restarts the budget more often (more independent 5% cohorts ⇒ higher cumulative
# exposure RATE), so N is not free — hence the derivation from, and code-enforcement of, the budget.
FDR_COHORT_SIZE = 8

# Hard windowed PROMOTION-ELIGIBILITY throttle (#529, §3.5) — the code that turns the §3.1 budget
# from a derivation assumption into an AGENT-ENFORCED cap. In the record-gate transaction (inside
# the existing BEGIN IMMEDIATE, no schema change, no new lock), the number of PRIOR committed
# binding FDR tests within the trailing FDR_THROTTLE_WINDOW_DAYS is counted at decision time; once
# it reaches FDR_NEAR_TERM_BINDING_BUDGET, further binding tests are PROMOTION-INELIGIBLE — STILL
# commit a fail-closed binding row and STILL advance the LORD++ stream (a real FDR test ran; a
# holdout was burned) but final_passed is forced False, so they cannot promote. NOTE (factory soft
# gate): the promote path no longer writes binding rows (stats are advisory), so this machinery is
# FROZEN — preserved for future re-tightening; the --fdr-throttle-override bypass was removed with
# the rest of the flag's plumbing.
#
# WHY (protected constants — raising the budget or shrinking nothing else re-opens the exposure).
# Only the FIRST FDR_NEAR_TERM_BINDING_BUDGET (=H_near) in-window binding tests can promote; those
# 16 CONSECUTIVE binding tests touch AT MOST ⌈16/8⌉+1 = 3 always-fresh-W0 cohorts (#324). Each
# slice of a touched cohort is DOMINATED by its full fresh cohort's ≥1-false-discovery event, whose
# probability is EXACTLY p_cohort = 1 − Π_{t≤8}(1 − α_t^dry) ≈ 2.47% (dry-spell up to the first
# discovery; misalignment-/wealth-immune). Union bound over ≤3 touched fresh cohorts ⇒ the all-null
# FALSE-PROMOTION probability over ANY rolling window is ≤ 3·p_cohort ≈ 7.2% (the typical
# cohort-aligned case is the smaller 1 − (1 − p_cohort)^2 ≈ 4.9%). This is a PER-DECISION cap on the
# count of PRIOR committed in-window binding rows — NOT a retrospective property of an arbitrary
# window. Equivalently it rate-caps exposure at ≤16 promotion-eligible tests / 365d, so the
# per-cohort FDR operating target's LIFETIME exposure is RATE-CAPPED (not unbounded at burst),
# though still unbounded-in-count (a genuine lifetime FDR bound is impossible on this funnel, §3.4).
FDR_NEAR_TERM_BINDING_BUDGET = 16   # H_near — max promotion-eligible binding tests per window
FDR_THROTTLE_WINDOW_DAYS = 365      # rolling window over which H_near was estimated (§3.1)


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
    ensuring γ_j > 0 for all j (log(max(1,2))=log(2)>0 handles j=1). Dividing by the sum over the
    ``n`` terms normalizes so Σ_{j=1..n} γ_j = 1.0 (± machine-epsilon).

    #529: ``n`` is the RESTART horizon (FDR_COHORT_SIZE), not a large truncation constant. A cohort
    is a self-contained LORD++ stream of length ≤ FDR_COHORT_SIZE, so its γ must sum to 1 over its
    own FDR_COHORT_SIZE terms — guaranteeing the dry-spell budget Σ_{t≤N} α_t^dry = W0 is spent in
    full within one cohort (the pre-#529 normalization over 10 000 terms wasted ~half the budget of
    a 64-cohort). A cohort never indexes γ past position FDR_COHORT_SIZE.
    """
    raw = [
        math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(max(j, 2)))))
        for j in range(1, n + 1)
    ]
    total = sum(raw)
    return [w / total for w in raw]


# Normalized over the cohort restart horizon (#529): γ sums to 1 across FDR_COHORT_SIZE terms so a
# cohort spends its full W0 dry-spell budget. A within-cohort position is always 1..FDR_COHORT_SIZE.
_LORD_GAMMA: list[float] = _compute_lord_gamma(FDR_COHORT_SIZE)


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
