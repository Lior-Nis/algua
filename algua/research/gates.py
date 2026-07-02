"""Promotion gate (backtested -> candidate) composer (#335 extraction).

The pure-maths responsibilities that ``evaluate_gate`` composes now live in cohesive sibling
modules (mirroring the ``backtest/bootstrap.py`` / ``backtest/neff.py`` precedent):

- ``algua.research.regime``   — volatility-tertile regime robustness + CAPM idiosyncratic-alpha.
- ``algua.research.fdr_lord``  — LORD++ online-FDR γ-sequence, cohort restarts, and α_t level.
- ``algua.research.dsr``       — DSR SR*/confidence, dispersion floor, effective funnel breadth.
- ``algua.research.haircut``   — the deflated-Sharpe multiple-testing haircut.
- ``algua.research._constants`` — the shared holdout-power floor (MIN_HOLDOUT_OBSERVATIONS).

This module keeps the gate-orchestration surface: ``GateCriteria``, ``GateDecision``, the
declarative ``GateSpec``/``GATE_SPECS``, the orchestration-level constants, and ``evaluate_gate``.
It RE-EXPORTS every moved name (see ``__all__``) so ``from algua.research.gates import X`` continues
to resolve byte-identically for all existing call sites (promotion.py, factor_fdr.py, store.py, the
CLIs, and the test suite). Pure refactor — no gate criterion, threshold, or
fail-closed/tighten-only semantic changed.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.research._constants import MIN_HOLDOUT_OBSERVATIONS
from algua.research.dsr import (
    DSR_ALPHA,
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    EULER_MASCHERONI,
    MAX_BOOTSTRAP_BLOCK_LEN_FRACTION,
    MIN_CORR_OVERLAP_BARS,
    MIN_FUNNEL_FLOOR_STRATEGIES,
    MIN_N_EFF_SIBLINGS,
    RHO_BAR_SHRINKAGE_K,
    dsr_confidence,
    dsr_sr_star,
    dsr_sr_star_annualized,
    effective_funnel_breadth,
    floored_trial_var_per_period,
)
from algua.research.fdr_lord import (
    _LORD_GAMMA,
    FDR_ALPHA,
    FDR_COHORT_SIZE,
    FDR_GAMMA_TRUNCATION,
    FDR_W0,
    _compute_lord_gamma,
    fdr_cohort_position,
    lord_plus_plus_level,
)
from algua.research.haircut import sharpe_haircut
from algua.research.regime import (
    IR_MIN_APPRAISAL_RATIO,
    IR_MIN_OVERLAP_BARS,
    IR_MIN_VOL,
    MIN_REGIME_OBSERVATIONS,
    MIN_REGIME_OVERLAP_BARS,
    MIN_REGIME_SHARPE,
    MIN_REGIME_VOL,
    N_REGIMES,
    VOL_ROLLING_WINDOW,
    InformationRatioResult,
    RegimeRobustnessResult,
    RegimeSlice,
    information_ratio,
    regime_robustness_check,
    regime_splits,
)

# Re-export surface: every name below was previously defined in this module and is imported directly
# from ``algua.research.gates`` by production code and tests. Listing them in ``__all__`` keeps the
# import compatibility explicit and silences the "imported but unused" lint on the re-exports.
__all__ = [
    "ANN",
    "DSR_ALPHA",
    "DSR_BOOTSTRAP_LOWER_QUANTILE",
    "DSR_BOOTSTRAP_RESAMPLES",
    "DOMINANCE_AUDIT_MIN_PROMOTIONS",
    "DOMINANCE_AUDIT_MIN_WINDOW_DAYS",
    "DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS",
    "EULER_MASCHERONI",
    "FDR_ALPHA",
    "FDR_COHORT_SIZE",
    "FDR_GAMMA_TRUNCATION",
    "FDR_W0",
    "FUNNEL_WINDOW_DAYS",
    "GATE_SPECS",
    "GateCriteria",
    "GateDecision",
    "GateSpec",
    "InformationRatioResult",
    "IR_MIN_APPRAISAL_RATIO",
    "IR_MIN_OVERLAP_BARS",
    "IR_MIN_VOL",
    "MAX_BOOTSTRAP_BLOCK_LEN_FRACTION",
    "MIN_CORR_OVERLAP_BARS",
    "MIN_FUNNEL_FLOOR_STRATEGIES",
    "MIN_HOLDOUT_OBSERVATIONS",
    "MIN_N_EFF_SIBLINGS",
    "MIN_REGIME_OBSERVATIONS",
    "MIN_REGIME_OVERLAP_BARS",
    "MIN_REGIME_SHARPE",
    "MIN_REGIME_VOL",
    "N_REGIMES",
    "PHASE3_COMPONENT_MASK",
    "RHO_BAR_SHRINKAGE_K",
    "RegimeRobustnessResult",
    "RegimeSlice",
    "VOL_ROLLING_WINDOW",
    "WalkForwardResult",
    "_LORD_GAMMA",
    "_compute_lord_gamma",
    "dsr_confidence",
    "dsr_sr_star",
    "dsr_sr_star_annualized",
    "effective_funnel_breadth",
    "evaluate_gate",
    "fdr_cohort_position",
    "floored_trial_var_per_period",
    "information_ratio",
    "lord_plus_plus_level",
    "regime_robustness_check",
    "regime_splits",
    "sharpe_haircut",
]

# Funnel-level multiple-testing window (Wall A). Protected constant, not an agent-tunable knob
# (relaxing it would weaken the gate). Rolling window keeps the bar bounded; the wait-out-the-window
# trade-off is accepted and auditable via search_trials.created_at.
FUNNEL_WINDOW_DAYS = 90

# Dominance-audit predeclaration (#221, Phase 3 Slice 4). CODEOWNERS-protected constants — these
# thresholds are committed HERE (before any audit data accumulates) so the Slice 5
# haircut-retirement audit cannot select them post-hoc. The retirement audit (Slice 5) filters
# decision_json rows where `phase3_component_mask` has all required bits set, then checks that no
# row where `haircut_would_have_blocked=True AND dsr_raw_N_pass=True` exceeds
# DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS over >= DOMINANCE_AUDIT_MIN_PROMOTIONS promotions
# across >= DOMINANCE_AUDIT_MIN_WINDOW_DAYS of real-traffic wall time. SHADOW/AUDIT ONLY in
# Slice 4 — these constants are read only by the Slice 5 audit, not by any gate pass/fail logic.
DOMINANCE_AUDIT_MIN_PROMOTIONS = 30        # min promotions in the audit window for a verdict
DOMINANCE_AUDIT_MIN_WINDOW_DAYS = 90       # min calendar days of traffic for the audit to bind
DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS = 0  # haircut-blocked-but-DSR-passed cases allowed: none

# Phase 3 component mask — bitmask recording which Phase 3 slices are active on a gate evaluation.
# Bits 0-4 = Slices 0-4; all five set = 0b11111 = 31. Stored as `phase3_component_mask` in
# decision_json so the Slice 5 audit can filter rows where all Phase 3 components are active.
# SHADOW/AUDIT ONLY — does not affect passed.
PHASE3_COMPONENT_MASK = 0b11111  # bits 0-4 = Phase 3 slices 0-4, all five active


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> candidate. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this
    min_holdout_observations: int = MIN_HOLDOUT_OBSERVATIONS  # Wall C: power floor, fails closed


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None
    breadth_provenance: str | None = None
    base_min_holdout_sharpe: float | None = None
    effective_min_holdout_sharpe: float | None = None
    own_lifetime_combos: int | None = None
    windowed_total_combos: int | None = None
    funnel_window_days: int | None = None
    pit_ok: bool | None = None
    pit_override: bool = False
    dsr_binding: bool = False
    dsr_confidence: float | None = None
    dsr_skip_reason: str | None = None
    dsr_n_trials: int | None = None
    dsr_trial_sr_var_ann: float | None = None
    dsr_t: int | None = None
    dsr_skew: float | None = None
    dsr_raw_kurtosis: float | None = None
    # Funnel-wide dispersion floor audit (#221, Phase 3 Slice 0). Populated when dsr_binding.
    dsr_funnel_floor_var_ann: float | None = None
    dsr_funnel_floor_n_strategies: int | None = None
    dsr_funnel_floor_n_total_rows: int | None = None
    # LORD++ FDR accounting (#220, Phase 2). Populated by run_gate() AFTER the atomic store
    # call (α_t is only known inside the write transaction). Non-binding rows leave all fdr_*
    # fields at their defaults (False/None); fdr_skip_reason explains why FDR was omitted.
    fdr_binding: bool = False
    fdr_p_value: float | None = None
    fdr_alpha_level: float | None = None
    fdr_test_index: int | None = None      # WITHIN-COHORT position (1..FDR_COHORT_SIZE), #324
    fdr_rejected: bool | None = None
    fdr_skip_reason: str | None = None
    # Cohort restart + cumulative-exposure audit (#324). Populated by run_gate on binding rows.
    # fdr_cohort: this row's 0-based cohort index. The remaining fields are AUDIT-ONLY (never
    # change pass/fail) and make the per-cohort re-scoping honest: they surface that FDR is
    # controlled per cohort of FDR_COHORT_SIZE binding tests, NOT per lifetime, and how much
    # cumulative exposure has accrued. fdr_expected_false_discoveries = FDR_ALPHA *
    # fdr_cohorts_completed is the honest upper bound on cumulative expected false discoveries over
    # completed independent cohorts (NOT conditioned on cohorts-with-discoveries, which would be
    # post-selection and understate exposure).
    fdr_cohort: int | None = None
    fdr_cohorts_completed: int | None = None
    fdr_binding_tests: int | None = None
    fdr_discoveries: int | None = None
    fdr_expected_false_discoveries: float | None = None
    # Returns-persistence audit (#221 Slice 1). Non-binding: True iff run_gate wrote a
    # holdout_returns row for this evaluation. False for pre-Slice-1 promotions (omit-not-fail).
    returns_available: bool = False
    # Serial-dependence bootstrap audit (#221 Slice 2). Binding only when dsr_binding AND
    # bootstrap_binding are both True (gates computes effective_bootstrap_binding = dsr_binding
    # AND bootstrap_binding). The caller (promotion.py) sets bootstrap_binding only when measured
    # breadth AND the OOS vector is present; gates binds on dsr_binding AND bootstrap_binding.
    # Non-binding: all dsr_bootstrap_* fields remain False/None (omit-not-fail).
    dsr_bootstrap_binding: bool = False
    dsr_bootstrap_lower: float | None = None
    dsr_bootstrap_seed: int | None = None
    dsr_bootstrap_b: int | None = None
    dsr_bootstrap_block_len: int | None = None
    # Effective independent trials audit (#221 Slice 3). SHADOW-ONLY: recorded, never fed into the
    # binding dsr_confidence (which keeps n_trials = raw N). Populated by promotion.run_gate.
    dsr_n_eff: int | None = None
    dsr_rho_bar: float | None = None
    dsr_n_siblings: int | None = None
    # Multi-regime robustness audit (#221 Slice 4). Populated by evaluate_gate when the check
    # binds (holdout_returns + market_returns present, overlap >= MIN_REGIME_OVERLAP_BARS).
    # When omitted (unavailable / insufficient overlap): regime_method explains why.
    regime_method: str | None = None          # "vol_tertile"|"unavailable"|"insufficient_overlap"
    n_regimes_attempted: int | None = None    # total regime slices fed to robustness_check
    n_regimes_surviving: int | None = None    # regimes that cleared power + vol floors
    per_regime_sharpes: list[float | None] | None = None  # per-slice Sharpe; None for dropped
    regime_robustness_binding: bool = False   # True iff the check was appended to checks
    # Market-beta / idiosyncratic-alpha screen (#328). Populated by evaluate_gate when the check
    # binds (holdout_returns + market_returns present, overlap >= IR_MIN_OVERLAP_BARS). When omitted
    # (unavailable / insufficient overlap): ir_method explains why and ir_binding is False.
    # market_beta is recorded for AUDIT ONLY — there is no hard beta cap (it would over-reject
    # legitimate high-alpha/high-beta strategies); the gate binds on the appraisal ratio.
    ir_method: str | None = None              # capm_appraisal | unavailable | insufficient_overlap
    ir_binding: bool = False                  # True iff the idiosyncratic_alpha check was appended
    ir_overlap_n: int | None = None           # date-joined holdout bars
    market_beta: float | None = None          # OLS slope vs the market benchmark (audit only)
    ir_alpha_ann: float | None = None         # annualized residual alpha (intercept)
    ir_residual_vol_ann: float | None = None  # annualized idiosyncratic (residual) volatility
    appraisal_ratio: float | None = None      # annualized alpha / residual vol (the gated value)
    # Dominance-audit shadow fields (#221 Slice 4). SHADOW/AUDIT ONLY — not in checks, not in
    # passed. Recorded on every evaluate_gate call so the Slice 5 retirement audit can accumulate
    # real-traffic statistics with pre-committed thresholds.
    # haircut_would_have_blocked: True iff the holdout Sharpe cleared the BASE bar but failed the
    #   haircut-inflated bar (the haircut is what blocked an otherwise-passing holdout). When the
    #   effective bar is +inf (degenerate zero-length holdout), True iff the base bar passed.
    # phase3_component_mask: PHASE3_COMPONENT_MASK (0b11111 = 31) — all five Slice 0-4 components
    #   active. Allows the audit to filter rows where the full Phase 3 stack was live.
    haircut_would_have_blocked: bool = False
    phase3_component_mask: int | None = None

    def to_dict(self) -> dict[str, Any]:
        # A degenerate holdout drives the effective bar to inf (fail-closed); null it so the
        # payload stays JSON-clean, mirroring how non-finite check values are nulled.
        eff = self.effective_min_holdout_sharpe

        def _f(x: float | None) -> float | None:
            return x if x is None or math.isfinite(x) else None

        return {
            "passed": self.passed,
            "checks": self.checks,
            "n_combos": self.n_combos,
            "breadth_provenance": self.breadth_provenance,
            "base_min_holdout_sharpe": self.base_min_holdout_sharpe,
            "effective_min_holdout_sharpe": (
                eff if eff is None or math.isfinite(eff) else None
            ),
            "own_lifetime_combos": self.own_lifetime_combos,
            "windowed_total_combos": self.windowed_total_combos,
            "funnel_window_days": self.funnel_window_days,
            "pit_ok": self.pit_ok,
            "pit_override": self.pit_override,
            "dsr_binding": self.dsr_binding,
            "dsr_confidence": _f(self.dsr_confidence),
            "dsr_skip_reason": self.dsr_skip_reason,
            "dsr_n_trials": self.dsr_n_trials,
            "dsr_trial_sr_var_ann": _f(self.dsr_trial_sr_var_ann),
            "dsr_t": self.dsr_t,
            "dsr_skew": _f(self.dsr_skew),
            "dsr_raw_kurtosis": _f(self.dsr_raw_kurtosis),
            "dsr_funnel_floor_var_ann": _f(self.dsr_funnel_floor_var_ann),
            "dsr_funnel_floor_n_strategies": self.dsr_funnel_floor_n_strategies,
            "dsr_funnel_floor_n_total_rows": self.dsr_funnel_floor_n_total_rows,
            "fdr_binding": self.fdr_binding,
            "fdr_p_value": _f(self.fdr_p_value),
            "fdr_alpha_level": _f(self.fdr_alpha_level),
            "fdr_test_index": self.fdr_test_index,
            "fdr_rejected": self.fdr_rejected,
            "fdr_skip_reason": self.fdr_skip_reason,
            "fdr_cohort": self.fdr_cohort,
            "fdr_cohorts_completed": self.fdr_cohorts_completed,
            "fdr_binding_tests": self.fdr_binding_tests,
            "fdr_discoveries": self.fdr_discoveries,
            "fdr_expected_false_discoveries": _f(self.fdr_expected_false_discoveries),
            "returns_available": self.returns_available,
            "dsr_bootstrap_binding": self.dsr_bootstrap_binding,
            "dsr_bootstrap_lower": _f(self.dsr_bootstrap_lower),
            "dsr_bootstrap_seed": self.dsr_bootstrap_seed,
            "dsr_bootstrap_b": self.dsr_bootstrap_b,
            "dsr_bootstrap_block_len": self.dsr_bootstrap_block_len,
            "dsr_n_eff": self.dsr_n_eff,
            "dsr_rho_bar": _f(self.dsr_rho_bar),
            "dsr_n_siblings": self.dsr_n_siblings,
            "regime_method": self.regime_method,
            "n_regimes_attempted": self.n_regimes_attempted,
            "n_regimes_surviving": self.n_regimes_surviving,
            "per_regime_sharpes": (
                [None if (x is None or not math.isfinite(x)) else x
                 for x in self.per_regime_sharpes]
                if self.per_regime_sharpes is not None else None
            ),
            "regime_robustness_binding": self.regime_robustness_binding,
            # Market-beta / idiosyncratic-alpha screen (#328)
            "ir_method": self.ir_method,
            "ir_binding": self.ir_binding,
            "ir_overlap_n": self.ir_overlap_n,
            "market_beta": _f(self.market_beta),
            "ir_alpha_ann": _f(self.ir_alpha_ann),
            "ir_residual_vol_ann": _f(self.ir_residual_vol_ann),
            "appraisal_ratio": _f(self.appraisal_ratio),
            # Dominance-audit shadow fields (#221 Slice 4)
            "haircut_would_have_blocked": self.haircut_would_have_blocked,
            "phase3_component_mask": self.phase3_component_mask,
        }


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
}


@dataclass(frozen=True)
class GateSpec:
    """Declarative description of one gate check (#40).

    Adding a gate is appending a spec here — no edits to the evaluation loop. `source` and
    `metric_key` locate the value on the WalkForwardResult; `threshold_attr` names the
    GateCriteria field; `op` is the comparison operator.
    """

    name: str
    source: str  # "holdout" -> wf.holdout_metrics, "stability" -> wf.stability
    metric_key: str
    threshold_attr: str
    op: str


GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("holdout_sharpe", "holdout", "sharpe", "min_holdout_sharpe", ">="),
    GateSpec("holdout_return", "holdout", "total_return", "min_holdout_return", ">"),
    GateSpec("pct_positive_windows", "stability", "pct_positive_windows",
             "min_pct_positive_windows", ">="),
    GateSpec("min_window_sharpe", "stability", "min_sharpe", "min_window_sharpe", ">="),
    GateSpec("min_holdout_observations", "holdout", "n_bars", "min_holdout_observations", ">="),
)


_HOLDOUT_SHARPE_SPEC = "holdout_sharpe"


def evaluate_gate(
    wf: WalkForwardResult,
    criteria: GateCriteria,
    *,
    n_combos: int | None = None,
    breadth_provenance: str | None = None,
    pit_ok: bool,
    allow_non_pit: bool = False,
    own_lifetime_combos: int | None = None,
    windowed_total_combos: int | None = None,
    funnel_window_days: int | None = None,
    dsr_binding: bool = False,
    dsr_trial_var_ann: float | None = None,
    dsr_funnel_floor_var_ann: float | None = None,
    dsr_funnel_floor_n_strategies: int | None = None,
    dsr_funnel_floor_n_total_rows: int | None = None,
    bootstrap_binding: bool = False,
    bootstrap_lower_confidence: float | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_b: int | None = None,
    bootstrap_block_len: int | None = None,
    market_returns: tuple[list[float], list[str]] | None = None,
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects.

    Driven by GATE_SPECS so the metric dict and the gate stay in sync without hand editing.

    The holdout-Sharpe threshold is DEFLATED by ``sharpe_haircut(n_combos, T)`` where T is the
    holdout sample size (``wf.holdout_metrics["n_bars"]``). This is the multiple-testing defense:
    the more combos the agent searched, the higher the bar the selected strategy must clear on the
    untouched holdout. At N=1 (or no breadth) the haircut is 0 and the effective bar equals the
    base. The other gate checks are left untouched. ``breadth_provenance`` ("measured"/"declared")
    is carried into the decision for the audit trail; it does not change the math.
    """
    sources = {"holdout": wf.holdout_metrics, "stability": wf.stability}
    base_holdout_sharpe = float(criteria.min_holdout_sharpe)
    haircut = (
        sharpe_haircut(n_combos, int(wf.holdout_metrics["n_bars"]))
        if n_combos is not None
        else 0.0
    )
    effective_holdout_sharpe = base_holdout_sharpe + haircut

    checks: list[dict[str, Any]] = []
    for spec in GATE_SPECS:
        value = float(sources[spec.source][spec.metric_key])
        threshold = float(getattr(criteria, spec.threshold_attr))
        if spec.name == _HOLDOUT_SHARPE_SPEC:
            threshold = effective_holdout_sharpe
        # A non-finite EFFECTIVE THRESHOLD means a degenerate (zero-length) holdout drove the
        # haircut to inf: there is no out-of-sample evidence, so the check FAILS CLOSED — never a
        # pass — and the inf threshold is nulled out of the payload to keep it JSON-clean.
        if not math.isfinite(threshold):
            checks.append({"name": spec.name, "value": None, "threshold": None,
                           "op": spec.op, "passed": False})
            continue
        # A non-finite metric (inf trivially clears >=/>, NaN is never a real result) is a
        # gate failure, never a pass — and is never recorded as a NaN/inf in the payload.
        if not math.isfinite(value):
            checks.append({"name": spec.name, "value": None, "threshold": threshold,
                           "op": spec.op, "passed": False})
            continue
        passed = _OPS[spec.op](value, threshold)
        checks.append({"name": spec.name, "value": value, "threshold": threshold,
                       "op": spec.op, "passed": bool(passed)})
    # PIT precondition (Wall B): boolean, not a metric comparison. pit_ok is computed by the
    # protected promotion orchestrator (presence + coverage). Non-PIT fails closed unless a human
    # passed allow_non_pit (recorded as an audited override).
    pit_passed = bool(pit_ok or allow_non_pit)
    pit_override = bool((not pit_ok) and allow_non_pit)
    checks.append({"name": "pit_required", "passed": pit_passed,
                   "pit_ok": bool(pit_ok), "override": "non_pit" if pit_override else None})
    # DSR evidence (#211): a tighten-only AND-check, appended ONLY when binding (measured trial
    # dispersion is available). When not binding it is omitted entirely so `passed` is unchanged.
    # Unit conversion lives here: holdout Sharpe and trial variance are ANNUALIZED; DSR per-period.
    dsr_conf: float | None = None
    dsr_skip_reason: str | None = None
    n_for_dsr = n_combos if n_combos is not None else 1
    t_hold = int(wf.holdout_metrics["n_bars"])
    skew = float(wf.holdout_metrics.get("skewness", 0.0))
    raw_kurt = float(wf.holdout_metrics.get("kurtosis", 3.0))
    sr_obs_ann = float(wf.holdout_metrics["sharpe"])
    # The bootstrap check is only armed when BOTH dsr_binding AND bootstrap_binding are True.
    # The caller (promotion.py) sets bootstrap_binding only when measured breadth AND the OOS
    # vector is present; gates binds on dsr_binding AND bootstrap_binding.
    # Using effective_bootstrap_binding throughout keeps the audit state self-consistent: if
    # dsr_binding=False the bootstrap check is never appended and all bootstrap audit fields
    # remain None/False, regardless of what the caller passed for bootstrap_binding.
    effective_bootstrap_binding = bool(dsr_binding and bootstrap_binding)
    if dsr_binding:
        var_pp = (dsr_trial_var_ann / ANN) if dsr_trial_var_ann is not None else None
        floor_pp = (
            dsr_funnel_floor_var_ann / ANN
            if dsr_funnel_floor_var_ann is not None and math.isfinite(dsr_funnel_floor_var_ann)
            else None
        )
        if var_pp is not None and math.isfinite(var_pp):
            dsr_conf = dsr_confidence(
                sr_obs_ann / math.sqrt(ANN), t_hold, skew, raw_kurt, n_for_dsr, var_pp,
                funnel_floor_var_per_period=floor_pp)
        passed_dsr = dsr_conf is not None and dsr_conf >= (1.0 - DSR_ALPHA)
        dsr_value = dsr_conf if (dsr_conf is not None and math.isfinite(dsr_conf)) else None
        checks.append({"name": "dsr_evidence", "value": dsr_value,
                       "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(passed_dsr)})
        if dsr_conf is None:
            # measured sweep exists but stats missing -> fail closed
            dsr_skip_reason = "no_dispersion"
        if effective_bootstrap_binding:
            boot_pass = (bootstrap_lower_confidence is not None
                         and bootstrap_lower_confidence >= (1.0 - DSR_ALPHA))
            boot_value = (bootstrap_lower_confidence
                          if (bootstrap_lower_confidence is not None
                              and math.isfinite(bootstrap_lower_confidence)) else None)
            checks.append({"name": "dsr_bootstrap", "value": boot_value,
                           "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(boot_pass)})
    else:
        dsr_skip_reason = "no_measured_dispersion"

    # Multi-regime robustness (#221 Slice 4): a tighten-only AND-check, appended ONLY when the
    # market series is available AND wf.holdout_returns is present AND overlap is sufficient.
    # When omitted (unavailable / insufficient_overlap), checks is byte-identical to today so
    # every existing promotion test (no market_returns) is unchanged.
    # Binding is INDEPENDENT of dsr_binding — purely a holdout-robustness check.
    regime_method: str | None = None
    n_regimes_attempted: int | None = None
    n_regimes_surviving: int | None = None
    per_regime_sharpes_out: list[float | None] | None = None
    regime_robustness_binding = False

    holdout_ret_vec = wf.holdout_returns  # tuple[list[float], list[str]] | None
    if holdout_ret_vec is None or market_returns is None:
        regime_method = "unavailable"
    else:
        strat_rets_list, strat_dates_list = holdout_ret_vec
        mkt_rets_list, mkt_dates_list = market_returns
        slices, overlap_n = regime_splits(
            strat_rets_list,
            strat_dates_list,
            mkt_rets_list,
            mkt_dates_list,
            n_regimes=N_REGIMES,
            vol_window=VOL_ROLLING_WINDOW,
        )
        if overlap_n < MIN_REGIME_OVERLAP_BARS:
            regime_method = "insufficient_overlap"
        else:
            res = regime_robustness_check(slices, min_obs=MIN_REGIME_OBSERVATIONS,
                                          min_sharpe=MIN_REGIME_SHARPE)
            checks.append({"name": "regime_robustness", "value": None,
                           "threshold": MIN_REGIME_SHARPE, "op": ">=",
                           "passed": bool(res.passed)})
            regime_method = "vol_tertile"
            regime_robustness_binding = True
            n_regimes_attempted = res.n_attempted
            n_regimes_surviving = res.n_surviving
            per_regime_sharpes_out = res.per_regime_sharpes

    # Market-beta / idiosyncratic-alpha screen (#328): a tighten-only AND-check appended ONLY when
    # holdout_returns + market_returns are present AND the date-join overlap >= IR_MIN_OVERLAP_BARS.
    # Omit-not-fail otherwise (byte-identical to today for market-less runs) — consistent with the
    # regime sibling. Armed-but-degenerate (constant mkt, zero residual, non-finite) fails closed.
    # Binding is INDEPENDENT of dsr_binding and regime binding — purely a beta-attribution screen.
    ir_method: str | None = None
    ir_binding = False
    ir_overlap_n: int | None = None
    market_beta: float | None = None
    ir_alpha_ann: float | None = None
    ir_residual_vol_ann: float | None = None
    appraisal_ratio: float | None = None
    if holdout_ret_vec is None or market_returns is None:
        ir_method = "unavailable"
    else:
        ir_strat_rets, ir_strat_dates = holdout_ret_vec
        ir_mkt_rets, ir_mkt_dates = market_returns
        ir = information_ratio(ir_strat_rets, ir_strat_dates, ir_mkt_rets, ir_mkt_dates)
        ir_overlap_n = ir.overlap_n
        if ir.overlap_n < IR_MIN_OVERLAP_BARS:
            ir_method = "insufficient_overlap"
        else:
            market_beta = ir.market_beta
            ir_alpha_ann = ir.alpha_ann
            ir_residual_vol_ann = ir.residual_vol_ann
            appraisal_ratio = ir.appraisal_ratio
            ir_passed = (
                (not ir.degenerate)
                and appraisal_ratio is not None
                and appraisal_ratio >= IR_MIN_APPRAISAL_RATIO
            )
            ir_value = (
                appraisal_ratio
                if (appraisal_ratio is not None and math.isfinite(appraisal_ratio))
                else None
            )
            checks.append({"name": "idiosyncratic_alpha", "value": ir_value,
                           "threshold": IR_MIN_APPRAISAL_RATIO, "op": ">=",
                           "passed": bool(ir_passed)})
            ir_method = "capm_appraisal"
            ir_binding = True

    # Dominance-audit shadow field (#221 Slice 4): did the haircut alone block an otherwise-passing
    # holdout? True iff the holdout Sharpe clears the BASE bar but fails the haircut-inflated bar.
    # The haircut is still BINDING (this is shadow recording only). When the effective bar is +inf
    # (degenerate zero-length holdout drove the haircut to inf), the haircut blocks everything the
    # base bar passed, so True iff the base bar passed. SHADOW/AUDIT ONLY — does not affect passed.
    haircut_would_have_blocked = bool(
        sr_obs_ann >= base_holdout_sharpe
        and (not math.isfinite(effective_holdout_sharpe) or sr_obs_ann < effective_holdout_sharpe)
    )

    return GateDecision(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        n_combos=n_combos,
        breadth_provenance=breadth_provenance,
        base_min_holdout_sharpe=base_holdout_sharpe,
        effective_min_holdout_sharpe=effective_holdout_sharpe,
        own_lifetime_combos=own_lifetime_combos,
        windowed_total_combos=windowed_total_combos,
        funnel_window_days=funnel_window_days,
        pit_ok=bool(pit_ok),
        pit_override=pit_override,
        dsr_binding=bool(dsr_binding),
        dsr_confidence=dsr_conf,
        dsr_skip_reason=dsr_skip_reason,
        dsr_n_trials=(n_for_dsr if dsr_binding else None),
        dsr_trial_sr_var_ann=(dsr_trial_var_ann if dsr_binding else None),
        dsr_t=(t_hold if dsr_binding else None),
        dsr_skew=(skew if dsr_binding else None),
        dsr_raw_kurtosis=(raw_kurt if dsr_binding else None),
        dsr_funnel_floor_var_ann=(dsr_funnel_floor_var_ann if dsr_binding else None),
        dsr_funnel_floor_n_strategies=(dsr_funnel_floor_n_strategies if dsr_binding else None),
        dsr_funnel_floor_n_total_rows=(dsr_funnel_floor_n_total_rows if dsr_binding else None),
        dsr_bootstrap_binding=effective_bootstrap_binding,
        dsr_bootstrap_lower=(bootstrap_lower_confidence if effective_bootstrap_binding else None),
        dsr_bootstrap_seed=(bootstrap_seed if effective_bootstrap_binding else None),
        dsr_bootstrap_b=(bootstrap_b if effective_bootstrap_binding else None),
        dsr_bootstrap_block_len=(bootstrap_block_len if effective_bootstrap_binding else None),
        regime_method=regime_method,
        n_regimes_attempted=n_regimes_attempted,
        n_regimes_surviving=n_regimes_surviving,
        per_regime_sharpes=per_regime_sharpes_out,
        regime_robustness_binding=regime_robustness_binding,
        ir_method=ir_method,
        ir_binding=ir_binding,
        ir_overlap_n=ir_overlap_n,
        market_beta=market_beta,
        ir_alpha_ann=ir_alpha_ann,
        ir_residual_vol_ann=ir_residual_vol_ann,
        appraisal_ratio=appraisal_ratio,
        haircut_would_have_blocked=haircut_would_have_blocked,
        phase3_component_mask=PHASE3_COMPONENT_MASK,
    )
