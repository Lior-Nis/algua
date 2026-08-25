"""``GateDecision`` -> dict serialization (extracted from ``algua/research/gates.py``, stage 7
task 4).

Pure mapping from a ``GateDecision`` to the exact JSON-safe payload persisted as
``gate_evaluations.decision_json`` (``algua/registry/promotion.py``) and threaded through the CLI
and the dominance-audit / negative-results consumers. Non-finite floats are nulled so the payload
stays JSON-clean (mirrors how non-finite check values are already nulled in ``checks``).

CODEOWNERS-protected (see ``CODEOWNERS`` + ``tests/test_repo_hygiene.py``'s
``INTEGRITY_CRITICAL_MODULES``): this function decides exactly which fields reach the persisted
decision record. It cannot flip ``passed`` (that is decided upstream in ``evaluate_gate``), but it
COULD silently drop an audit-only field — e.g. one of the dominance-audit shadow fields
(``haircut_would_have_blocked``, ``phase3_component_mask``) that the Slice 5 retirement audit reads
from ``decision_json`` — which would quietly disarm that downstream guard with no error anywhere.
That is exactly the "an agent could change this alone and weaken a guarantee" case the CODEOWNERS
file exists to catch, so this module is walled the same way ``gates.py`` is.

``GateDecision.to_dict()`` delegates to :func:`gate_decision_to_dict`; existing call sites keep
using ``decision.to_dict()`` unchanged.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only: avoids a runtime cycle with ``algua.research.gates`` (which imports
    # ``gate_decision_to_dict`` from here to implement ``GateDecision.to_dict``).
    from algua.research.gates import GateDecision


def gate_decision_to_dict(decision: GateDecision) -> dict[str, Any]:
    """The exact dict payload ``GateDecision.to_dict()`` returns."""
    # A degenerate holdout drives the effective bar to inf (fail-closed); null it so the
    # payload stays JSON-clean, mirroring how non-finite check values are nulled.
    eff = decision.effective_min_holdout_sharpe

    def _f(x: float | None) -> float | None:
        return x if x is None or math.isfinite(x) else None

    return {
        "passed": decision.passed,
        "checks": decision.checks,
        "n_combos": decision.n_combos,
        "breadth_provenance": decision.breadth_provenance,
        "base_min_holdout_sharpe": decision.base_min_holdout_sharpe,
        "effective_min_holdout_sharpe": (
            eff if eff is None or math.isfinite(eff) else None
        ),
        "own_lifetime_combos": decision.own_lifetime_combos,
        "windowed_total_combos": decision.windowed_total_combos,
        "funnel_window_days": decision.funnel_window_days,
        "pit_ok": decision.pit_ok,
        "pit_override": decision.pit_override,
        "dsr_binding": decision.dsr_binding,
        "dsr_confidence": _f(decision.dsr_confidence),
        "dsr_skip_reason": decision.dsr_skip_reason,
        "dsr_n_trials": decision.dsr_n_trials,
        "dsr_trial_sr_var_ann": _f(decision.dsr_trial_sr_var_ann),
        "dsr_t": decision.dsr_t,
        "dsr_skew": _f(decision.dsr_skew),
        "dsr_raw_kurtosis": _f(decision.dsr_raw_kurtosis),
        "dsr_funnel_floor_var_ann": _f(decision.dsr_funnel_floor_var_ann),
        "dsr_funnel_floor_n_strategies": decision.dsr_funnel_floor_n_strategies,
        "dsr_funnel_floor_n_total_rows": decision.dsr_funnel_floor_n_total_rows,
        "fdr_binding": decision.fdr_binding,
        "fdr_p_value": _f(decision.fdr_p_value),
        "fdr_alpha_level": _f(decision.fdr_alpha_level),
        "fdr_test_index": decision.fdr_test_index,
        "fdr_rejected": decision.fdr_rejected,
        "fdr_skip_reason": decision.fdr_skip_reason,
        "fdr_cohort": decision.fdr_cohort,
        "fdr_cohorts_completed": decision.fdr_cohorts_completed,
        "fdr_binding_tests": decision.fdr_binding_tests,
        "fdr_discoveries": decision.fdr_discoveries,
        "fdr_expected_false_discoveries": _f(decision.fdr_expected_false_discoveries),
        "returns_available": decision.returns_available,
        "dsr_bootstrap_binding": decision.dsr_bootstrap_binding,
        "dsr_bootstrap_lower": _f(decision.dsr_bootstrap_lower),
        "dsr_bootstrap_seed": decision.dsr_bootstrap_seed,
        "dsr_bootstrap_b": decision.dsr_bootstrap_b,
        "dsr_bootstrap_block_len": decision.dsr_bootstrap_block_len,
        "dsr_n_eff": decision.dsr_n_eff,
        "dsr_rho_bar": _f(decision.dsr_rho_bar),
        "dsr_n_siblings": decision.dsr_n_siblings,
        "regime_method": decision.regime_method,
        "n_regimes_attempted": decision.n_regimes_attempted,
        "n_regimes_surviving": decision.n_regimes_surviving,
        "per_regime_sharpes": (
            [None if (x is None or not math.isfinite(x)) else x
             for x in decision.per_regime_sharpes]
            if decision.per_regime_sharpes is not None else None
        ),
        "regime_robustness_binding": decision.regime_robustness_binding,
        # Market-beta / idiosyncratic-alpha screen (#328)
        "ir_method": decision.ir_method,
        "ir_binding": decision.ir_binding,
        "ir_overlap_n": decision.ir_overlap_n,
        "market_beta": _f(decision.market_beta),
        "ir_alpha_ann": _f(decision.ir_alpha_ann),
        "ir_residual_vol_ann": _f(decision.ir_residual_vol_ann),
        "appraisal_ratio": _f(decision.appraisal_ratio),
        # Dominance-audit shadow fields (#221 Slice 4)
        "haircut_would_have_blocked": decision.haircut_would_have_blocked,
        "phase3_component_mask": decision.phase3_component_mask,
    }
