import itertools
import math

import pytest

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    _LORD_GAMMA,
    DSR_ALPHA,
    EULER_MASCHERONI,
    FDR_ALPHA,
    FDR_COHORT_SIZE,
    FDR_NEAR_TERM_BINDING_BUDGET,
    FDR_THROTTLE_WINDOW_DAYS,
    FDR_W0,
    FUNNEL_WINDOW_DAYS,
    MIN_HOLDOUT_OBSERVATIONS,
    GateCriteria,
    GateDecision,
    dsr_confidence,
    effective_funnel_breadth,
    evaluate_gate,
    lord_plus_plus_level,
    sharpe_haircut,
)


def _wf(holdout_sharpe=0.8, holdout_return=0.05, pct_positive=0.75, min_sharpe=0.1, n_bars=100):
    return WalkForwardResult(
        strategy="ew", config_hash="abc", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": n_bars,
                         "total_return": holdout_return, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": holdout_sharpe, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.0, "std_sharpe": 0.3, "min_sharpe": min_sharpe,
                   "pct_positive_windows": pct_positive},
    )


def test_all_thresholds_met_passes():
    # n_combos=1 ⇒ zero haircut, so a clean result clears the base bar.
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=1, pit_ok=True)
    assert isinstance(d, GateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == {
        "holdout_sharpe", "holdout_return", "pct_positive_windows", "min_window_sharpe",
        "min_holdout_observations", "holdout_sharpe_floor", "pit_required"}
    assert all(c["passed"] for c in d.checks)
    assert d.n_combos == 1


# --- the factory soft gate: binding/advisory partition (2026-08-10 spec) ---------------------

# The integrity floor — the ONLY checks allowed to veto `passed`.
_BINDING_FLOOR = {"min_holdout_observations", "holdout_sharpe_floor", "pit_required"}


def test_binding_partition_is_floor_only():
    # Arm everything armable without market data: DSR + spec checks. The set of checks NOT marked
    # advisory must be exactly the integrity floor.
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=10, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=0.04)
    binding = {c["name"] for c in d.checks if not c.get("advisory")}
    assert binding == _BINDING_FLOOR
    advisory = {c["name"] for c in d.checks if c.get("advisory") is True}
    assert advisory == {"holdout_sharpe", "holdout_return", "pct_positive_windows",
                        "min_window_sharpe", "dsr_evidence"}


def test_passed_composes_binding_checks_only():
    # A decision failing EVERY statistical check but clearing the floor must PASS; a decision
    # clearing every stat but failing any floor check must FAIL.
    all_stats_fail = evaluate_gate(
        _wf(holdout_sharpe=0.01, holdout_return=-0.05, pct_positive=0.0, min_sharpe=-2.0),
        GateCriteria(), n_combos=200, pit_ok=True, dsr_binding=True, dsr_trial_var_ann=400.0)
    assert all_stats_fail.passed is True
    failed = {c["name"] for c in all_stats_fail.checks if not c["passed"]}
    assert failed  # the stats really did fail
    assert all(c.get("advisory") is True
               for c in all_stats_fail.checks if not c["passed"])
    # Floor failure (short holdout) vetoes even a perfect stat sheet.
    floor_fail = evaluate_gate(_wf(n_bars=10), GateCriteria(), n_combos=1, pit_ok=True)
    assert floor_fail.passed is False


def test_holdout_sharpe_floor_strictly_positive():
    # value = the RAW holdout Sharpe (same value the holdout_sharpe check scores), threshold 0.0,
    # strict >. Zero and negative fail; positive passes; the check is BINDING (no advisory key).
    pos = evaluate_gate(_wf(holdout_sharpe=0.01), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    floor = next(c for c in pos.checks if c["name"] == "holdout_sharpe_floor")
    assert floor == {"name": "holdout_sharpe_floor", "value": 0.01, "threshold": 0.0,
                     "op": ">", "passed": True}
    assert pos.passed is True
    for bad in (0.0, -0.5):
        d = evaluate_gate(_wf(holdout_sharpe=bad), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
        floor = next(c for c in d.checks if c["name"] == "holdout_sharpe_floor")
        assert floor["passed"] is False and d.passed is False


def test_holdout_sharpe_floor_non_finite_fails_closed():
    for bad in (float("nan"), float("inf"), float("-inf")):
        d = evaluate_gate(_wf(holdout_sharpe=bad), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
        floor = next(c for c in d.checks if c["name"] == "holdout_sharpe_floor")
        assert floor["passed"] is False
        assert floor["value"] is None  # never a raw NaN/inf in the payload
        assert d.passed is False


def test_each_floor_check_alone_vetoes():
    # Each binding floor check fail-closes the gate on its own, with every stat passing.
    short = evaluate_gate(_wf(n_bars=62), GateCriteria(), n_combos=1, pit_ok=True)
    assert short.passed is False
    assert next(c for c in short.checks
                if c["name"] == "min_holdout_observations")["passed"] is False
    non_pit = evaluate_gate(_wf(), GateCriteria(), n_combos=1, pit_ok=False)
    assert non_pit.passed is False
    assert next(c for c in non_pit.checks if c["name"] == "pit_required")["passed"] is False
    losing = evaluate_gate(_wf(holdout_sharpe=-0.1), GateCriteria(**_LAX), n_combos=1,
                           pit_ok=True)
    assert losing.passed is False
    assert next(c for c in losing.checks
                if c["name"] == "holdout_sharpe_floor")["passed"] is False


# --- multiple-testing haircut (deflated Sharpe) ---------------------------------------------


def test_haircut_is_zero_at_n1():
    assert sharpe_haircut(1, 100) == 0.0


def test_haircut_known_value_n9_t100():
    # sqrt(2*ln(9)) * sqrt(ANN) / sqrt(T), ANN=252, T=100.
    expected = math.sqrt(2.0 * math.log(9)) * math.sqrt(ANN) / math.sqrt(100)
    assert sharpe_haircut(9, 100) == expected
    assert math.isclose(expected, 3.32776, rel_tol=1e-4)


def test_haircut_monotonic_in_n():
    prev = -1.0
    for n in (1, 2, 4, 9, 50, 200):
        h = sharpe_haircut(n, 100)
        assert h >= prev
        prev = h


def test_haircut_uses_holdout_sample_size():
    # Larger T ⇒ tighter standard error ⇒ smaller haircut.
    assert sharpe_haircut(50, 400) < sharpe_haircut(50, 100)


def test_haircut_fails_closed_on_degenerate_holdout():
    # A zero-length (or negative) holdout has NO out-of-sample evidence: the haircut must NOT
    # collapse to 0 (which would waive the multiple-testing penalty entirely). It fails closed by
    # returning +inf, so the effective holdout-Sharpe bar becomes unreachable.
    assert sharpe_haircut(9, 0) == math.inf
    assert sharpe_haircut(1, 0) == math.inf
    assert sharpe_haircut(9, -5) == math.inf


def test_degenerate_holdout_makes_gate_fail_not_pass():
    # n_bars=0 -> the holdout_sharpe check must FAIL (never pass), even with an otherwise stellar
    # holdout Sharpe, because a zero-length holdout is no evidence at all.
    d = evaluate_gate(_wf(holdout_sharpe=99.0, n_bars=0), GateCriteria(), n_combos=9, pit_ok=True)
    assert d.passed is False
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["passed"] is False


def test_degenerate_holdout_to_dict_nulls_inf_threshold_and_is_json_serializable():
    import json

    # n_bars=0 drives effective_min_holdout_sharpe to inf; to_dict() must null it so inf never
    # reaches JSON — regression guard against re-leaking a non-finite value into the payload.
    d = evaluate_gate(_wf(holdout_sharpe=99.0, n_bars=0), GateCriteria(), n_combos=9, pit_ok=True)
    d_dict = d.to_dict()
    assert d_dict["effective_min_holdout_sharpe"] is None
    json.dumps(d_dict)  # must not raise (no inf/NaN in payload)


def test_n1_effective_equals_base():
    d = evaluate_gate(_wf(), GateCriteria(min_holdout_sharpe=0.5), n_combos=1, pit_ok=True)
    assert d.base_min_holdout_sharpe == 0.5
    assert d.effective_min_holdout_sharpe == 0.5
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["threshold"] == 0.5


def test_more_combos_strictly_raises_effective_bar():
    low = evaluate_gate(_wf(), GateCriteria(), n_combos=2, pit_ok=True)
    high = evaluate_gate(_wf(), GateCriteria(), n_combos=200, pit_ok=True)
    assert high.effective_min_holdout_sharpe > low.effective_min_holdout_sharpe > 0.5


def test_deflated_bar_recorded_but_advisory_at_large_n():
    # Holdout sharpe 0.8 clears base 0.5 at N=1; the N=200 haircut lifts the recorded bar above
    # 0.8 so the (advisory) holdout_sharpe check FAILS — but the soft gate still passes: the
    # deflation is telemetry now, not a veto.
    base = GateCriteria(min_holdout_sharpe=0.5)
    at_one = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=1, pit_ok=True)
    at_many = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=200, pit_ok=True)
    assert at_one.passed is True
    assert at_many.passed is True
    failed = [c for c in at_many.checks if not c["passed"]]
    assert [c["name"] for c in failed] == ["holdout_sharpe"]
    assert failed[0]["advisory"] is True
    # The deflated threshold is still computed and recorded byte-identically.
    assert failed[0]["threshold"] == at_many.effective_min_holdout_sharpe > 0.8


def test_effective_threshold_equals_base_plus_haircut():
    d = evaluate_gate(
        _wf(n_bars=100), GateCriteria(min_holdout_sharpe=0.5), n_combos=9, pit_ok=True)
    assert math.isclose(
        d.effective_min_holdout_sharpe, 0.5 + sharpe_haircut(9, 100), rel_tol=1e-12
    )


def test_provenance_carried_into_decision_and_dict():
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=4, breadth_provenance="declared", pit_ok=True)
    assert d.breadth_provenance == "declared"
    assert d.to_dict()["breadth_provenance"] == "declared"
    assert "effective_min_holdout_sharpe" in d.to_dict()


def test_low_holdout_sharpe_marks_advisory_check_failed_without_veto():
    d = evaluate_gate(_wf(holdout_sharpe=0.1), GateCriteria(min_holdout_sharpe=0.5), pit_ok=True)
    failed = [c for c in d.checks if not c["passed"]]
    assert [c["name"] for c in failed] == ["holdout_sharpe"]
    assert failed[0]["advisory"] is True
    assert d.passed is True  # soft gate: the stat records the miss but does not veto


def test_zero_holdout_return_fails_strict_gt_advisory():
    d = evaluate_gate(_wf(holdout_return=0.0), GateCriteria(), pit_ok=True)
    failed = [c for c in d.checks if not c["passed"]]
    assert [c["name"] for c in failed] == ["holdout_return"]
    assert failed[0]["advisory"] is True
    assert d.passed is True


def test_low_pct_positive_and_negative_window_fail_advisory():
    d = evaluate_gate(_wf(pct_positive=0.4, min_sharpe=-0.5), GateCriteria(), pit_ok=True)
    failed = [c for c in d.checks if not c["passed"]]
    assert {c["name"] for c in failed} == {"pct_positive_windows", "min_window_sharpe"}
    assert all(c["advisory"] is True for c in failed)
    assert d.passed is True


def test_infinite_metric_fails_gate_not_passes():
    # float('inf') trivially satisfies >=/>; it must instead fail the check.
    d = evaluate_gate(_wf(holdout_sharpe=float("inf")), GateCriteria(), pit_ok=True)
    assert d.passed is False
    failed = [c for c in d.checks if c["name"] == "holdout_sharpe"]
    assert failed and failed[0]["passed"] is False


def test_nan_metric_fails_gate_and_is_not_recorded_as_value():
    # NaN must not be recorded as a passing value in the decision payload.
    import math

    d = evaluate_gate(_wf(holdout_sharpe=float("nan")), GateCriteria(), pit_ok=True)
    assert d.passed is False
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["passed"] is False
    # The recorded value is never a raw NaN (it is nulled out instead).
    assert check["value"] is None or not math.isnan(check["value"])


def test_nan_gate_decision_is_json_serializable():
    import json

    decision = evaluate_gate(_wf(holdout_sharpe=float("nan")), GateCriteria(), pit_ok=True)
    json.dumps(decision.to_dict())


def test_to_dict_serializable():
    import json
    json.dumps(evaluate_gate(_wf(), GateCriteria(), pit_ok=True).to_dict())


def test_gate_checks_are_table_driven():
    # #40: gate checks come from a declarative spec, not hand-built literals per call site.
    # holdout_sharpe_floor + pit_required are the two non-table integrity-floor checks.
    from algua.research.gates import GATE_SPECS

    names_from_table = {spec.name for spec in GATE_SPECS}
    names_from_eval = {c["name"] for c in evaluate_gate(_wf(), GateCriteria(), pit_ok=True).checks}
    assert names_from_eval == names_from_table | {"pit_required", "holdout_sharpe_floor"}
    # Each spec points at a real GateCriteria threshold attribute.
    for spec in GATE_SPECS:
        assert hasattr(GateCriteria(), spec.threshold_attr)
    # The advisory partition is declared on the table: only the observations floor binds there.
    assert {s.name for s in GATE_SPECS if not s.advisory} == {"min_holdout_observations"}


# --- DS-integrity walls (issue 137) ---------------------------------------------------------


def test_constants_defaults():
    assert FUNNEL_WINDOW_DAYS == 90
    assert MIN_HOLDOUT_OBSERVATIONS == 63


def test_effective_funnel_breadth_is_max():
    assert effective_funnel_breadth(own_lifetime=10, windowed_total=3) == 10
    assert effective_funnel_breadth(own_lifetime=3, windowed_total=10) == 10
    assert effective_funnel_breadth(own_lifetime=0, windowed_total=0) == 0


_LAX = dict(min_holdout_sharpe=-100, min_holdout_return=-100, min_pct_positive_windows=0,
            min_window_sharpe=-100)


def test_min_holdout_observations_fails_closed_below_floor():
    d = evaluate_gate(_wf(n_bars=10), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    floor = next(c for c in d.checks if c["name"] == "min_holdout_observations")
    assert floor["passed"] is False and d.passed is False


def test_min_holdout_observations_passes_at_floor():
    d = evaluate_gate(_wf(n_bars=63), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    floor = next(c for c in d.checks if c["name"] == "min_holdout_observations")
    assert floor["passed"] is True


def test_pit_required_fails_closed():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=False)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is False and pit["override"] is None and d.passed is False


def test_pit_override_passes_and_flags():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=False, allow_non_pit=True)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is True and pit["override"] == "non_pit" and d.pit_override is True


def test_pit_ok_passes_clean():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is True and d.pit_ok is True and d.pit_override is False


def test_dsr_constants():
    assert EULER_MASCHERONI == pytest.approx(0.5772156649015329)
    assert DSR_ALPHA == 0.05


def test_dsr_n1_collapses_to_psr_against_zero():
    # N<=1 -> SR*=0; PSR for SR_pp=0.1, T=252, normal moments.
    # z = 0.1*sqrt(251)/sqrt(1+0.5*0.1**2) ~= 1.580 -> Phi ~= 0.9429
    c = dsr_confidence(0.1, 252, 0.0, 3.0, 1, 0.04)
    assert c == pytest.approx(0.9429, abs=2e-3)


def test_dsr_high_benchmark_rejects():
    # N=10 with sizeable trial dispersion lifts SR* well above SR_obs -> low confidence
    c = dsr_confidence(0.1, 252, 0.0, 3.0, 10, 0.04)
    assert c is not None and c < 0.5


def test_dsr_monotonic_in_n_and_sharpe():
    base = dsr_confidence(0.15, 252, 0.0, 3.0, 5, 0.04)
    assert dsr_confidence(0.15, 252, 0.0, 3.0, 50, 0.04) < base   # more trials -> stricter
    assert dsr_confidence(0.25, 252, 0.0, 3.0, 5, 0.04) > base    # higher SR -> higher conf


def test_dsr_fail_closed_guards():
    assert dsr_confidence(0.1, 1, 0.0, 3.0, 5, 0.04) is None       # T<=1
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 0, 0.04) is None     # N<1
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 5, -0.01) is None    # negative variance
    assert dsr_confidence(float("nan"), 252, 0.0, 3.0, 5, 0.04) is None
    # denominator <= 0: large positive skew vs SR drives 1 - skew*SR + (k-1)/4*SR^2 negative
    # (1 - 3.0*1.0 + (3.0-1)/4*1.0^2 = 1 - 3 + 0.5 = -1.5); note -skew*SR is +ve for negative skew,
    # which would INCREASE the term, so the trigger requires positive skew.
    assert dsr_confidence(1.0, 252, 3.0, 3.0, 1, 0.0) is None


def test_dsr_zero_variance_is_psr():
    # trial_sr_var=0 -> SR*=0 -> equals the N=1 PSR value
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 9, 0.0) == pytest.approx(
        dsr_confidence(0.1, 252, 0.0, 3.0, 1, 0.04), abs=1e-9)


def _wf_with(holdout, stability):
    from algua.backtest.walkforward import WalkForwardResult
    return WalkForwardResult(
        strategy="s", config_hash="c", data_source="d", snapshot_id=None, timeframe="1d",
        seed=None, period={"start": "2020-01-01", "end": "2021-01-01"}, windows=4,
        holdout_frac=0.2, window_metrics=[], holdout_metrics=holdout, stability=stability)


# a passing-on-everything-but-DSR walk-forward. Sharpe is set high enough to clear the
# search-breadth-deflated holdout-Sharpe bar at the n_combos these tests use (n=500 -> bar ~4.03),
# so the only check that can flip `passed` is the DSR check under test.
_GOOD_HOLDOUT = {
    "sharpe": 7.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0}
_GOOD_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def test_dsr_omitted_when_not_binding_does_not_change_passed():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    d = evaluate_gate(wf, GateCriteria(), n_combos=10, pit_ok=True, dsr_binding=False)
    assert d.passed is True
    assert all(c["name"] != "dsr_evidence" for c in d.checks)
    assert d.dsr_binding is False and d.dsr_confidence is None


def test_dsr_armed_failure_recorded_but_never_vetoes():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    # huge trial dispersion + many trials -> SR* far above the holdout Sharpe -> DSR check fails,
    # marked advisory — the soft gate still passes on the floor.
    d = evaluate_gate(wf, GateCriteria(), n_combos=500, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=400.0)
    dsr = next(c for c in d.checks if c["name"] == "dsr_evidence")
    assert dsr["passed"] is False and dsr["advisory"] is True
    assert d.passed is True


def test_dsr_armed_missing_variance_recorded_failed_advisory():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    d = evaluate_gate(wf, GateCriteria(), n_combos=10, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=None)
    dsr = next(c for c in d.checks if c["name"] == "dsr_evidence")
    assert dsr["passed"] is False and dsr["advisory"] is True
    assert d.dsr_confidence is None
    assert d.passed is True  # a missing-stat advisory row records the gap, never vetoes


def test_advisory_stats_never_change_passed():
    # Soft-gate invariant (replaces the old tighten-only invariant): over a grid of decisions,
    # `passed` is a pure function of the integrity floor — arming/failing the DSR stack can
    # NEVER flip it in either direction.
    for sharpe, nbars, binding, var in itertools.product(
            [0.2, 0.6, 1.2], [80, 252], [False, True], [None, 0.0, 4.0, 400.0]):
        holdout = {"sharpe": sharpe, "total_return": 0.1, "n_bars": nbars,
                   "skewness": 0.0, "kurtosis": 3.0}
        stab = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}
        wf = _wf_with(holdout, stab)
        old = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True, dsr_binding=False)
        new = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True,
                            dsr_binding=binding, dsr_trial_var_ann=var)
        assert new.passed == old.passed == all(
            c["passed"] for c in new.checks if not c.get("advisory"))


# ---------------------------------------------------------------------------
# Task 1 — LORD++ alpha-wealth level (#220, Phase 2)
# ---------------------------------------------------------------------------

def test_fdr_constants():
    assert FDR_ALPHA == 0.05
    # #529: W0 UNCHANGED at FDR_ALPHA/2 (round-2's doubling reverted); cohort recalibrated 64→8;
    # the two throttle constants pin the near-term exposure budget H_near / its window.
    assert FDR_W0 == pytest.approx(FDR_ALPHA / 2)
    assert FDR_COHORT_SIZE == 8
    assert FDR_NEAR_TERM_BINDING_BUDGET == 16
    assert FDR_THROTTLE_WINDOW_DAYS == 365


def test_gamma_weights_normalized_over_cohort_size():
    # #529: γ is normalized over the RESTART horizon (FDR_COHORT_SIZE terms), so it sums to exactly
    # 1.0 across the cohort — a cohort spends its full W0 dry-spell budget.
    assert len(_LORD_GAMMA) == FDR_COHORT_SIZE
    assert abs(sum(_LORD_GAMMA) - 1.0) < 1e-9


def test_gamma_weights_are_positive():
    assert all(g > 0 for g in _LORD_GAMMA)


def test_gamma_weights_are_decreasing_across_cohort():
    # The raw γ formula is monotone-decreasing across the whole N=8 cohort horizon.
    for j in range(1, len(_LORD_GAMMA)):
        assert _LORD_GAMMA[j] <= _LORD_GAMMA[j - 1], f"γ not decreasing at j={j+1}"


def test_lord_no_discoveries_alpha_decreasing():
    # With no prior discoveries, α_t = γ_t · W_0 → decreases across the cohort horizon (1..N).
    levels = [
        lord_plus_plus_level(t, [], alpha=FDR_ALPHA, w0=FDR_W0)
        for t in range(1, FDR_COHORT_SIZE + 1)
    ]
    # All positive within the cohort (weights are positive over 1..N).
    assert all(lv > 0 for lv in levels)
    # Monotone non-increasing across the cohort.
    for i in range(1, len(levels)):
        assert levels[i] <= levels[i - 1] + 1e-15, f"level not decreasing at t={i+1}"


def test_lord_cohort_spends_full_budget():
    # #529: Σ_{t=1..N} α_t^dry == W0 (γ normalized over the restart horizon) AND the pinned
    # recalibrated first two dry-spell levels (§3.1 table row N=8).
    dry = [lord_plus_plus_level(t, [], alpha=FDR_ALPHA, w0=FDR_W0)
           for t in range(1, FDR_COHORT_SIZE + 1)]
    assert sum(dry) == pytest.approx(FDR_W0)
    assert dry[0] == pytest.approx(0.00764, abs=1e-4)
    assert dry[1] == pytest.approx(0.00382, abs=1e-4)
    # Pin the FULL dry-spell vector against the §3.1 table row so the S(16) arithmetic below is
    # proven to use the ACTUAL level function, not hand-copied table constants.
    expected = [0.00764, 0.00382, 0.00325, 0.00271, 0.00229, 0.00198, 0.00175, 0.00156]
    assert dry == pytest.approx(expected, abs=1e-4)


def test_lord_all_null_first_discovery_probability():
    # #529 §3.5 adversarial: per FRESH cohort, P(≥1 false discovery) = p_cohort EXACTLY, computed
    # from the ACTUAL dry-spell levels (pre-first-discovery state — levels are dry-spell UP TO the
    # first discovery, so the ≥1 event lives entirely in this product). Assert it is ≈ FDR_ALPHA/2
    # (≈2.5%), NOT the 72% an alpha-FLOOR would have produced.
    # ASSUMPTION (standard LORD++): null p-values are super-uniform and the LORD++
    # predictability/independence conditions hold — the SAME assumption the pre-existing per-cohort
    # FDR guarantee already relies on, not a new one.
    dry = [lord_plus_plus_level(t, [], alpha=FDR_ALPHA, w0=FDR_W0)
           for t in range(1, FDR_COHORT_SIZE + 1)]
    p_cohort = 1.0 - math.prod(1.0 - a for a in dry)
    assert p_cohort <= FDR_ALPHA / 2 + 1e-6
    assert p_cohort == pytest.approx(0.0247, abs=1e-3)


def test_lord_retry_surface_within_near_term_budget():
    # #529 §3.1/§3.5: the cohort-ALIGNED idealized near-term surface S(16) stays within the 5%
    # budget at N=8, AND — the load-bearing bound — the RIGOROUS worst-case for ANY rolling window
    # (≤ ceil(H_near/N)+1 == 3 always-fresh cohorts touched) stays ≤ 8%. Also pin H_near to the cap.
    dry = [lord_plus_plus_level(t, [], alpha=FDR_ALPHA, w0=FDR_W0)
           for t in range(1, FDR_COHORT_SIZE + 1)]
    p_cohort = 1.0 - math.prod(1.0 - a for a in dry)
    h_near = FDR_NEAR_TERM_BINDING_BUDGET
    aligned = 1.0 - (1.0 - p_cohort) ** (h_near / FDR_COHORT_SIZE)
    assert aligned <= 0.05
    max_touched_cohorts = math.ceil(h_near / FDR_COHORT_SIZE) + 1
    assert max_touched_cohorts == 3
    worst_case = 1.0 - (1.0 - p_cohort) ** max_touched_cohorts
    assert worst_case <= 0.08
    # The cap MUST equal H_near (the derivation horizon) — they can't desync.
    assert h_near == FDR_NEAR_TERM_BINDING_BUDGET


def test_lord_first_discovery_bumps_alpha():
    # A discovery at τ_1=1 adds (α-W_0)·γ_{t-1} to every subsequent α_t.
    # At t=2: α_2_with = γ_2·W_0 + (α-W_0)·γ_1 > α_2_without = γ_2·W_0
    alpha_2_no_disc = lord_plus_plus_level(2, [], alpha=FDR_ALPHA, w0=FDR_W0)
    alpha_2_disc1 = lord_plus_plus_level(2, [1], alpha=FDR_ALPHA, w0=FDR_W0)
    assert alpha_2_disc1 > alpha_2_no_disc

    # The bump equals (α-W_0)·γ_{t-τ_1}
    expected_bump = (FDR_ALPHA - FDR_W0) * _LORD_GAMMA[0]  # γ_{2-1}=γ_1=_LORD_GAMMA[0]
    assert alpha_2_disc1 - alpha_2_no_disc == pytest.approx(expected_bump, rel=1e-9)


def test_lord_multiple_discoveries_replenish():
    # More discoveries → more replenishment → higher α_t vs no-discovery baseline. t=6 is within
    # the N=8 cohort horizon (γ is 0 past FDR_COHORT_SIZE).
    t = 6
    alpha_no_disc = lord_plus_plus_level(t, [], alpha=FDR_ALPHA, w0=FDR_W0)
    alpha_1disc = lord_plus_plus_level(t, [1], alpha=FDR_ALPHA, w0=FDR_W0)
    alpha_3disc = lord_plus_plus_level(t, [1, 3, 5], alpha=FDR_ALPHA, w0=FDR_W0)
    assert alpha_no_disc < alpha_1disc < alpha_3disc


def test_lord_manual_recursion_check():
    # Verify the formula directly for a small stream.
    # t=3, τ_1=1, τ_2=2:
    # α_3 = γ_3·W_0 + (α-W_0)·γ_{3-1} + α·γ_{3-2}
    #      = γ_3·W_0 + (α-W_0)·γ_2 + α·γ_1
    g1, g2, g3 = _LORD_GAMMA[0], _LORD_GAMMA[1], _LORD_GAMMA[2]
    expected = g3 * FDR_W0 + (FDR_ALPHA - FDR_W0) * g2 + FDR_ALPHA * g1
    computed = lord_plus_plus_level(3, [1, 2], alpha=FDR_ALPHA, w0=FDR_W0)
    assert computed == pytest.approx(expected, rel=1e-12)


def test_lord_fail_closed_guards():
    # t < 1 → 0.0 (conservative; can't pass a level-0 test)
    assert lord_plus_plus_level(0, [], alpha=FDR_ALPHA, w0=FDR_W0) == 0.0
    assert lord_plus_plus_level(-1, [], alpha=FDR_ALPHA, w0=FDR_W0) == 0.0
    # non-finite alpha/w0
    assert lord_plus_plus_level(1, [], alpha=float("nan"), w0=FDR_W0) == 0.0
    assert lord_plus_plus_level(1, [], alpha=FDR_ALPHA, w0=float("inf")) == 0.0
    # discovery index >= t (not a past discovery)
    assert lord_plus_plus_level(2, [2], alpha=FDR_ALPHA, w0=FDR_W0) == 0.0
    # discovery index < 1
    assert lord_plus_plus_level(2, [0], alpha=FDR_ALPHA, w0=FDR_W0) == 0.0


def test_lord_injected_params():
    # Calling with different alpha/w0 produces a proportionally scaled level.
    alpha_half = lord_plus_plus_level(1, [], alpha=0.025, w0=0.0125)
    alpha_full = lord_plus_plus_level(1, [], alpha=FDR_ALPHA, w0=FDR_W0)
    assert alpha_half == pytest.approx(alpha_full / 2, rel=1e-9)


def test_lord_t1_no_discoveries_equals_gamma1_times_w0():
    # α_1 = γ_1 · W_0 (base case, no prior discoveries)
    expected = _LORD_GAMMA[0] * FDR_W0
    result = lord_plus_plus_level(1, [], alpha=FDR_ALPHA, w0=FDR_W0)
    assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Task 5 (#220 Phase 2): GateDecision FDR fields
# ---------------------------------------------------------------------------


def test_gate_decision_has_fdr_fields_with_defaults():
    d = GateDecision(passed=True, checks=[])
    assert d.fdr_binding is False
    assert d.fdr_p_value is None
    assert d.fdr_alpha_level is None
    assert d.fdr_test_index is None
    assert d.fdr_rejected is None
    assert d.fdr_skip_reason is None


def test_gate_decision_to_dict_includes_fdr_keys():
    d = GateDecision(passed=True, checks=[])
    out = d.to_dict()
    for key in ("fdr_binding", "fdr_p_value", "fdr_alpha_level",
                "fdr_test_index", "fdr_rejected", "fdr_skip_reason"):
        assert key in out, f"to_dict() missing {key!r}"


def test_gate_decision_to_dict_fdr_fields_populated():
    d = GateDecision(
        passed=False, checks=[],
        fdr_binding=True, fdr_p_value=0.02, fdr_alpha_level=0.001645,
        fdr_test_index=3, fdr_rejected=False, fdr_skip_reason=None,
    )
    out = d.to_dict()
    assert out["fdr_binding"] is True
    assert out["fdr_p_value"] == pytest.approx(0.02)
    assert out["fdr_alpha_level"] == pytest.approx(0.001645)
    assert out["fdr_test_index"] == 3
    assert out["fdr_rejected"] is False
    assert out["fdr_skip_reason"] is None


def test_gate_decision_to_dict_non_binding_fdr():
    d = GateDecision(
        passed=True, checks=[],
        fdr_binding=False, fdr_skip_reason="no_measured_dispersion",
    )
    out = d.to_dict()
    assert out["fdr_binding"] is False
    assert out["fdr_p_value"] is None
    assert out["fdr_skip_reason"] == "no_measured_dispersion"
