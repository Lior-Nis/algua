import itertools
import math

import pytest

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    DSR_ALPHA,
    EULER_MASCHERONI,
    FUNNEL_WINDOW_DAYS,
    MIN_HOLDOUT_OBSERVATIONS,
    GateCriteria,
    GateDecision,
    dsr_confidence,
    effective_funnel_breadth,
    evaluate_gate,
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
        "min_holdout_observations", "pit_required"}
    assert all(c["passed"] for c in d.checks)
    assert d.n_combos == 1


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


def test_passes_at_n1_fails_at_large_n_with_identical_metrics():
    # Holdout sharpe 0.8 clears base 0.5 at N=1, but the N=200 haircut lifts the bar above 0.8.
    base = GateCriteria(min_holdout_sharpe=0.5)
    at_one = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=1, pit_ok=True)
    at_many = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=200, pit_ok=True)
    assert at_one.passed is True
    assert at_many.passed is False
    failed = [c["name"] for c in at_many.checks if not c["passed"]]
    assert failed == ["holdout_sharpe"]


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


def test_low_holdout_sharpe_fails_that_check():
    d = evaluate_gate(_wf(holdout_sharpe=0.1), GateCriteria(min_holdout_sharpe=0.5), pit_ok=True)
    assert d.passed is False
    assert [c["name"] for c in d.checks if not c["passed"]] == ["holdout_sharpe"]


def test_zero_holdout_return_fails_strict_gt():
    d = evaluate_gate(_wf(holdout_return=0.0), GateCriteria(), pit_ok=True)
    assert d.passed is False
    assert [c["name"] for c in d.checks if not c["passed"]] == ["holdout_return"]


def test_low_pct_positive_and_negative_window_fail():
    d = evaluate_gate(_wf(pct_positive=0.4, min_sharpe=-0.5), GateCriteria(), pit_ok=True)
    assert d.passed is False
    failed = {c["name"] for c in d.checks if not c["passed"]}
    assert failed == {"pct_positive_windows", "min_window_sharpe"}


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
    from algua.research.gates import GATE_SPECS

    names_from_table = {spec.name for spec in GATE_SPECS}
    names_from_eval = {c["name"] for c in evaluate_gate(_wf(), GateCriteria(), pit_ok=True).checks}
    assert names_from_eval == names_from_table | {"pit_required"}
    # Each spec points at a real GateCriteria threshold attribute.
    for spec in GATE_SPECS:
        assert hasattr(GateCriteria(), spec.threshold_attr)


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


def test_dsr_binding_can_only_reject():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    # huge trial dispersion + many trials -> SR* far above the holdout Sharpe -> DSR fails
    d = evaluate_gate(wf, GateCriteria(), n_combos=500, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=400.0)
    assert d.passed is False
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)


def test_dsr_binding_missing_variance_fails_closed():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    d = evaluate_gate(wf, GateCriteria(), n_combos=10, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=None)
    assert d.passed is False
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)
    assert d.dsr_confidence is None


def test_tighten_only_invariant():
    # new_pass == old_pass AND (not dsr_binding or dsr_pass), over a grid of decisions.
    for sharpe, nbars, binding, var in itertools.product(
            [0.2, 0.6, 1.2], [80, 252], [False, True], [None, 0.0, 4.0, 400.0]):
        holdout = {"sharpe": sharpe, "total_return": 0.1, "n_bars": nbars,
                   "skewness": 0.0, "kurtosis": 3.0}
        stab = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}
        wf = _wf_with(holdout, stab)
        old = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True, dsr_binding=False)
        new = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True,
                            dsr_binding=binding, dsr_trial_var_ann=var)
        dsr_check = next((c for c in new.checks if c["name"] == "dsr_evidence"), None)
        dsr_pass = (dsr_check is None) or dsr_check["passed"]
        assert new.passed == (old.passed and ((not binding) or dsr_pass))
