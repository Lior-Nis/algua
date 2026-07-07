import json
import math

import pytest

from algua.research.forward_gates import (
    DEGRADATION_FACTOR,
    FORWARD_SHARPE_CONFIDENCE,
    MAX_FORWARD_DRAWDOWN,
    MAX_STALENESS_SESSIONS,
    MIN_FORWARD_OBSERVATIONS,
    MIN_FORWARD_VOL,
    MIN_SESSION_COVERAGE,
    SHARPE_FLOOR,
    ForwardEvidence,
    ForwardGateCriteria,
    ForwardGateDecision,
    evaluate_forward_gate,
)


def passing_evidence(**over):
    # realized_sharpe=4.0 clears the 95% PSR wall at n=63 (needs an observed annual Sharpe ~3.32);
    # skew=0.0, kurtosis=3.0 are the Gaussian (neutral) moments for the Sharpe-SE adjustment.
    base = dict(n_return_observations=63, session_coverage=0.95, realized_sharpe=4.0,
                realized_skew=0.0, realized_kurtosis=3.0,
                realized_vol=0.10, realized_max_drawdown=0.10, holdout_sharpe=1.0,
                n_reconcile_failures=0, n_defective_ticks=0, kill_switch_tripped=False,
                global_halt_engaged=False, n_kill_trips_in_window=0, single_account_ok=True,
                activities_ok=True, n_external_cash_flows=0, n_unattributable_fills=0,
                staleness_sessions=2)
    return ForwardEvidence(**(base | over))


def _check(decision: ForwardGateDecision, name: str) -> dict:
    return next(c for c in decision.checks if c["name"] == name)


ALL_CHECK_NAMES = {
    "min_forward_observations", "session_coverage", "realized_sharpe", "realized_sharpe_lcb",
    "min_forward_vol", "max_forward_drawdown", "reconcile_ok", "no_defective_ticks",
    "kill_switch_clear", "global_halt_clear", "no_kill_trips_in_window", "single_account",
    "activities_ok", "no_external_cash_flows", "no_unattributable_fills", "max_staleness_sessions",
}


def test_protected_constants():
    # Protected defaults from the #124 spec — NOT agent-tunable knobs.
    assert MIN_FORWARD_OBSERVATIONS == 63
    assert MIN_SESSION_COVERAGE == 0.9
    assert DEGRADATION_FACTOR == 0.5
    assert SHARPE_FLOOR == 0.3
    assert MIN_FORWARD_VOL == 0.02
    assert MAX_FORWARD_DRAWDOWN == 0.25
    assert MAX_STALENESS_SESSIONS == 5
    assert FORWARD_SHARPE_CONFIDENCE == 0.95


def test_all_pass_baseline():
    d = evaluate_forward_gate(passing_evidence(), ForwardGateCriteria())
    assert isinstance(d, ForwardGateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == ALL_CHECK_NAMES
    assert all(c["passed"] for c in d.checks)


# --- one field flipped across its boundary fails exactly that check --------------------------

# NOTE: the point-performance bar ('realized_sharpe') is NOT probed here — a low realized Sharpe
# co-fails the new 'realized_sharpe_lcb' significance wall, breaking this loop's "exactly one check
# fails" invariant. The point bar is exercised in isolation by the test_point_bar_* tests below.
FLIP_CASES = [
    ({"n_return_observations": 62}, "min_forward_observations"),
    ({"session_coverage": 0.89}, "session_coverage"),
    ({"realized_vol": 0.01}, "min_forward_vol"),
    ({"realized_max_drawdown": 0.26}, "max_forward_drawdown"),
    # integrity: each field flipped fails
    ({"n_reconcile_failures": 1}, "reconcile_ok"),
    ({"n_defective_ticks": 1}, "no_defective_ticks"),
    ({"kill_switch_tripped": True}, "kill_switch_clear"),
    ({"global_halt_engaged": True}, "global_halt_clear"),
    ({"n_kill_trips_in_window": 1}, "no_kill_trips_in_window"),
    # account hygiene
    ({"single_account_ok": False}, "single_account"),
    ({"activities_ok": False}, "activities_ok"),
    ({"n_external_cash_flows": 1}, "no_external_cash_flows"),
    ({"n_unattributable_fills": 1}, "no_unattributable_fills"),
    # staleness: over the 5-session cap, and None (no admissible ticks) fails closed
    ({"staleness_sessions": 6}, "max_staleness_sessions"),
    ({"staleness_sessions": None}, "max_staleness_sessions"),
]


@pytest.mark.parametrize("over,failing", FLIP_CASES, ids=[
    "obs_62", "coverage_0.89",
    "vol_0.01", "drawdown_0.26", "reconcile_failure", "defective_tick", "kill_switch_tripped",
    "global_halt_engaged", "kill_trip_in_window", "single_account_failed", "activities_not_ok",
    "external_cash_flow", "unattributable_fill", "staleness_6", "staleness_none",
])
def test_single_field_flip_fails_exactly_that_check(over, failing):
    d = evaluate_forward_gate(passing_evidence(**over), ForwardGateCriteria())
    assert d.passed is False
    assert _check(d, failing)["passed"] is False
    for c in d.checks:
        if c["name"] != failing:
            assert c["passed"] is True, f"{c['name']} unexpectedly failed"


# --- boundary values pass (>= / <= semantics) -------------------------------------------------

# The realized_sharpe point-bar boundaries are exercised in isolation by test_point_bar_* below
# (low Sharpes at the bar legitimately fail the significance wall, so they can't be overall-pass
# boundary cases here).
BOUNDARY_PASS_CASES = [
    {"session_coverage": 0.9},
    {"realized_vol": 0.02},
    {"realized_max_drawdown": 0.25},
    {"staleness_sessions": 5},
]


@pytest.mark.parametrize("over", BOUNDARY_PASS_CASES, ids=[
    "coverage_at_0.9", "vol_at_0.02", "drawdown_at_0.25", "staleness_at_5",
])
def test_boundary_values_pass(over):
    d = evaluate_forward_gate(passing_evidence(**over), ForwardGateCriteria())
    assert d.passed is True


# --- point performance bar ('realized_sharpe') probed in isolation via _check --------------------
# These deliberately sit at low Sharpes where the significance wall (realized_sharpe_lcb) fails, so
# we assert ONLY the point-bar check, never overall d.passed.


def test_point_bar_below_degradation():
    # bar = max(0.5 * holdout, 0.3); holdout=1.0 -> bar 0.5; realized 0.49 misses it.
    d = evaluate_forward_gate(
        passing_evidence(holdout_sharpe=1.0, realized_sharpe=0.49), ForwardGateCriteria())
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["threshold"] == 0.5


def test_point_bar_below_floor():
    # holdout=0.4 -> 0.5*0.4=0.2 < floor -> bar is the 0.3 floor; realized 0.29 misses it.
    d = evaluate_forward_gate(
        passing_evidence(holdout_sharpe=0.4, realized_sharpe=0.29), ForwardGateCriteria())
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["threshold"] == 0.3


def test_point_bar_at_degradation():
    # exactly at the degradation bar: holdout=1.0 -> bar 0.5, realized 0.5 passes (>=).
    d = evaluate_forward_gate(
        passing_evidence(holdout_sharpe=1.0, realized_sharpe=0.5), ForwardGateCriteria())
    assert _check(d, "realized_sharpe")["passed"] is True


def test_point_bar_just_above_floor():
    # floor regime: holdout=0.4 -> bar 0.3, realized 0.31 passes.
    d = evaluate_forward_gate(
        passing_evidence(holdout_sharpe=0.4, realized_sharpe=0.31), ForwardGateCriteria())
    assert _check(d, "realized_sharpe")["passed"] is True


# --- statistical-significance wall (realized_sharpe_lcb / PSR) ----------------------------------


def test_significant_point_but_insignificant_fails():
    # The #432 scenario: a 63-obs draw clears the point bar (0.5) at realized Sharpe 1.0 but is
    # NOT statistically distinguishable from zero -> the significance wall fails the gate.
    d = evaluate_forward_gate(
        passing_evidence(holdout_sharpe=1.0, realized_sharpe=1.0, n_return_observations=63),
        ForwardGateCriteria())
    assert _check(d, "realized_sharpe")["passed"] is True
    lcb = _check(d, "realized_sharpe_lcb")
    assert lcb["passed"] is False
    assert lcb["op"] == ">="
    assert lcb["threshold"] == FORWARD_SHARPE_CONFIDENCE
    assert d.passed is False
    json.dumps(d.to_dict(), allow_nan=False)


def test_lcb_pass_baseline():
    # The realized_sharpe=4.0 baseline clears the 95% PSR wall at n=63.
    d = evaluate_forward_gate(passing_evidence(), ForwardGateCriteria())
    lcb = _check(d, "realized_sharpe_lcb")
    assert lcb["passed"] is True
    assert lcb["value"] >= FORWARD_SHARPE_CONFIDENCE


def test_lcb_fail_closed_on_underpowered():
    # n<=1 has no standard error (sqrt(T-1)==0) -> the PSR is undefined and fails closed. (The
    # obs-floor check also fails here, which is fine; we assert only the lcb check's fields.)
    d = evaluate_forward_gate(
        passing_evidence(n_return_observations=1), ForwardGateCriteria())
    lcb = _check(d, "realized_sharpe_lcb")
    assert lcb["passed"] is False
    assert lcb["value"] is None
    assert lcb["detail"]
    json.dumps(d.to_dict(), allow_nan=False)


def test_lcb_more_observations_rescue_significance():
    # A marginal realized Sharpe (1.2) that fails at n=63 clears the wall with a much longer
    # window (700 obs) — the SE shrinks with T, which is the sanctioned remedy, not a weaker bar.
    short = evaluate_forward_gate(
        passing_evidence(realized_sharpe=1.2, n_return_observations=63), ForwardGateCriteria())
    assert _check(short, "realized_sharpe_lcb")["passed"] is False
    long = evaluate_forward_gate(
        passing_evidence(realized_sharpe=1.2, n_return_observations=700), ForwardGateCriteria())
    assert _check(long, "realized_sharpe_lcb")["passed"] is True


def test_lcb_tightened_confidence_is_agent_allowed():
    # Agents may TIGHTEN (raise) the confidence; a baseline that clears 0.95 can fail at 0.999.
    d = evaluate_forward_gate(
        passing_evidence(), ForwardGateCriteria(forward_sharpe_confidence=0.999))
    assert _check(d, "realized_sharpe_lcb")["passed"] is False


# --- fail-closed paths -------------------------------------------------------------------------


def test_no_holdout_sharpe_fails_closed():
    # No qualified backtest gate row (holdout_sharpe=None) -> the performance check can never
    # pass; the strategy needs a fresh `research promote` first.
    d = evaluate_forward_gate(passing_evidence(holdout_sharpe=None), ForwardGateCriteria())
    assert d.passed is False
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["threshold"] is None
    assert c["detail"] == "no qualified backtest gate row for current identity"


def test_staleness_none_carries_detail():
    d = evaluate_forward_gate(passing_evidence(staleness_sessions=None), ForwardGateCriteria())
    c = _check(d, "max_staleness_sessions")
    assert c["passed"] is False
    assert c["value"] is None
    assert c["detail"] == "no admissible evidence ticks"


def test_nan_metric_fails_closed_with_null_value():
    # A non-finite metric is never a pass and never lands as NaN in the payload (JSON-clean).
    d = evaluate_forward_gate(passing_evidence(realized_sharpe=float("nan")),
                              ForwardGateCriteria())
    assert d.passed is False
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["value"] is None


def test_to_dict_json_round_trips_with_nan_input():
    d = evaluate_forward_gate(passing_evidence(realized_sharpe=float("nan")),
                              ForwardGateCriteria())
    payload = d.to_dict()
    assert payload["passed"] is False
    # allow_nan=False proves no NaN/inf leaked into the payload.
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == payload


def test_nan_holdout_sharpe_fails_closed():
    # Defensive: a non-finite holdout bar must never pass (max() with NaN is treacherous).
    d = evaluate_forward_gate(passing_evidence(holdout_sharpe=float("nan")),
                              ForwardGateCriteria())
    assert d.passed is False
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["threshold"] is None


def test_inf_realized_vol_fails_closed():
    # inf trivially clears a >= floor; the finite-guard must fail it, never pass it.
    d = evaluate_forward_gate(passing_evidence(realized_vol=float("inf")),
                              ForwardGateCriteria())
    assert d.passed is False
    c = _check(d, "min_forward_vol")
    assert c["passed"] is False
    assert c["value"] is None
    json.dumps(d.to_dict(), allow_nan=False)


@pytest.mark.parametrize("crit_over", [
    # max(0.25, nan) == 0.25: a NaN floor would silently RELAX the bar below the protected 0.3.
    {"sharpe_floor": float("nan")},
    # -inf factor collapses the bar to the floor regardless of holdout — also a relaxation.
    {"degradation_factor": float("-inf")},
], ids=["nan_sharpe_floor", "neg_inf_degradation_factor"])
def test_non_finite_performance_criteria_fail_closed(crit_over):
    # NaN/inf criteria components must fail closed, not relax the performance bar.
    ev = passing_evidence(holdout_sharpe=0.5, realized_sharpe=0.26)
    d = evaluate_forward_gate(ev, ForwardGateCriteria(**crit_over))
    assert d.passed is False
    c = _check(d, "realized_sharpe")
    assert c["passed"] is False
    assert c["threshold"] is None
    assert c["detail"] == "non-finite performance criteria (degradation_factor/sharpe_floor)"
    json.dumps(d.to_dict(), allow_nan=False)


# --- agent-tightened criteria respected --------------------------------------------------------


def test_tightened_observation_floor_fails_63_obs_evidence():
    # Agents may TIGHTEN thresholds (never relax — relaxation is human-only at the CLI).
    d = evaluate_forward_gate(passing_evidence(),
                              ForwardGateCriteria(min_forward_observations=80))
    assert d.passed is False
    c = _check(d, "min_forward_observations")
    assert c["passed"] is False
    assert c["threshold"] == 80


def test_single_account_check_key_and_failure_message():
    # An evidence with single_account_ok=False produces a failed check keyed "single_account".
    ev = passing_evidence(single_account_ok=False)
    decision = evaluate_forward_gate(ev, ForwardGateCriteria())
    failed = {c["name"] for c in decision.checks if not c["passed"]}
    assert "single_account" in failed
    assert "single_tenant" not in {c["name"] for c in decision.checks}


def test_evaluation_is_pure():
    ev = passing_evidence()
    crit = ForwardGateCriteria()
    d1 = evaluate_forward_gate(ev, crit)
    d2 = evaluate_forward_gate(ev, crit)
    assert d1.to_dict() == d2.to_dict()
    assert math.isfinite(_check(d1, "realized_sharpe")["threshold"])
