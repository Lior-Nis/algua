import json
import math

import pytest

from algua.research.forward_gates import (
    DEGRADATION_FACTOR,
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
    base = dict(n_return_observations=63, session_coverage=0.95, realized_sharpe=0.8,
                realized_vol=0.10, realized_max_drawdown=0.10, holdout_sharpe=1.0,
                n_reconcile_failures=0, n_defective_ticks=0, kill_switch_tripped=False,
                global_halt_engaged=False, n_kill_trips_in_window=0, single_tenant_ok=True,
                activities_ok=True, n_external_cash_flows=0, n_unattributable_fills=0,
                staleness_sessions=2)
    return ForwardEvidence(**(base | over))


def _check(decision: ForwardGateDecision, name: str) -> dict:
    return next(c for c in decision.checks if c["name"] == name)


ALL_CHECK_NAMES = {
    "min_forward_observations", "session_coverage", "realized_sharpe", "min_forward_vol",
    "max_forward_drawdown", "reconcile_ok", "no_defective_ticks", "kill_switch_clear",
    "global_halt_clear", "no_kill_trips_in_window", "single_tenant", "activities_ok",
    "no_external_cash_flows", "no_unattributable_fills", "max_staleness_sessions",
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


def test_all_pass_baseline():
    d = evaluate_forward_gate(passing_evidence(), ForwardGateCriteria())
    assert isinstance(d, ForwardGateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == ALL_CHECK_NAMES
    assert all(c["passed"] for c in d.checks)


# --- one field flipped across its boundary fails exactly that check --------------------------

FLIP_CASES = [
    ({"n_return_observations": 62}, "min_forward_observations"),
    ({"session_coverage": 0.89}, "session_coverage"),
    # bar = max(0.5 * holdout, 0.3); holdout=1.0 -> bar 0.5
    ({"realized_sharpe": 0.49}, "realized_sharpe"),
    # holdout=0.4 -> 0.5*0.4=0.2 < floor -> bar is the 0.3 floor
    ({"holdout_sharpe": 0.4, "realized_sharpe": 0.29}, "realized_sharpe"),
    ({"realized_vol": 0.01}, "min_forward_vol"),
    ({"realized_max_drawdown": 0.26}, "max_forward_drawdown"),
    # integrity: each field flipped fails
    ({"n_reconcile_failures": 1}, "reconcile_ok"),
    ({"n_defective_ticks": 1}, "no_defective_ticks"),
    ({"kill_switch_tripped": True}, "kill_switch_clear"),
    ({"global_halt_engaged": True}, "global_halt_clear"),
    ({"n_kill_trips_in_window": 1}, "no_kill_trips_in_window"),
    # account hygiene
    ({"single_tenant_ok": False}, "single_tenant"),
    ({"activities_ok": False}, "activities_ok"),
    ({"n_external_cash_flows": 1}, "no_external_cash_flows"),
    ({"n_unattributable_fills": 1}, "no_unattributable_fills"),
    # staleness: over the 5-session cap, and None (no admissible ticks) fails closed
    ({"staleness_sessions": 6}, "max_staleness_sessions"),
    ({"staleness_sessions": None}, "max_staleness_sessions"),
]


@pytest.mark.parametrize("over,failing", FLIP_CASES, ids=[
    "obs_62", "coverage_0.89", "sharpe_below_degradation_bar", "sharpe_below_floor_bar",
    "vol_0.01", "drawdown_0.26", "reconcile_failure", "defective_tick", "kill_switch_tripped",
    "global_halt_engaged", "kill_trip_in_window", "multi_tenant", "activities_not_ok",
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

BOUNDARY_PASS_CASES = [
    # exactly at the degradation bar: holdout=1.0 -> bar 0.5, realized 0.5 passes (>=)
    {"realized_sharpe": 0.5},
    # floor regime: holdout=0.4 -> bar 0.3, realized 0.31 passes
    {"holdout_sharpe": 0.4, "realized_sharpe": 0.31},
    {"session_coverage": 0.9},
    {"realized_max_drawdown": 0.25},
    {"staleness_sessions": 5},
]


@pytest.mark.parametrize("over", BOUNDARY_PASS_CASES, ids=[
    "sharpe_at_degradation_bar", "sharpe_just_above_floor", "coverage_at_0.9",
    "drawdown_at_0.25", "staleness_at_5",
])
def test_boundary_values_pass(over):
    d = evaluate_forward_gate(passing_evidence(**over), ForwardGateCriteria())
    assert d.passed is True


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


# --- agent-tightened criteria respected --------------------------------------------------------


def test_tightened_observation_floor_fails_63_obs_evidence():
    # Agents may TIGHTEN thresholds (never relax — relaxation is human-only at the CLI).
    d = evaluate_forward_gate(passing_evidence(),
                              ForwardGateCriteria(min_forward_observations=80))
    assert d.passed is False
    c = _check(d, "min_forward_observations")
    assert c["passed"] is False
    assert c["threshold"] == 80


def test_evaluation_is_pure():
    ev = passing_evidence()
    crit = ForwardGateCriteria()
    d1 = evaluate_forward_gate(ev, crit)
    d2 = evaluate_forward_gate(ev, crit)
    assert d1.to_dict() == d2.to_dict()
    assert math.isfinite(_check(d1, "realized_sharpe")["threshold"])
