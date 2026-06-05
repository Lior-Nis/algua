import math

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    GateCriteria,
    GateDecision,
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
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=1)
    assert isinstance(d, GateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == {
        "holdout_sharpe", "holdout_return", "pct_positive_windows", "min_window_sharpe"}
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
    d = evaluate_gate(_wf(holdout_sharpe=99.0, n_bars=0), GateCriteria(), n_combos=9)
    assert d.passed is False
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["passed"] is False


def test_n1_effective_equals_base():
    d = evaluate_gate(_wf(), GateCriteria(min_holdout_sharpe=0.5), n_combos=1)
    assert d.base_min_holdout_sharpe == 0.5
    assert d.effective_min_holdout_sharpe == 0.5
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["threshold"] == 0.5


def test_more_combos_strictly_raises_effective_bar():
    low = evaluate_gate(_wf(), GateCriteria(), n_combos=2)
    high = evaluate_gate(_wf(), GateCriteria(), n_combos=200)
    assert high.effective_min_holdout_sharpe > low.effective_min_holdout_sharpe > 0.5


def test_passes_at_n1_fails_at_large_n_with_identical_metrics():
    # Holdout sharpe 0.8 clears base 0.5 at N=1, but the N=200 haircut lifts the bar above 0.8.
    base = GateCriteria(min_holdout_sharpe=0.5)
    at_one = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=1)
    at_many = evaluate_gate(_wf(holdout_sharpe=0.8), base, n_combos=200)
    assert at_one.passed is True
    assert at_many.passed is False
    failed = [c["name"] for c in at_many.checks if not c["passed"]]
    assert failed == ["holdout_sharpe"]


def test_effective_threshold_equals_base_plus_haircut():
    d = evaluate_gate(_wf(n_bars=100), GateCriteria(min_holdout_sharpe=0.5), n_combos=9)
    assert math.isclose(
        d.effective_min_holdout_sharpe, 0.5 + sharpe_haircut(9, 100), rel_tol=1e-12
    )


def test_provenance_carried_into_decision_and_dict():
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=4, breadth_provenance="declared")
    assert d.breadth_provenance == "declared"
    assert d.to_dict()["breadth_provenance"] == "declared"
    assert "effective_min_holdout_sharpe" in d.to_dict()


def test_low_holdout_sharpe_fails_that_check():
    d = evaluate_gate(_wf(holdout_sharpe=0.1), GateCriteria(min_holdout_sharpe=0.5))
    assert d.passed is False
    assert [c["name"] for c in d.checks if not c["passed"]] == ["holdout_sharpe"]


def test_zero_holdout_return_fails_strict_gt():
    d = evaluate_gate(_wf(holdout_return=0.0), GateCriteria())
    assert d.passed is False
    assert [c["name"] for c in d.checks if not c["passed"]] == ["holdout_return"]


def test_low_pct_positive_and_negative_window_fail():
    d = evaluate_gate(_wf(pct_positive=0.4, min_sharpe=-0.5), GateCriteria())
    assert d.passed is False
    failed = {c["name"] for c in d.checks if not c["passed"]}
    assert failed == {"pct_positive_windows", "min_window_sharpe"}


def test_infinite_metric_fails_gate_not_passes():
    # float('inf') trivially satisfies >=/>; it must instead fail the check.
    d = evaluate_gate(_wf(holdout_sharpe=float("inf")), GateCriteria())
    assert d.passed is False
    failed = [c for c in d.checks if c["name"] == "holdout_sharpe"]
    assert failed and failed[0]["passed"] is False


def test_nan_metric_fails_gate_and_is_not_recorded_as_value():
    # NaN must not be recorded as a passing value in the decision payload.
    import math

    d = evaluate_gate(_wf(holdout_sharpe=float("nan")), GateCriteria())
    assert d.passed is False
    check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert check["passed"] is False
    # The recorded value is never a raw NaN (it is nulled out instead).
    assert check["value"] is None or not math.isnan(check["value"])


def test_nan_gate_decision_is_json_serializable():
    import json

    json.dumps(evaluate_gate(_wf(holdout_sharpe=float("nan")), GateCriteria()).to_dict())


def test_to_dict_serializable():
    import json
    json.dumps(evaluate_gate(_wf(), GateCriteria()).to_dict())


def test_gate_checks_are_table_driven():
    # #40: gate checks come from a declarative spec, not hand-built literals per call site.
    from algua.research.gates import GATE_SPECS

    names_from_table = {spec.name for spec in GATE_SPECS}
    names_from_eval = {c["name"] for c in evaluate_gate(_wf(), GateCriteria()).checks}
    assert names_from_table == names_from_eval
    # Each spec points at a real GateCriteria threshold attribute.
    for spec in GATE_SPECS:
        assert hasattr(GateCriteria(), spec.threshold_attr)
