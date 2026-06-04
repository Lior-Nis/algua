from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate


def _wf(holdout_sharpe=0.8, holdout_return=0.05, pct_positive=0.75, min_sharpe=0.1):
    return WalkForwardResult(
        strategy="ew", config_hash="abc", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": 100,
                         "total_return": holdout_return, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": holdout_sharpe, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.0, "std_sharpe": 0.3, "min_sharpe": min_sharpe,
                   "pct_positive_windows": pct_positive},
    )


def test_all_thresholds_met_passes():
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=9)
    assert isinstance(d, GateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == {
        "holdout_sharpe", "holdout_return", "pct_positive_windows", "min_window_sharpe"}
    assert all(c["passed"] for c in d.checks)
    assert d.n_combos == 9


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
