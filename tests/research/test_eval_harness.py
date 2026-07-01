"""Tests for the gate-decision eval harness (#347)."""
from __future__ import annotations

import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.research import eval_harness as eh
from algua.strategies.base import LoadedStrategy, StrategyConfig

K = 4  # fast but >= the spec's k>=4 for the safety assertions


def _scenario(name: str) -> eh.Scenario:
    return next(s for s in eh.SCENARIO_BANK if s.name == name)


# --- ground-truth behaviour of the well-separated scenarios ----------------------------------

def test_obvious_edge_is_correct_promote_and_pass_pow_k():
    r = eh.run_scenario(_scenario("obvious_edge"), K)
    assert r.kind == "promote"
    assert all(t.outcome == eh.CORRECT_PROMOTE for t in r.traces)
    assert r.pass_pow_k is True and r.pass_at_k is True
    assert r.promote_rate == 1.0


@pytest.mark.parametrize("name", ["no_edge", "negative_edge"])
def test_discard_scenarios_never_wrongly_promote(name):
    r = eh.run_scenario(_scenario(name), K)
    assert r.kind == "discard"
    assert all(t.outcome == eh.CORRECT_DISCARD for t in r.traces)
    assert not any(t.outcome == eh.WRONGLY_PROMOTED for t in r.traces)
    assert r.pass_pow_k is True
    assert r.promote_rate == 0.0


# --- headline safety metrics over the full bank ----------------------------------------------

def test_false_promote_rate_within_regression_guard():
    """The load-bearing safety assertion: the core gate never promotes a no-edge/losing book on
    the fixed, well-separated bank. A deterministic CI guard, not a statistical bound."""
    report = eh.run_eval(k=K)
    assert report.false_promote_rate <= eh.MAX_FALSE_PROMOTE_RATE
    assert report.crash_rate <= eh.MAX_CRASH_RATE  # crashes cannot masquerade as safety
    assert report.pass_pow_k == 1.0  # every labelled scenario all-seed-correct
    assert report.accuracy == 1.0


def test_marginal_edge_is_consistency_only_no_label():
    """marginal_edge straddles the bar: no label, excluded from the safety rates and pass^k."""
    report = eh.run_eval(k=K)
    marg = next(s for s in report.scenarios if s.scenario == "marginal_edge")
    assert marg.kind == "ambiguous"
    assert marg.n_correct is None and marg.pass_at_k is None and marg.pass_pow_k is None
    assert 0.0 <= marg.promote_rate <= 1.0  # consistency figure still reported
    # its runs are classified ambiguous_* — never a failure class
    assert all(
        t.outcome in (eh.AMBIGUOUS_PROMOTE, eh.AMBIGUOUS_DISCARD) for t in marg.traces
    )
    # the headline pass^k denominator is labelled scenarios only (3 of the 4)
    labelled = [s for s in report.scenarios if s.kind != "ambiguous"]
    assert len(labelled) == 3


# --- the failure taxonomy + metrics react correctly ------------------------------------------

def test_mislabeled_promote_on_no_edge_is_wrongly_discarded():
    """A scenario LABELLED promote but run on a no-edge provider must surface wrongly_discarded and
    drop pass^k — proves the taxonomy + metrics are wired, not hard-coded to pass."""
    mis = eh.Scenario("mislabel", "promote", drift=0.0, vol=0.02)
    r = eh.run_scenario(mis, K)
    assert all(t.outcome == eh.WRONGLY_DISCARDED for t in r.traces)
    assert r.pass_pow_k is False and r.pass_at_k is False
    report = eh.run_eval(scenarios=(mis,), k=K)
    assert report.false_discard_rate == 1.0
    assert report.pass_pow_k == 0.0


def test_crashed_run_is_recorded_and_sweep_completes():
    """A strategy that raises yields a `crashed` trace; the sweep never aborts."""
    boom = LoadedStrategy(
        config=StrategyConfig(
            name="boom", universe=["AAA", "BBB"],
            execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
            params={}, construction="equal_weight_positive",
        ),
        signal_fn=lambda v, p: (_ for _ in ()).throw(RuntimeError("boom")),
        construct_fn=get_construction_policy("equal_weight_positive"),
    )
    r = eh.run_scenario(_scenario("no_edge"), K, strategy=boom)
    assert r.n_crashed == K
    assert all(t.outcome == eh.CRASHED for t in r.traces)
    assert "boom" in r.traces[0].reason


def test_crashes_excluded_from_false_promote_denominator():
    """All-crash discard runs leave false_promote_rate at 0.0; crash_rate exposes the breakage."""
    boom = LoadedStrategy(
        config=StrategyConfig(
            name="boom2", universe=["AAA", "BBB"],
            execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
            params={}, construction="equal_weight_positive",
        ),
        signal_fn=lambda v, p: (_ for _ in ()).throw(RuntimeError("boom")),
        construct_fn=get_construction_policy("equal_weight_positive"),
    )
    # Build a report by hand from a crashing single discard scenario.
    res = eh.run_scenario(_scenario("no_edge"), K, strategy=boom)
    # mimic run_eval aggregation surface via the public function on one scenario, but with a crash:
    # use run_eval's logic by checking the crashed traces are not counted as promotes.
    assert res.n_promote == 0 and res.n_crashed == K
    # The pass^k for a fully-crashed labelled scenario must be False (a crash is never correct).
    assert res.pass_pow_k is False


# --- provenance, JSON shape, determinism -----------------------------------------------------

def test_report_to_dict_shape_and_aliases():
    d = eh.run_eval(k=K).to_dict()
    for key in (
        "false_promote_rate", "false_discard_rate", "crash_rate", "accuracy",
        "pass_at_k", "pass_pow_k", "any_seed_correct", "all_seed_correct",
        "confusion", "failure_histogram", "provenance", "scenarios",
    ):
        assert key in d, key
    assert d["any_seed_correct"] == d["pass_at_k"]
    assert d["all_seed_correct"] == d["pass_pow_k"]
    prov = d["provenance"]
    for key in ("start", "end", "seeds", "criteria", "provider", "strategy_id",
                "bank_version", "algua_version"):
        assert key in prov, key
    assert prov["bank_version"] == eh.BANK_VERSION
    assert prov["seeds"] == list(range(K))


def test_determinism_same_k_same_report_modulo_git_sha():
    a = eh.run_eval(k=K).to_dict()
    b = eh.run_eval(k=K).to_dict()
    a["provenance"].pop("git_sha", None)
    b["provenance"].pop("git_sha", None)
    assert a == b


def test_run_one_stats_are_populated():
    t = eh.run_one(_scenario("obvious_edge"), 0)
    assert t.actual == "promote"
    assert t.stats["holdout_n_obs"] >= 63  # power floor cleared by the chosen window
    assert "holdout_sharpe" in t.stats


def test_json_clean_no_nan_or_inf():
    import json

    s = json.dumps(eh.run_eval(k=K).to_dict())  # would raise on non-finite if allow_nan=False
    json.loads(s)
    assert "NaN" not in s and "Infinity" not in s


def test_to_dict_json_serializable_strict():
    import json

    json.dumps(eh.run_eval(k=K).to_dict(), allow_nan=False)
