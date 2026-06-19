import json
import math
from datetime import date

import pytest

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.db import connect, migrate
from algua.registry.promotion import BreadthContext, run_gate
from algua.registry.store import SqliteStrategyRepository
from algua.research import gates
from algua.research.gates import (
    FUNNEL_WINDOW_DAYS,
    MIN_FUNNEL_FLOOR_STRATEGIES,
    GateCriteria,
    dsr_confidence,
    effective_funnel_breadth,
)

# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_promotion.py; intentionally local to this module
# so this file stands alone)
# ---------------------------------------------------------------------------

_GATE_NAME = "cross_sectional_momentum"
_GATE_START = date(2024, 1, 1)
_GATE_END = date(2024, 6, 1)
_GATE_HOLDOUT = {
    "sharpe": 7.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0,
}
_GATE_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _gate_repo(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add(_GATE_NAME)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    return repo


def _gate_wf(sharpe: float = 7.0):
    return WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[], holdout_metrics={**_GATE_HOLDOUT, "sharpe": sharpe},
        stability=dict(_GATE_STAB))


def _breadth(repo, provenance: str, *, n: int = 5) -> BreadthContext:
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    n_funnel = effective_funnel_breadth(n, windowed_total)
    return BreadthContext(n_funnel, n, windowed_total, provenance)


def _run_gate(repo, breadth, sharpe: float = 7.0):
    return run_gate(
        repo, _gate_wf(sharpe), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=breadth, universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")


def test_min_funnel_floor_strategies_value():
    assert MIN_FUNNEL_FLOOR_STRATEGIES == 5


def test_floor_below_own_leaves_confidence_unchanged():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.04)
    own = dsr_confidence(**base)
    floored = dsr_confidence(**base, funnel_floor_var_per_period=0.01)  # floor < own
    assert own is not None and floored is not None
    assert floored == own  # max(0.04, 0.01) == 0.04 -> no change


def test_floor_above_own_lowers_confidence():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.01)
    own = dsr_confidence(**base)
    floored = dsr_confidence(**base, funnel_floor_var_per_period=0.09)  # floor > own
    assert own is not None and floored is not None
    assert floored < own  # higher SR* -> lower confidence


def test_floor_none_is_phase1_behavior():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.04)
    assert dsr_confidence(**base, funnel_floor_var_per_period=None) == dsr_confidence(**base)


def test_tighten_only_property_over_grid():
    # For every (own_var, floor_var): the floored confidence is <= the un-floored one (never up).
    thresh = 1.0 - gates.DSR_ALPHA
    for own in [0.0, 0.005, 0.01, 0.04, 0.09, 0.16]:
        for floor in [None, 0.0, 0.005, 0.04, 0.09, 0.25]:
            base = dict(sr_obs_per_period=0.12, t=90, skew=-0.2, raw_kurtosis=4.0,
                        n_trials=40, trial_sr_var_per_period=own)
            old = dsr_confidence(**base)
            new = dsr_confidence(**base, funnel_floor_var_per_period=floor)
            if old is None or new is None:
                continue
            assert new <= old + 1e-12
            old_pass = old >= thresh
            new_pass = new >= thresh
            # tighten-only: a pass can only be revoked, never created.
            assert not (new_pass and not old_pass)


# ---------------------------------------------------------------------------
# Integration tests: funnel floor wired into run_gate (#221 Slice 0, Task 3)
#
# These exercise the REAL run_gate + real SqliteStrategyRepository path to
# prove that promotion.py calls repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS)
# and passes the FunnelFloor fields into evaluate_gate / decision_json.
# Harness mirrors tests/test_promotion.py (reuses the same pattern).
# ---------------------------------------------------------------------------

# Sibling strategy names for the funnel (≥5 needed to activate the floor).
_SIBLINGS = ["strat_a", "strat_b", "strat_c", "strat_d", "strat_e"]
# High-dispersion trial stats for each sibling (annualized variance = 0.25 ≫ own 0.001).
_SIBLING_VAR = 0.25
_OWN_VAR = 0.001  # near-duplicate own sweep — floor should dominate


def _build_dispersed_funnel(repo) -> None:
    """Add 5 sibling strategies with high trial-Sharpe dispersion to the search_trials table.
    The strategy under promotion (_GATE_NAME) has a low own_var (≈0); the funnel floor will
    be the mean of per-strategy pooled vars ≈ _SIBLING_VAR (much larger than _OWN_VAR)."""
    for sname in _SIBLINGS:
        repo.record_search_trial(
            sname, 3, "{}",
            trial_sharpe_count=10,
            trial_sharpe_mean=0.4,
            trial_sharpe_var_ann=_SIBLING_VAR,
        )
    # The strategy under promotion has a single low-dispersion sweep.
    repo.record_search_trial(
        _GATE_NAME, 5, "{}",
        trial_sharpe_count=5,
        trial_sharpe_mean=0.5,
        trial_sharpe_var_ann=_OWN_VAR,
    )


def test_floor_recorded_in_decision_when_funnel_dispersed(tmp_path):
    """With ≥5 sibling strategies with real dispersion, dsr_funnel_floor_var_ann must be
    not None and > 0, and dsr_funnel_floor_n_strategies must be ≥ 5."""
    repo = _gate_repo(tmp_path)
    _build_dispersed_funnel(repo)
    breadth = _breadth(repo, "measured")
    outcome = _run_gate(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    assert d.dsr_funnel_floor_var_ann is not None
    assert d.dsr_funnel_floor_var_ann > 0
    assert d.dsr_funnel_floor_n_strategies is not None
    assert d.dsr_funnel_floor_n_strategies >= MIN_FUNNEL_FLOOR_STRATEGIES


def test_floor_fields_appear_in_to_dict_and_decision_json(tmp_path):
    """dsr_funnel_floor_* audit fields must survive the to_dict() + DB round-trip."""
    repo = _gate_repo(tmp_path)
    _build_dispersed_funnel(repo)
    breadth = _breadth(repo, "measured")
    outcome = _run_gate(repo, breadth)
    d_dict = outcome.decision.to_dict()
    assert d_dict["dsr_funnel_floor_var_ann"] is not None
    assert d_dict["dsr_funnel_floor_n_strategies"] is not None
    assert d_dict["dsr_funnel_floor_n_total_rows"] is not None
    # Verify the persisted decision_json matches.
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    stored = json.loads(row["decision_json"])
    assert stored.get("dsr_funnel_floor_var_ann") is not None
    assert stored.get("dsr_funnel_floor_n_strategies") is not None


def test_floor_none_when_funnel_too_small(tmp_path):
    """Fewer than 5 strategies with finite dispersion → var_ann = None (Phase-1 fail-open),
    but the audit count fields should still be recorded (n_strategies and n_total_rows)."""
    repo = _gate_repo(tmp_path)
    # Only 3 siblings → below MIN_FUNNEL_FLOOR_STRATEGIES (5) → floor is None (fail-open).
    for sname in _SIBLINGS[:3]:
        repo.record_search_trial(
            sname, 3, "{}",
            trial_sharpe_count=10,
            trial_sharpe_mean=0.4,
            trial_sharpe_var_ann=_SIBLING_VAR,
        )
    repo.record_search_trial(
        _GATE_NAME, 5, "{}",
        trial_sharpe_count=5,
        trial_sharpe_mean=0.5,
        trial_sharpe_var_ann=_OWN_VAR,
    )
    breadth = _breadth(repo, "measured")
    outcome = _run_gate(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    assert d.dsr_funnel_floor_var_ann is None       # fail-open (< 5 strategies)
    assert d.dsr_funnel_floor_n_strategies is not None
    # Exactly 4 finite-dispersion strategies: 3 siblings + _GATE_NAME (promoted strategy is NOT
    # self-excluded). 4 < MIN_FUNNEL_FLOOR_STRATEGIES (5) → floor is None (fail-open).
    assert d.dsr_funnel_floor_n_strategies == 4


def test_floor_tightens_dsr_confidence_vs_isolated(tmp_path):
    """The dispersed funnel's floor (high var) tightens DSR confidence relative to a funnel
    with too few siblings (where floor is None → Phase-1 behavior).

    Concretely: own_var ≈ 0.001, funnel floor ≈ 0.25. When the floor is active the
    effective trial_sr_var is ≈ 0.25, producing a LOWER dsr_confidence than when the floor
    is absent (own_var = 0.001 used directly). Both cases use sharpe=7.0 (very high) so the
    floor effect is the ONLY meaningful confidence delta."""
    # --- Isolated funnel (3 siblings, floor inactive → Phase-1) ---
    own_var_per_period = _OWN_VAR / ANN
    floor_var_per_period = _SIBLING_VAR / ANN
    # Use n_trials=5 (own breadth), t=252, Sharpe=7.0 mapped to sr_obs_per_period=7/sqrt(252).
    sr_per = 7.0 / math.sqrt(252)
    conf_isolated = dsr_confidence(
        sr_obs_per_period=sr_per, t=252, skew=0.0, raw_kurtosis=3.0,
        n_trials=5, trial_sr_var_per_period=own_var_per_period,
        funnel_floor_var_per_period=None,
    )
    conf_floored = dsr_confidence(
        sr_obs_per_period=sr_per, t=252, skew=0.0, raw_kurtosis=3.0,
        n_trials=5, trial_sr_var_per_period=own_var_per_period,
        funnel_floor_var_per_period=floor_var_per_period,
    )
    assert conf_isolated is not None and conf_floored is not None
    # The floor must be active (floor > own).
    assert floor_var_per_period > own_var_per_period
    # Tighten-only: floored confidence must be ≤ isolated confidence.
    assert conf_floored < conf_isolated, (
        f"Floor {floor_var_per_period} > own {own_var_per_period} but "
        f"confidence did not decrease: isolated={conf_isolated}, floored={conf_floored}"
    )


def test_floor_absent_for_declared_breadth(tmp_path):
    """Declared breadth → dsr_binding=False → floor fields must all be None."""
    repo = _gate_repo(tmp_path)
    _build_dispersed_funnel(repo)
    breadth = _breadth(repo, "declared")
    outcome = _run_gate(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is False
    assert d.dsr_funnel_floor_var_ann is None
    assert d.dsr_funnel_floor_n_strategies is None
    assert d.dsr_funnel_floor_n_total_rows is None


# ---------------------------------------------------------------------------
# Finding A regression tests: degenerate own variance must NEVER be rescued
# by a positive funnel floor (fail-closed on own merits first).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_var", [-1.0, float("-inf"), float("nan"), float("inf")])
def test_degenerate_own_var_not_rescued_by_floor(bad_var: float):
    """dsr_confidence with a degenerate own trial_sr_var_per_period must return None regardless
    of a finite positive funnel floor — the floor must NEVER rescue a FAIL into a PASS.

    Covers: trial_sr_var_per_period in {-1.0, -inf, nan, +inf} each paired with
    funnel_floor_var_per_period=0.04 (finite, positive). Other args are valid.
    This is the core tighten-only invariant: a degenerate own variance is a data-quality failure,
    not a confidence number, and the floor is irrelevant to that failure.
    """
    result = dsr_confidence(
        sr_obs_per_period=0.10,
        t=120,
        skew=0.0,
        raw_kurtosis=3.0,
        n_trials=50,
        trial_sr_var_per_period=bad_var,
        funnel_floor_var_per_period=0.04,
    )
    assert result is None, (
        f"Expected None for degenerate trial_sr_var_per_period={bad_var!r} "
        f"with a positive floor, got {result!r}"
    )


def test_evaluate_gate_degenerate_own_var_negative_with_positive_floor_fails_closed(tmp_path):
    """evaluate_gate with dsr_binding=True, dsr_trial_var_ann=-1.0 (negative own), and
    dsr_funnel_floor_var_ann=0.5 (positive floor) must produce a FAILED dsr_evidence check
    (passed=False) and dsr_confidence=None — the floor must not rescue the degenerate variance.

    This is an integration-level regression: ensures the evaluate_gate → dsr_confidence path
    respects the fail-closed invariant end-to-end.
    """
    wf = _gate_wf(sharpe=7.0)
    decision = gates.evaluate_gate(
        wf,
        GateCriteria(),
        n_combos=50,
        breadth_provenance="measured",
        pit_ok=True,
        dsr_binding=True,
        dsr_trial_var_ann=-1.0,       # degenerate: negative own variance
        dsr_funnel_floor_var_ann=0.5,  # positive floor — must NOT rescue
        dsr_funnel_floor_n_strategies=6,
        dsr_funnel_floor_n_total_rows=30,
    )
    # dsr_confidence must be None (degenerate own → fail-closed regardless of floor)
    assert decision.dsr_confidence is None, (
        f"Expected dsr_confidence=None for negative own var with positive floor, "
        f"got {decision.dsr_confidence!r}"
    )
    # The dsr_evidence check must be present and failed
    dsr_checks = [c for c in decision.checks if c["name"] == "dsr_evidence"]
    assert len(dsr_checks) == 1, f"Expected exactly one dsr_evidence check, got {dsr_checks!r}"
    assert dsr_checks[0]["passed"] is False, (
        f"dsr_evidence check must be passed=False for degenerate own var, got {dsr_checks[0]!r}"
    )
    # The overall gate decision must be False
    assert decision.passed is False
