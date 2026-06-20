import math

import pytest
from scipy.stats import norm

from algua.backtest._constants import ANN
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    MAX_BOOTSTRAP_BLOCK_LEN_FRACTION,
    GateCriteria,
    dsr_confidence,
    dsr_sr_star,
    dsr_sr_star_annualized,
    evaluate_gate,
    floored_trial_var_per_period,
)


def test_constants():
    assert DSR_BOOTSTRAP_RESAMPLES == 2000
    assert DSR_BOOTSTRAP_LOWER_QUANTILE == 0.05
    assert MAX_BOOTSTRAP_BLOCK_LEN_FRACTION == 0.5


def test_floored_var_own_first_then_max():
    assert floored_trial_var_per_period(0.04, 0.01) == 0.04        # floor below own -> own
    assert floored_trial_var_per_period(0.01, 0.09) == 0.09        # floor above own -> floor
    assert floored_trial_var_per_period(0.04, None) == 0.04        # no floor -> own
    assert floored_trial_var_per_period(-1.0, 0.09) is None        # degenerate own fails closed
    assert floored_trial_var_per_period(float("nan"), 0.09) is None


def test_dsr_sr_star_matches_inline_formula():
    assert dsr_sr_star(1, 0.04) == 0.0           # n<=1 -> 0
    assert dsr_sr_star(0, 0.04) is None          # n<1 -> None
    assert dsr_sr_star(50, -1.0) is None         # negative var -> None
    v = dsr_sr_star(50, 0.04)
    assert v is not None and v > 0.0


def test_dsr_sr_star_consistency_with_dsr_confidence():
    # dsr_confidence must use exactly dsr_sr_star internally — verify the refactor is behavior-true.
    n, var = 40, 0.04
    sr_star = dsr_sr_star(n, var)
    # Reconstruct the closed-form confidence from sr_star and compare to dsr_confidence.
    sr, t, skew, kurt = 0.12, 90, -0.2, 4.0
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    assert dsr_confidence(sr, t, skew, kurt, n, var) == float(norm.cdf(z))


def test_dsr_sr_star_annualized_floors_and_converts():
    # annualized var / ANN, floored, then sr_star
    own_ann, floor_ann, n = 0.04 * ANN, 0.09 * ANN, 40
    assert dsr_sr_star_annualized(n, own_ann, floor_ann) == dsr_sr_star(n, 0.09)
    assert dsr_sr_star_annualized(n, None, floor_ann) is None      # no own var -> None


# ---------------------------------------------------------------------------
# Task 3: dsr_bootstrap AND-check in evaluate_gate
# ---------------------------------------------------------------------------

_HOLDOUT = {
    "sharpe": 7.0,
    "total_return": 0.2,
    "n_bars": 252,
    "skewness": 0.0,
    "kurtosis": 3.0,
}
_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


@pytest.fixture
def make_wf():
    """Build a WalkForwardResult with configurable holdout Sharpe; all non-DSR checks pass."""

    def _make(sharpe: float = 7.0) -> WalkForwardResult:
        return WalkForwardResult(
            strategy="test_strat",
            config_hash="c",
            data_source="synthetic",
            snapshot_id=None,
            timeframe="1d",
            seed=None,
            period={"start": "2024-01-01", "end": "2024-06-01"},
            windows=4,
            holdout_frac=0.2,
            window_metrics=[],
            holdout_metrics={**_HOLDOUT, "sharpe": sharpe},
            stability=dict(_STAB),
        )

    return _make


def test_bootstrap_check_appended_only_when_binding(make_wf):
    wf = make_wf(sharpe=2.0)
    # binding -> appended
    d = evaluate_gate(wf, GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                      dsr_trial_var_ann=0.04 * 252, bootstrap_binding=True,
                      bootstrap_lower_confidence=0.99)
    assert any(c["name"] == "dsr_bootstrap" for c in d.checks)
    # not binding -> omitted entirely
    d2 = evaluate_gate(wf, GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                       dsr_trial_var_ann=0.04 * 252, bootstrap_binding=False)
    assert all(c["name"] != "dsr_bootstrap" for c in d2.checks)


def test_bootstrap_none_is_failed_when_binding(make_wf):
    d = evaluate_gate(make_wf(sharpe=7.0), GateCriteria(), n_combos=5, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=0.04 * 252, bootstrap_binding=True,
                      bootstrap_lower_confidence=None)
    chk = next(c for c in d.checks if c["name"] == "dsr_bootstrap")
    assert chk["passed"] is False
    assert d.passed is False


def test_bootstrap_tighten_only(make_wf):
    # For any bootstrap value, the gate's pass is old_pass AND (NOT binding OR bootstrap_pass).
    wf = make_wf(sharpe=7.0)
    base = dict(criteria=GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                dsr_trial_var_ann=0.04 * 252)
    old = evaluate_gate(wf, **base)                                   # no bootstrap
    for lower in [None, 0.0, 0.5, 0.99, 1.0]:
        new = evaluate_gate(wf, **base, bootstrap_binding=True, bootstrap_lower_confidence=lower)
        if new.passed:
            assert old.passed                                        # never FAIL->PASS
    # audit fields surface
    d = evaluate_gate(wf, **base, bootstrap_binding=True, bootstrap_lower_confidence=0.97,
                      bootstrap_seed=42, bootstrap_b=2000, bootstrap_block_len=8)
    dd = d.to_dict()
    assert dd["dsr_bootstrap_lower"] == 0.97 and dd["dsr_bootstrap_seed"] == 42
    assert dd["dsr_bootstrap_b"] == 2000 and dd["dsr_bootstrap_block_len"] == 8


# Finding 2: audit-flag consistency when dsr_binding=False but bootstrap_binding=True
def test_bootstrap_audit_consistent_when_dsr_not_binding(make_wf):
    """dsr_binding=False, bootstrap_binding=True must produce a self-consistent audit state.

    No dsr_bootstrap check must appear, dsr_bootstrap_binding must be False,
    and all dsr_bootstrap_* audit scalars must be None.
    """
    wf = make_wf(sharpe=7.0)
    d = evaluate_gate(
        wf, GateCriteria(), n_combos=5, pit_ok=True,
        dsr_binding=False,
        bootstrap_binding=True,
        bootstrap_lower_confidence=0.99,
        bootstrap_seed=42,
        bootstrap_b=2000,
        bootstrap_block_len=8,
    )
    # No dsr_bootstrap check in checks
    assert all(c["name"] != "dsr_bootstrap" for c in d.checks), (
        "dsr_bootstrap check must be absent when dsr_binding=False"
    )
    # Audit flag must be False (consistent: no check => not armed)
    assert d.dsr_bootstrap_binding is False, (
        f"dsr_bootstrap_binding should be False but got {d.dsr_bootstrap_binding}"
    )
    # All scalar audit fields must be None
    assert d.dsr_bootstrap_seed is None
    assert d.dsr_bootstrap_b is None
    assert d.dsr_bootstrap_block_len is None
    assert d.dsr_bootstrap_lower is None
