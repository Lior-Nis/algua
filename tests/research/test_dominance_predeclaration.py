"""CI-enforcing tests for the dominance-audit predeclaration scaffolding (Task 4, Slice 4 of #221).

These tests FAIL if the predeclared constants or shadow audit fields are absent from gates.py.
That is intentional — the constants must be committed with the code that reads them so the
retirement audit (Slice 5) can filter decision_json rows with pre-committed thresholds.
"""
from __future__ import annotations

import datetime
import math

from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import (
    GateCriteria,
    evaluate_gate,
    sharpe_haircut,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOLDOUT = {
    "sharpe": 7.0,
    "total_return": 0.2,
    "n_bars": 252,
    "skewness": 0.0,
    "kurtosis": 3.0,
}
_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _make_wf(
    *,
    sharpe: float = 7.0,
    n_bars: int = 252,
    holdout_returns: tuple[list[float], list[str]] | None = None,
    market_returns: tuple[list[float], list[str]] | None = None,
) -> WalkForwardResult:
    """Build a WalkForwardResult that passes all non-regime gate checks."""
    return WalkForwardResult(
        strategy="test_strat",
        config_hash="c",
        data_source="synthetic",
        snapshot_id=None,
        timeframe="1d",
        seed=None,
        period={"start": "2020-01-01", "end": "2021-01-01"},
        windows=4,
        holdout_frac=0.2,
        window_metrics=[],
        holdout_metrics={**_HOLDOUT, "sharpe": sharpe, "n_bars": n_bars},
        stability=dict(_STAB),
        holdout_returns=holdout_returns,
        market_returns=market_returns,
    )


def _robust_dates(n: int, start_date: datetime.date = datetime.date(2020, 1, 1)) -> list[str]:
    return [(start_date + datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ---------------------------------------------------------------------------
# Test 1 — CI enforcement: constants must be importable with correct values
# ---------------------------------------------------------------------------

def test_dominance_audit_constants_predeclared() -> None:
    """CI enforcement: this test FAILS if any dominance-audit constant is absent from gates.py.

    These constants are predeclared here (not post-hoc) so the Slice 5 retirement audit can
    filter decision_json rows (where phase3_component_mask has all required bits set) and check
    that no 'haircut_fail AND dsr_raw_N_pass' case exceeds DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS
    over >= DOMINANCE_AUDIT_MIN_PROMOTIONS promotions across >= DOMINANCE_AUDIT_MIN_WINDOW_DAYS.
    """
    from algua.research.gates import (  # noqa: PLC0415 — intentional late import for CI enforcement
        DOMINANCE_AUDIT_MIN_PROMOTIONS,
        DOMINANCE_AUDIT_MIN_WINDOW_DAYS,
        DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS,
    )
    assert DOMINANCE_AUDIT_MIN_PROMOTIONS == 30
    assert DOMINANCE_AUDIT_MIN_WINDOW_DAYS == 90
    assert DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS == 0


# ---------------------------------------------------------------------------
# Test 2 — phase3_component_mask recorded on every GateDecision
# ---------------------------------------------------------------------------

def test_phase3_component_mask_recorded() -> None:
    """Every evaluate_gate call records phase3_component_mask == 0b11111 (slices 0-4 active = 31)
    and both shadow fields appear in to_dict().
    """
    from algua.research.gates import PHASE3_COMPONENT_MASK  # noqa: PLC0415

    d = evaluate_gate(_make_wf(sharpe=7.0), GateCriteria(), n_combos=5, pit_ok=True)
    assert d.phase3_component_mask == 0b11111
    assert d.phase3_component_mask == PHASE3_COMPONENT_MASK
    dct = d.to_dict()
    assert "haircut_would_have_blocked" in dct
    assert "phase3_component_mask" in dct


# ---------------------------------------------------------------------------
# Test 3 — haircut_would_have_blocked is True iff the haircut is the blocker
# ---------------------------------------------------------------------------

def test_haircut_would_have_blocked_true_when_haircut_is_the_blocker() -> None:
    """holdout sharpe BETWEEN base bar and base+haircut -> haircut_would_have_blocked == True.
    holdout sharpe above base+haircut -> False.

    We use n_combos=10, n_bars=252 to get a nonzero haircut, then pick a sharpe that sits
    strictly between base_holdout_sharpe (0.5) and effective_holdout_sharpe (0.5 + haircut).
    """
    n_combos = 10
    n_bars = 252
    base = GateCriteria().min_holdout_sharpe  # 0.5
    haircut = sharpe_haircut(n_combos, n_bars)
    effective = base + haircut
    assert haircut > 0, "haircut must be nonzero for this test to be meaningful"

    # Sharpe strictly between base and effective -> passes base, fails effective -> True
    sharpe_between = (base + effective) / 2.0
    assert base < sharpe_between < effective

    wf_between = _make_wf(sharpe=sharpe_between, n_bars=n_bars)
    d_between = evaluate_gate(wf_between, GateCriteria(), n_combos=n_combos, pit_ok=True)
    assert d_between.haircut_would_have_blocked is True, (
        f"Expected haircut_would_have_blocked=True for sharpe={sharpe_between:.4f} "
        f"(base={base:.4f}, effective={effective:.4f})"
    )

    # Sharpe above effective -> passes both -> False
    sharpe_above = effective + 1.0
    wf_above = _make_wf(sharpe=sharpe_above, n_bars=n_bars)
    d_above = evaluate_gate(wf_above, GateCriteria(), n_combos=n_combos, pit_ok=True)
    assert d_above.haircut_would_have_blocked is False, (
        f"Expected haircut_would_have_blocked=False for sharpe={sharpe_above:.4f} "
        f"(effective={effective:.4f})"
    )

    # Sharpe below base -> fails base -> fails effective -> False (base was blocking, not haircut)
    sharpe_below_base = base - 0.1
    wf_below = _make_wf(sharpe=sharpe_below_base, n_bars=n_bars)
    d_below = evaluate_gate(wf_below, GateCriteria(), n_combos=n_combos, pit_ok=True)
    assert d_below.haircut_would_have_blocked is False, (
        f"Expected haircut_would_have_blocked=False for sharpe={sharpe_below_base:.4f} "
        f"(base={base:.4f}) — base bar was the blocker, not haircut"
    )


def test_haircut_would_have_blocked_false_when_no_haircut() -> None:
    """With n_combos=1 the haircut is 0 -> effective == base -> nothing is "between" -> False."""
    base = GateCriteria().min_holdout_sharpe
    haircut = sharpe_haircut(1, 252)
    assert haircut == 0.0

    # Sharpe just above base (would have been "between" if there were a haircut, but there isn't)
    sharpe_above_base = base + 0.1
    wf = _make_wf(sharpe=sharpe_above_base, n_bars=252)
    d = evaluate_gate(wf, GateCriteria(), n_combos=1, pit_ok=True)
    assert d.haircut_would_have_blocked is False


def test_haircut_would_have_blocked_inf_effective() -> None:
    """Degenerate holdout (n_bars=0) drives haircut to inf -> effective bar = inf.
    haircut_would_have_blocked must be True iff sharpe >= base (the base bar would pass but the
    haircut makes it unreachable — exactly 'the haircut is what blocks it').
    """
    base = GateCriteria().min_holdout_sharpe
    haircut = sharpe_haircut(5, 0)
    assert not math.isfinite(haircut)

    # Sharpe above base but holdout is degenerate -> True (haircut blocks everything above base)
    wf_above_base = _make_wf(sharpe=base + 1.0, n_bars=0)
    d_above = evaluate_gate(wf_above_base, GateCriteria(), n_combos=5, pit_ok=True)
    assert d_above.haircut_would_have_blocked is True

    # Sharpe below base -> False (base was the blocker, not the haircut)
    wf_below_base = _make_wf(sharpe=base - 0.1, n_bars=0)
    d_below = evaluate_gate(wf_below_base, GateCriteria(), n_combos=5, pit_ok=True)
    assert d_below.haircut_would_have_blocked is False


# ---------------------------------------------------------------------------
# Test 4 — shadow fields do NOT affect passed
# ---------------------------------------------------------------------------

def test_shadow_fields_do_not_affect_passed() -> None:
    """haircut_would_have_blocked and phase3_component_mask are audit-only: they must not
    enter decision.passed or appear in the checks list.

    Specifically: a wf where haircut_would_have_blocked=True (sharpe between base and effective)
    must still have its `passed` determined only by the existing checks, not by these fields.
    """
    n_combos = 10
    n_bars = 252
    base = GateCriteria().min_holdout_sharpe
    haircut = sharpe_haircut(n_combos, n_bars)
    effective = base + haircut

    # Sharpe strictly between base and effective: haircut_would_have_blocked=True
    sharpe_between = (base + effective) / 2.0

    d = evaluate_gate(
        _make_wf(sharpe=sharpe_between, n_bars=n_bars),
        GateCriteria(),
        n_combos=n_combos,
        pit_ok=True,
    )

    assert d.haircut_would_have_blocked is True  # confirms shadow field is True

    # `passed` must be False because the holdout_sharpe check itself fails (sharpe < effective)
    # The key test: the holdout_sharpe check caused passed=False, NOT haircut_would_have_blocked
    holdout_check = next(c for c in d.checks if c["name"] == "holdout_sharpe")
    assert holdout_check["passed"] is False, "holdout_sharpe check must fail (sharpe < effective)"
    assert d.passed is False

    # Shadow fields must NOT appear in checks
    check_names = {c["name"] for c in d.checks}
    assert "haircut_would_have_blocked" not in check_names
    assert "phase3_component_mask" not in check_names

    # Now verify with sharpe above effective: haircut_would_have_blocked=False, gate can pass
    sharpe_above = effective + 1.0
    d_above = evaluate_gate(
        _make_wf(sharpe=sharpe_above, n_bars=n_bars),
        GateCriteria(),
        n_combos=n_combos,
        pit_ok=True,
    )
    assert d_above.haircut_would_have_blocked is False
    # phase3_component_mask is still recorded regardless of outcome
    assert d_above.phase3_component_mask == 0b11111


# ---------------------------------------------------------------------------
# Test 5 — both fields appear in to_dict with correct types
# ---------------------------------------------------------------------------

def test_shadow_fields_in_to_dict_types() -> None:
    """Both shadow fields appear in to_dict() with correct Python types."""
    d = evaluate_gate(_make_wf(sharpe=7.0), GateCriteria(), n_combos=5, pit_ok=True)
    dct = d.to_dict()

    assert "haircut_would_have_blocked" in dct
    assert "phase3_component_mask" in dct
    assert isinstance(dct["haircut_would_have_blocked"], bool)
    assert isinstance(dct["phase3_component_mask"], int)
    assert dct["phase3_component_mask"] == 31  # 0b11111


# ---------------------------------------------------------------------------
# Test 6 — phase3_component_mask is 0b11111 == 31 on every call (not None)
# ---------------------------------------------------------------------------

def test_phase3_component_mask_always_set() -> None:
    """phase3_component_mask must be an int (never None) on every evaluate_gate call."""
    for sharpe in [0.1, 0.5, 1.0, 5.0, 10.0]:
        d = evaluate_gate(_make_wf(sharpe=sharpe), GateCriteria(), pit_ok=True)
        assert d.phase3_component_mask == 0b11111, (
            f"phase3_component_mask must be 0b11111 (31) for sharpe={sharpe}"
        )
