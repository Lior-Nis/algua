from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.portfolio.construction import (
    CONSTRUCTION_POLICIES,
    ConstructionError,
    apply_gross_utilization,
    equal_weight_positive,
    get_construction_policy,
    score_proportional_long,
    top_k_equal_weight,
    validate_construction_params,
)

_EMPTY = pd.DataFrame()  # starter policies ignore `view`


def test_top_k_equal_weight_selects_top_k_equal():
    scores = pd.Series({"A": 0.3, "B": 0.1, "C": 0.2, "D": -0.5})
    w = top_k_equal_weight(scores, _EMPTY, {"top_k": 2})
    assert set(w.index) == {"A", "C"}
    assert w.to_dict() == pytest.approx({"A": 0.5, "C": 0.5})


def test_top_k_tie_break_is_deterministic_by_symbol():
    # B and C tie at 0.2; with top_k=2 and A=0.3 highest, the tie must resolve to the
    # lexicographically-smaller symbol (B), regardless of input order.
    ordered = pd.Series({"A": 0.3, "B": 0.2, "C": 0.2})
    shuffled = pd.Series({"C": 0.2, "A": 0.3, "B": 0.2})
    wo = top_k_equal_weight(ordered, _EMPTY, {"top_k": 2})
    ws = top_k_equal_weight(shuffled, _EMPTY, {"top_k": 2})
    assert set(wo.index) == {"A", "B"}
    assert set(ws.index) == {"A", "B"}


def test_policies_drop_nonfinite_scores_not_zero_fill():
    scores = pd.Series({"A": 0.3, "B": np.nan, "C": 0.2})
    # B is dropped (no opinion), NOT treated as a 0.0 score that could be selected.
    w = top_k_equal_weight(scores, _EMPTY, {"top_k": 3})
    assert set(w.index) == {"A", "C"}


def test_policies_fail_closed_on_non_numeric_scores():
    scores = pd.Series({"A": "high", "B": "low"})
    with pytest.raises(ConstructionError):
        top_k_equal_weight(scores, _EMPTY, {"top_k": 1})


def test_equal_weight_positive():
    scores = pd.Series({"A": 1.0, "B": -1.0, "C": 0.0, "D": 2.0})
    w = equal_weight_positive(scores, _EMPTY, {})
    assert set(w.index) == {"A", "D"}
    assert w.to_dict() == pytest.approx({"A": 0.5, "D": 0.5})


def test_equal_weight_positive_all_nonpositive_is_flat():
    scores = pd.Series({"A": -1.0, "B": 0.0})
    assert equal_weight_positive(scores, _EMPTY, {}).empty


def test_score_proportional_long_normalizes_positives_to_gross_one():
    scores = pd.Series({"A": 3.0, "B": 1.0, "C": -5.0})
    w = score_proportional_long(scores, _EMPTY, {})
    assert w.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    assert float(w.sum()) == pytest.approx(1.0)


def test_get_construction_policy_unknown_raises():
    with pytest.raises(ConstructionError):
        get_construction_policy("does_not_exist")


def test_validate_top_k_requires_positive_int():
    validate_construction_params("top_k_equal_weight", {"top_k": 3})
    for bad in ({}, {"top_k": 0}, {"top_k": -1}, {"top_k": 2.5}, {"top_k": True}, {"top_k": "3"}):
        with pytest.raises(ConstructionError):
            validate_construction_params("top_k_equal_weight", bad)


def test_validate_rejects_unknown_keys_and_nonfinite_values():
    with pytest.raises(ConstructionError):
        validate_construction_params("equal_weight_positive", {"surprise": 1})
    with pytest.raises(ConstructionError):
        validate_construction_params("top_k_equal_weight", {"top_k": 2, "x": float("nan")})


def test_dispatch_view_is_read_only():
    with pytest.raises(TypeError):
        CONSTRUCTION_POLICIES["new"] = None  # type: ignore[index]


# --- gross utilization (#560) -----------------------------------------------------------------

def _w(**kv):
    return pd.Series(kv, dtype="float64")


def test_a_full_gross_vector_is_scaled_to_the_target():
    """The #560 fix. Policies normalize gross to exactly max_gross_exposure, which breaches the
    realized-gross wall as soon as the book appreciates -- the wall marks positions against an
    equity denominator capped at the original allocation, so profit alone pushes gross over 1."""
    out = apply_gross_utilization(_w(A=0.5, B=0.5), target_gross=0.98, max_gross=1.0)
    assert out.to_dict() == {"A": 0.49, "B": 0.49}
    assert abs(out.abs().sum() - 0.98) < 1e-12


def test_a_vector_already_inside_the_target_is_left_alone():
    """Tighten-only. Inflating up to the target would UNDO a capacity cap or a tighten-only
    overlay -- both of which reduce gross deliberately."""
    small = _w(A=0.2, B=0.1)
    assert apply_gross_utilization(small, target_gross=0.98, max_gross=1.0).to_dict() == \
        small.to_dict()


def test_an_over_leveraged_vector_is_NOT_rescued():
    """The rail must still reject it.

    Scaling a vector whose gross exceeds max_gross into range would convert a hard "this strategy
    is over-leveraged" rejection into a silent rescue -- exactly what a gross rail exists to stop.
    """
    hot = _w(A=1.5, B=1.5)
    assert apply_gross_utilization(hot, target_gross=0.98, max_gross=1.0).to_dict() == hot.to_dict()


def test_scaling_only_ever_reduces_magnitude():
    """A vector that passed the per-symbol rail and the capacity cap must still pass them."""
    before = _w(A=0.6, B=-0.4)
    after = apply_gross_utilization(before, target_gross=0.98, max_gross=1.0)
    assert (after.abs() <= before.abs() + 1e-12).all()
    assert (np.sign(after) == np.sign(before)).all(), "scaling must not flip a side"


def test_a_non_finite_gross_is_left_for_the_validator():
    bad = _w(A=float("nan"), B=0.5)
    out = apply_gross_utilization(bad, target_gross=0.98, max_gross=1.0)
    assert out.equals(bad)


def test_empty_weights_are_returned_unchanged():
    empty = pd.Series(dtype="float64")
    assert apply_gross_utilization(empty, target_gross=0.98, max_gross=1.0).empty
