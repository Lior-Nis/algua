from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.portfolio.construction import (
    CONSTRUCTION_POLICIES,
    ConstructionError,
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
