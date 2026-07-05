"""Unit tests for the pure intake helpers (#317): the cents-floored capital slice and the
deterministic FIFO ordering. The admission DECISION itself is transactional and lives in
``StrategyRepository.intake_candidate_to_paper`` (covered by tests/test_paper_intake.py)."""
from __future__ import annotations

import pytest

from algua.registry.intake import Candidate, order_candidates, slice_capital

# ---------------------------------------------------------------------------
# slice_capital — cents-floored, exact at the boundary
# ---------------------------------------------------------------------------

def test_slice_floors_to_cents():
    assert slice_capital(100_000, 3) == 33_333.33
    assert slice_capital(100_000, 5) == 20_000.0


def test_slice_non_positive_equity():
    assert slice_capital(0, 4) == 0.0
    assert slice_capital(-100, 4) <= 0


def test_slice_never_rounds_up_k_slices_sum_within_equity():
    # k slices must sum to <= equity for any k <= max_concurrent (integer-cents floor guarantee),
    # even at an equity that does not divide evenly. 100000/3 -> 33333.33; 3 * 33333.33 = 99999.99.
    slc = slice_capital(100_000, 3)
    assert slc * 3 <= 100_000


def test_slice_capital_zero_max_concurrent_raises():
    with pytest.raises(ValueError, match='max_concurrent must be positive'):
        slice_capital(100_000, 0)


def test_slice_never_rounds_up_at_sub_cent_equity():
    # Regression: a sub-cent equity must floor DOWN, never up (round(0.019*100)==2 would give a
    # $0.02 slice > $0.019 equity, violating the never-rounds-up contract and Σ<=equity).
    assert slice_capital(0.019, 1) <= 0.019
    assert slice_capital(0.019, 1) == 0.01


def test_slice_floors_binary_float_artifact_at_exact_cent():
    # 0.29 * 100 == 28.9999… in binary float; the decimal floor must still give 29 cents, not 28.
    assert slice_capital(0.29, 1) == 0.29


# ---------------------------------------------------------------------------
# order_candidates — FIFO by monotonic entry_id, tie-break sid
# ---------------------------------------------------------------------------

def test_order_empty():
    assert order_candidates([]) == []


def test_order_fifo_by_entry_id():
    cands = [
        Candidate('c', 30, 1),
        Candidate('a', 10, 2),
        Candidate('b', 20, 3),
    ]
    assert [c.name for c in order_candidates(cands)] == ['a', 'b', 'c']


def test_entry_id_beats_sid():
    # earlier entry_id + HIGHER sid must order before later entry_id + lower sid
    early_high_sid = Candidate('early', 10, 999)
    late_low_sid = Candidate('late', 20, 1)
    assert [c.name for c in order_candidates([late_low_sid, early_high_sid])] == ['early', 'late']


def test_sid_breaks_entry_id_ties():
    a = Candidate('a', 10, 5)
    b = Candidate('b', 10, 2)
    assert [c.name for c in order_candidates([a, b])] == ['b', 'a']
