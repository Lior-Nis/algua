from __future__ import annotations

import pytest

from algua.registry.intake import Candidate, IntakePlan, plan_intake, slice_capital


def _plan(candidates, **kw):
    defaults = dict(
        occupied_slots=0,
        total_allocated=0.0,
        equity=100_000.0,
        max_concurrent=10,
    )
    defaults.update(kw)
    return plan_intake(candidates, **defaults)


def test_empty_candidates_computes_slice_only():
    plan = _plan([], equity=100_000.0, max_concurrent=4)
    assert isinstance(plan, IntakePlan)
    assert plan.admit == []
    assert plan.queued == []
    assert plan.slice_capital == 25_000.0
    assert plan.occupied_before == 0
    assert plan.max_concurrent == 4


def test_empty_book_admits_all_in_fifo_order():
    cands = [
        Candidate('c', '2026-01-03T00:00:00', 1),
        Candidate('a', '2026-01-01T00:00:00', 2),
        Candidate('b', '2026-01-02T00:00:00', 3),
    ]
    plan = _plan(cands, equity=100_000.0, max_concurrent=10)
    assert plan.admit == ['a', 'b', 'c']
    assert plan.queued == []


def test_fifo_by_entry_ts_beats_sid():
    # earlier entry_ts + HIGHER sid must be admitted before later entry_ts + lower sid
    early_high_sid = Candidate('early', '2026-01-01T00:00:00', 999)
    late_low_sid = Candidate('late', '2026-01-02T00:00:00', 1)
    plan = _plan([late_low_sid, early_high_sid], equity=100_000.0, max_concurrent=10)
    assert plan.admit == ['early', 'late']


def test_sid_breaks_entry_ts_ties():
    a = Candidate('a', '2026-01-01T00:00:00', 5)
    b = Candidate('b', '2026-01-01T00:00:00', 2)
    plan = _plan([a, b], equity=100_000.0, max_concurrent=10)
    assert plan.admit == ['b', 'a']


def test_slot_cap_fully_binds_queues_everything():
    cands = [
        Candidate('a', '2026-01-01T00:00:00', 1),
        Candidate('b', '2026-01-02T00:00:00', 2),
    ]
    plan = _plan(cands, occupied_slots=3, equity=100_000.0, max_concurrent=3)
    assert plan.admit == []
    assert plan.queued == ['a', 'b']


def test_slot_cap_binds_partway():
    cands = [
        Candidate('a', '2026-01-01T00:00:00', 1),
        Candidate('b', '2026-01-02T00:00:00', 2),
    ]
    plan = _plan(cands, occupied_slots=0, equity=100_000.0, max_concurrent=1)
    assert plan.admit == ['a']
    assert plan.queued == ['b']


def test_equity_headroom_binds_before_slot_cap():
    # slice = 100/10 = 10; total_allocated=85 leaves room for exactly one 10-slice
    # (85+10=95<=100 admit; 95+10=105>100 queue). Slot cap (10) never binds.
    cands = [
        Candidate('a', '2026-01-01T00:00:00', 1),
        Candidate('b', '2026-01-02T00:00:00', 2),
        Candidate('c', '2026-01-03T00:00:00', 3),
    ]
    plan = _plan(cands, occupied_slots=0, total_allocated=85.0, equity=100.0, max_concurrent=10)
    assert plan.slice_capital == 10.0
    assert plan.admit == ['a']
    assert plan.queued == ['b', 'c']


def test_slice_floors_to_cents():
    assert slice_capital(100_000, 3) == 33_333.33
    assert slice_capital(100_000, 5) == 20_000.0


def test_slice_non_positive_equity():
    assert slice_capital(0, 4) == 0.0
    assert slice_capital(-100, 4) <= 0


def test_non_positive_slice_admits_nothing():
    cands = [Candidate('a', '2026-01-01T00:00:00', 1)]
    plan = _plan(cands, equity=0.0, max_concurrent=4)
    assert plan.slice_capital <= 0
    assert plan.admit == []
    assert plan.queued == ['a']


def test_slice_capital_zero_max_concurrent_raises():
    with pytest.raises(ValueError, match='max_concurrent must be positive'):
        slice_capital(100_000, 0)


def test_plan_intake_zero_max_concurrent_raises():
    with pytest.raises(ValueError, match='max_concurrent must be positive'):
        plan_intake(
            [],
            occupied_slots=0,
            total_allocated=0.0,
            equity=100_000.0,
            max_concurrent=0,
        )
