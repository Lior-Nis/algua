import pytest

from algua.registry import allocations
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _conn(tmp_path):
    conn = connect(tmp_path / "a.db")
    migrate(conn)
    return conn


def _strategy(conn, name="s1"):
    repo = SqliteStrategyRepository(conn)
    repo.add(name)
    return repo.get(name).id


def _alloc(conn, sid, capital, equity, actor="human"):
    with conn:
        allocations.allocate_locked(conn, sid, capital, actor, equity)


def test_allocate_and_active(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    _alloc(conn, sid, 10_000.0, 50_000.0)
    a = allocations.active_allocation(conn, sid)
    assert a is not None and a["capital"] == 10_000.0
    assert allocations.total_allocated(conn) == 10_000.0


def test_reallocation_replaces_not_doublecounts(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    _alloc(conn, sid, 10_000.0, 50_000.0)
    _alloc(conn, sid, 20_000.0, 50_000.0)
    # old row revoked; only the new capital counts toward the sum
    assert allocations.total_allocated(conn) == 20_000.0
    assert allocations.active_allocation(conn, sid)["capital"] == 20_000.0


def test_sum_cannot_exceed_equity(tmp_path):
    conn = _conn(tmp_path)
    s1, s2 = _strategy(conn, "s1"), _strategy(conn, "s2")
    _alloc(conn, s1, 40_000.0, 50_000.0)
    with pytest.raises(allocations.AllocationError, match="exceeds"):
        _alloc(conn, s2, 20_000.0, 50_000.0)


def test_deallocate_requires_flat(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    _alloc(conn, sid, 10_000.0, 50_000.0)
    allocations.deallocate(conn, sid, actor="human", is_flat=True)
    assert allocations.active_allocation(conn, sid) is None
    _alloc(conn, sid, 10_000.0, 50_000.0)
    with pytest.raises(allocations.AllocationError, match="flat"):
        allocations.deallocate(conn, sid, actor="human", is_flat=False)


def test_revoke_active_locked_no_commit_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    sid = repo.get("s1").id
    _alloc(conn, sid, 10_000.0, 50_000.0)
    allocations.revoke_active_locked(conn, sid)
    assert allocations.active_allocation(conn, sid) is None
    allocations.revoke_active_locked(conn, sid)  # idempotent: no-op, no error
    assert allocations.active_allocation(conn, sid) is None


def _set_stage(conn, sid, stage):
    conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, sid))
    conn.commit()


def test_allocate_in_lane_rejects_wrong_stage(tmp_path):
    # A strategy at 'idea' cannot be allocated into a lane that only admits 'live'.
    conn = _conn(tmp_path)
    sid = _strategy(conn)  # newly added -> stage 'idea'
    with pytest.raises(allocations.AllocationError, match="stage"):
        allocations.allocate_in_lane(
            conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0,
            allowed_stages=frozenset({"live"}))
    assert allocations.active_allocation(conn, sid) is None


def test_allocate_in_lane_count_cap(tmp_path):
    # Two paper-lane strategies, cap of 1: the first allocates; a count-INCREASING allocation of the
    # second (no active row yet) is refused by the count cap, not the capital cap.
    conn = _conn(tmp_path)
    s1, s2 = _strategy(conn, "s1"), _strategy(conn, "s2")
    _set_stage(conn, s1, "paper")
    _set_stage(conn, s2, "paper")
    allocations.allocate_in_lane(
        conn, s1, capital=1_000.0, actor="agent", account_equity=1_000_000.0,
        allowed_stages=frozenset({"paper"}), max_concurrent=1)
    with pytest.raises(allocations.CountCapReached, match="capacity"):
        allocations.allocate_in_lane(
            conn, s2, capital=1_000.0, actor="agent", account_equity=1_000_000.0,
            allowed_stages=frozenset({"paper"}), max_concurrent=1)
    assert allocations.active_allocation(conn, s2) is None


def test_allocate_in_lane_resize_is_cap_exempt(tmp_path):
    # A tenant that already holds an active allocation can be RESIZED even at max_concurrent=1 —
    # a resize admits no new tenant, so it is exempt from the count cap.
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    _set_stage(conn, sid, "paper")
    allocations.allocate_in_lane(
        conn, sid, capital=1_000.0, actor="agent", account_equity=1_000_000.0,
        allowed_stages=frozenset({"paper"}), max_concurrent=1)
    # Re-allocate (resize) the SAME tenant at cap 1 — must not raise.
    allocations.allocate_in_lane(
        conn, sid, capital=2_000.0, actor="agent", account_equity=1_000_000.0,
        allowed_stages=frozenset({"paper"}), max_concurrent=1)
    assert allocations.active_allocation(conn, sid)["capital"] == 2_000.0
