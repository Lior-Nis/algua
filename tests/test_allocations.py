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


def test_allocate_and_active(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    a = allocations.active_allocation(conn, sid)
    assert a is not None and a["capital"] == 10_000.0
    assert allocations.total_allocated(conn) == 10_000.0


def test_reallocation_replaces_not_doublecounts(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    allocations.allocate(conn, sid, capital=20_000.0, actor="human", account_equity=50_000.0)
    # old row revoked; only the new capital counts toward the sum
    assert allocations.total_allocated(conn) == 20_000.0
    assert allocations.active_allocation(conn, sid)["capital"] == 20_000.0


def test_sum_cannot_exceed_equity(tmp_path):
    conn = _conn(tmp_path)
    s1, s2 = _strategy(conn, "s1"), _strategy(conn, "s2")
    allocations.allocate(conn, s1, capital=40_000.0, actor="human", account_equity=50_000.0)
    with pytest.raises(allocations.AllocationError, match="exceeds"):
        allocations.allocate(conn, s2, capital=20_000.0, actor="human", account_equity=50_000.0)


def test_deallocate_requires_flat(tmp_path):
    conn = _conn(tmp_path)
    sid = _strategy(conn)
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    allocations.deallocate(conn, sid, actor="human", is_flat=True)
    assert allocations.active_allocation(conn, sid) is None
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    with pytest.raises(allocations.AllocationError, match="flat"):
        allocations.deallocate(conn, sid, actor="human", is_flat=False)


def test_revoke_active_locked_no_commit_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    sid = repo.get("s1").id
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    allocations.revoke_active_locked(conn, sid)
    assert allocations.active_allocation(conn, sid) is None
    allocations.revoke_active_locked(conn, sid)  # idempotent: no-op, no error
    assert allocations.active_allocation(conn, sid) is None
