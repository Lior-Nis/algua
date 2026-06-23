import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import allocations
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def _paper_strategy(tmp_path):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    return repo


def test_bench_to_dormant_requires_reason(tmp_path):
    repo = _paper_strategy(tmp_path)
    with pytest.raises(TransitionError, match="reason"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="")


def test_bench_to_dormant_with_reason_succeeds(tmp_path):
    repo = _paper_strategy(tmp_path)
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT,
                              reason="seasonal signal degradation")
    assert rec.stage is Stage.DORMANT
    assert repo.list_transitions("s1")[-1]["reason"] == "seasonal signal degradation"


def _live_strategy(tmp_path):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    return repo, conn


def test_live_to_dormant_rejected_when_not_flat(tmp_path):
    repo, conn = _live_strategy(tmp_path)
    conn.execute(
        "INSERT INTO live_fills(activity_id, strategy, symbol, qty, price, fill_ts) "
        "VALUES (?,?,?,?,?,?)",
        ("a1", "s1", "AAPL", 5.0, 100.0, "2026-01-01T00:00:00Z"))
    conn.commit()
    with pytest.raises(TransitionError, match="flat"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")


def test_live_to_dormant_flat_succeeds_and_revokes_allocation(tmp_path):
    repo, conn = _live_strategy(tmp_path)
    allocations.allocate(conn, repo.get("s1").id, capital=10_000.0, actor="human",
                         account_equity=50_000.0)
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, rec.id) is None


def test_apply_transition_revoke_enforces_flatness_atomically(tmp_path):
    # #247: flatness is now enforced INSIDE apply_transition's revoke+CAS transaction, not only by
    # the transitions.py caller. Calling apply_transition(revoke_allocation=True) DIRECTLY on a
    # non-flat strategy must raise and leave BOTH the stage and the allocation untouched (atomic).
    # Before the fix, store ran no flatness check here, so this would have benched a strategy still
    # holding an open position (the TOCTOU's orphan).
    repo, conn = _live_strategy(tmp_path)
    rec = repo.get("s1")
    allocations.allocate(conn, rec.id, capital=10_000.0, actor="human", account_equity=50_000.0)
    conn.execute(
        "INSERT INTO live_fills(activity_id, strategy, symbol, qty, price, fill_ts) "
        "VALUES (?,?,?,?,?,?)",
        ("a1", "s1", "AAPL", 5.0, 100.0, "2026-01-01T00:00:00Z"))
    conn.commit()
    with pytest.raises(TransitionError, match="flat"):
        repo.apply_transition(rec, Stage.DORMANT, Actor.AGENT, reason="bench",
                              revoke_allocation=True)
    assert repo.get("s1").stage is Stage.LIVE                      # stage CAS rolled back
    assert allocations.active_allocation(conn, rec.id) is not None  # allocation NOT revoked


def test_apply_transition_revoke_is_top_level_only(tmp_path):
    # #247: the BEGIN IMMEDIATE bench path refuses to run inside an already-open transaction (a
    # manual BEGIN there raises, and the blanket rollback could discard a surrounding tx).
    repo, conn = _live_strategy(tmp_path)
    rec = repo.get("s1")
    conn.execute("BEGIN")
    try:
        with pytest.raises(RuntimeError, match="top level"):
            repo.apply_transition(rec, Stage.DORMANT, Actor.AGENT, reason="bench",
                                  revoke_allocation=True)
    finally:
        conn.rollback()
