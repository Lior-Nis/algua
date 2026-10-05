"""Story 2.2 §2.3, §2.5: which transitions select a drain; ``operator.lock`` on paper-lane exits.

T14: the edges that keep the allocation inside the paper book never reach the selector.
T17: every paper-lane exit takes ``operator.lock``. ``paper -> dormant`` and ``forward_tested ->
live`` join the edges locked before Story 2.2 and are refused with their own message; the earlier
edges keep theirs byte for byte. A held lock refuses before any guard is selected.
"""
from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution import lane_exit
from algua.operator.schedule import operator_run_lock
from algua.registry import allocations, transitions
from algua.registry.db import connect, migrate
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from tests._deployment_helpers import force_legacy_strategy

NEW = "operator.lock is held; a paper-lane exit cannot interleave with a paper tick"
RETIREMENT = "operator.lock is held; deployment retirement cannot interleave with a paper tick"


def _strategy_at(tmp_path, monkeypatch, source: Stage):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    sid = repo.add(name="s1").id
    force_legacy_strategy(conn, sid, stage=source.value)
    with conn:
        allocations.allocate_locked(conn, sid, 10_000.0, "human", 50_000.0)
    monkeypatch.setattr(transitions, "_compute_hashes",
                        lambda name: ArtifactIdentity("c", "cfg", "d"))
    return repo, conn, sid


def _move(repo, target: Stage, **kw: Any):
    extra: dict[str, Any] = {}
    if target is Stage.LIVE:
        extra = {"approval_verifier": lambda *a: True,
                 "forward_certificate_verifier": lambda *a: {}}
    reason = "bench" if target is Stage.DORMANT else None
    return transition_strategy(repo, "s1", target, Actor.HUMAN, reason=reason, **extra, **kw)


@pytest.fixture
def selections(monkeypatch) -> list[tuple[Stage, Stage]]:
    seen: list[tuple[Stage, Stage]] = []
    real = lane_exit.select_exit_guard

    def spy(repo, name, source, target):
        seen.append((source, target))
        return real(repo, name, source, target)

    monkeypatch.setattr(lane_exit, "select_exit_guard", spy)
    return seen


@pytest.mark.parametrize(("source", "target"), [
    (Stage.PAPER, Stage.FORWARD_TESTED),  # human raw (Story 2.1 deletes this edge for everyone)
    (Stage.FORWARD_TESTED, Stage.PAPER),
])
def test_t14_edges_inside_the_paper_book_never_select_a_drain(
        tmp_path, monkeypatch, empty_exit_venues, selections, source, target):
    repo, conn, sid = _strategy_at(tmp_path, monkeypatch, source)
    injected: list[Any] = []

    rec = _move(repo, target, exit_guard_selector=lambda *a: injected.append(a))

    assert rec.stage is target
    assert injected == [] and selections == []
    assert allocations.active_allocation(conn, sid) is not None  # the slice stays in the book
    assert empty_exit_venues.builds == []


@pytest.mark.parametrize(("source", "target", "message"), [
    (Stage.PAPER, Stage.DORMANT, NEW),
    (Stage.FORWARD_TESTED, Stage.LIVE, NEW),
    (Stage.PAPER, Stage.CANDIDATE, RETIREMENT),
    (Stage.PAPER, Stage.RETIRED, RETIREMENT),
    (Stage.FORWARD_TESTED, Stage.RETIRED, RETIREMENT),
    (Stage.LIVE, Stage.RETIRED, RETIREMENT),
])
def test_t17_a_held_operator_lock_refuses_before_selection(
        tmp_path, monkeypatch, empty_exit_venues, selections, source, target, message):
    repo, conn, sid = _strategy_at(tmp_path, monkeypatch, source)

    with operator_run_lock(tmp_path / "operator.lock", job="paper", host=socket.gethostname(),
                           pid=os.getpid()):
        with pytest.raises(TransitionError) as exc:
            _move(repo, target)

    assert str(exc.value) == message
    assert selections == [] and empty_exit_venues.builds == []
    assert repo.get("s1").stage is source
    assert allocations.active_allocation(conn, sid) is not None


@pytest.mark.parametrize("target", [Stage.PAPER, Stage.DORMANT])
def test_t17_live_exits_outside_retirement_still_take_no_lock(
        tmp_path, monkeypatch, empty_exit_venues, selections, target):
    repo, _conn, _sid = _strategy_at(tmp_path, monkeypatch, Stage.LIVE)

    with operator_run_lock(tmp_path / "operator.lock", job="paper", host=socket.gethostname(),
                           pid=os.getpid()):
        rec = _move(repo, target)

    assert rec.stage is target
    assert selections == [(Stage.LIVE, target)]
