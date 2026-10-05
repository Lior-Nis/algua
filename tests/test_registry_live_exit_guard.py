"""`lane_exit.select_exit_guard`, live branch (moved verbatim from `registry_cmd`; Story 2.2 T15).

On a revoked/absent per-strategy authorization a `live -> exit` transition must DRAIN the strategy's
resting orders via ACCOUNT-LEVEL credentials — never fall open to a positions-only check that
ignores an OPEN resting order (#497 H1 / the #451 orphan class). Only when EVEN the account
credentials are unavailable does the exit FAIL CLOSED. Messages and audit actions are byte-identical
to the CLI-side selection they replace; only their timing moved (inside the lock, after validation).
"""

import json
import os
import socket
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.audit import log as audit_log
from algua.cli.main import app
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution import lane_exit
from algua.execution.lane_exit import LiveExitGuard
from algua.execution.live_ledger import backfill_broker_order_id, record_live_order
from algua.operator.schedule import operator_run_lock
from algua.registry import allocations, live_gate
from algua.registry.db import connect, migrate
from algua.registry.live_gate import LiveAuthorizationError
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

UNAVAILABLE = (
    "cannot exit live strategy 's1': its per-strategy authorization is unavailable (revoked) and "
    "no live credentials are configured to drain resting orders at the account level; run `algua "
    "live flatten` to clear resting orders, then retry")


def _live_repo(tmp_path) -> tuple[SqliteStrategyRepository, sqlite3.Connection, int]:
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    sid = repo.add(name="s1").id
    with conn:
        conn.execute("UPDATE strategies SET stage='live' WHERE id=?", (sid,))
    return repo, conn, sid


def _seed_alloc(conn: sqlite3.Connection, sid: int) -> None:
    with conn:
        allocations.allocate_locked(conn, sid, 10_000.0, "human", 50_000.0)


class _FakeDrainBroker:
    """A minimal account-credential drain broker: canned open orders + a cancel log + a canned
    (here empty) activity feed. Mirrors the surface `LiveExitGuard` drives."""

    def __init__(self, open_orders: list[dict], activities: list[dict] | None = None) -> None:
        self._open_orders = open_orders
        self._activities = activities or []
        self.canceled: list[str] = []

    def list_open_orders(self) -> list[dict]:
        return [o for o in self._open_orders if o["id"] not in self.canceled]

    def cancel_order(self, order_id: str) -> None:
        self.canceled.append(order_id)

    def account_activities(self, after=None) -> list[dict]:
        return self._activities


def _auth():
    from algua.contracts.types import LiveAuthorization
    return LiveAuthorization(1, "c", "cf", "d", "lior", "t")


def _raise_revoked(*_a, **_k):
    raise LiveAuthorizationError("revoked")


def _actions(conn) -> list[tuple[str, str]]:
    return [(r["action"], r["reason"]) for r in audit_log.read(conn, strategy="s1")]


def test_revoked_auth_drains_resting_order_via_account_path(tmp_path, monkeypatch):
    # Per-strategy authorization is revoked AND the strategy has a resting open order. The exit
    # must cancel that order via the account-credential drain BEFORE the allocation is shed.
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    drain = _FakeDrainBroker(open_orders=[{"id": "boid-1", "client_order_id": "coid-1"}])

    monkeypatch.setattr(live_gate, "verify_live_authorization", _raise_revoked)
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", lambda: drain)

    guard = lane_exit.select_exit_guard(repo, "s1", Stage.LIVE, Stage.DORMANT)
    assert isinstance(guard, LiveExitGuard)
    # The account-creds drain path was chosen and audited (not a silent skip), byte for byte.
    assert _actions(conn) == [(
        "live_exit_drain_account_creds",
        "per-strategy authorization unavailable (revoked); draining resting orders via "
        "account-level live credentials")]

    # Drive the transition with the guard: the resting order is cancelled before the exit commits.
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.HUMAN, reason="bench",
                              exit_guard_selector=lambda *_a: guard)
    assert drain.canceled == ["boid-1"]  # THIS strategy's resting order was cancelled
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, sid) is None


def test_no_account_credentials_fails_closed(tmp_path, monkeypatch):
    # Authorization revoked AND no account credentials to drain with -> FAIL CLOSED, never fall open
    # to a positions-only check that ignores the resting order.
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)

    monkeypatch.setattr(live_gate, "verify_live_authorization", _raise_revoked)
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", lambda: None)

    with pytest.raises(TransitionError) as exc:
        lane_exit.select_exit_guard(repo, "s1", Stage.LIVE, Stage.DORMANT)

    assert str(exc.value) == UNAVAILABLE
    assert _actions(conn) == [(
        "live_exit_drain_unavailable",
        "live authorization unavailable (revoked) AND live credentials not configured; cannot "
        "drain resting orders")]
    # Nothing moved: the guard raised before any transition.
    assert repo.get("s1").stage is Stage.LIVE
    assert allocations.active_allocation(conn, sid) is not None


def test_authorized_path_builds_guard_from_authorized_broker(tmp_path, monkeypatch):
    # The happy authorized path: a verified authorization builds the drain from the authorized
    # trading broker (the account-creds fallback is never reached). The allowed-signers path is
    # read through `live_gate` at call time, so one attribute re-targets it.
    repo, conn, sid = _live_repo(tmp_path)
    sentinel = object()
    signers = tmp_path / "allowed_signers"
    seen = []

    def _fallback_forbidden():
        raise AssertionError("account-creds fallback used on the happy authorized path")

    def _verify(_conn, _repo, _name, path):
        seen.append(path)
        return _auth()

    monkeypatch.setattr(live_gate, "ALLOWED_SIGNERS_PATH", signers)
    monkeypatch.setattr(live_gate, "verify_live_authorization", _verify)
    monkeypatch.setattr(lane_exit, "build_live_broker", lambda authorization: sentinel)
    # If the fallback were taken this would blow up; it must not be called.
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", _fallback_forbidden)

    guard = lane_exit.select_exit_guard(repo, "s1", Stage.LIVE, Stage.DORMANT)
    assert isinstance(guard, LiveExitGuard)
    assert guard._broker is sentinel
    assert seen == [signers]
    assert _actions(conn) == []


def test_a_paper_lane_source_never_reads_the_live_authorization(tmp_path, monkeypatch,
                                                                empty_exit_venues):
    repo, conn, sid = _live_repo(tmp_path)
    with conn:
        conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (sid,))

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("verify_live_authorization must not be called for a paper source")

    monkeypatch.setattr(live_gate, "verify_live_authorization", _must_not_be_called)
    guard = lane_exit.select_exit_guard(repo, "s1", Stage.PAPER, Stage.DORMANT)
    assert not isinstance(guard, LiveExitGuard)
    assert "live" not in empty_exit_venues.builds


def test_a_source_outside_both_lanes_has_no_drain(tmp_path):
    repo, _conn, _sid = _live_repo(tmp_path)
    with pytest.raises(ValueError, match="no exit drain for dormant -> retired"):
        lane_exit.select_exit_guard(repo, "s1", Stage.DORMANT, Stage.RETIRED)


def test_a_cli_live_exit_still_drains(tmp_path, monkeypatch):
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    drain = _FakeDrainBroker(open_orders=[{"id": "boid-1", "client_order_id": "coid-1"}])
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "reg.db"))
    monkeypatch.setattr(live_gate, "verify_live_authorization", _raise_revoked)
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", lambda: drain)

    result = CliRunner().invoke(app, ["registry", "transition", "s1", "--to", "dormant",
                                      "--actor", "human", "--reason", "bench"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == {"ok": True, "name": "s1", "stage": "dormant"}
    assert drain.canceled == ["boid-1"]
    assert allocations.active_allocation(conn, sid) is None


@pytest.fixture
def no_live_build(monkeypatch) -> list[str]:
    built: list[str] = []
    monkeypatch.setattr(live_gate, "verify_live_authorization",
                        lambda *a: built.append("verify") or _raise_revoked())
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", lambda: built.append("drain"))
    monkeypatch.setattr(lane_exit, "build_live_broker", lambda a: built.append("live"))
    return built


def test_timing_pin_a_reasonless_live_bench_builds_no_broker(tmp_path, no_live_build):
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)

    with pytest.raises(TransitionError) as exc:
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.HUMAN)

    assert str(exc.value) == "transition to dormant requires a non-empty reason"
    assert no_live_build == []
    assert not [a for a, _ in _actions(conn) if a.startswith("live_exit_drain_")]


def test_timing_pin_a_live_retirement_under_a_held_lock_builds_no_broker(
        tmp_path, monkeypatch, no_live_build):
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)
    lock = tmp_path / "operator.lock"
    monkeypatch.setattr("algua.operator.deployment_lock._operator_lock_path", lambda: lock)

    with operator_run_lock(lock, job="paper", host=socket.gethostname(), pid=os.getpid()):
        with pytest.raises(TransitionError) as exc:
            transition_strategy(repo, "s1", Stage.RETIRED, Actor.HUMAN)

    assert str(exc.value) == (
        "operator.lock is held; deployment retirement cannot interleave with a paper tick")
    assert no_live_build == []
    assert not [a for a, _ in _actions(conn) if a.startswith("live_exit_drain_")]
    assert repo.get("s1").stage is Stage.LIVE
