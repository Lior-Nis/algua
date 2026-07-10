"""`registry_cmd._live_exit_guard`: on a revoked/absent per-strategy authorization a `live -> exit`
transition must DRAIN the strategy's resting orders via ACCOUNT-LEVEL credentials — never fall open
to a positions-only check that ignores an OPEN resting order (#497 H1 / the #451 orphan class).
Only when EVEN the account credentials are unavailable does the exit FAIL CLOSED."""

import sqlite3

import pytest

from algua.audit import log as audit_log
from algua.cli import registry_cmd
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution.lane_exit import LiveExitGuard
from algua.execution.live_ledger import backfill_broker_order_id, record_live_order
from algua.registry import allocations, live_gate
from algua.registry.db import connect, migrate
from algua.registry.live_gate import LiveAuthorizationError
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


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


def test_revoked_auth_drains_resting_order_via_account_path(tmp_path, monkeypatch):
    # Per-strategy authorization is revoked AND the strategy has a resting open order. The exit
    # must cancel that order via the account-credential drain BEFORE the allocation is shed.
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    drain = _FakeDrainBroker(open_orders=[{"id": "boid-1", "client_order_id": "coid-1"}])

    monkeypatch.setattr(live_gate, "verify_live_authorization", _raise_revoked)
    monkeypatch.setattr(registry_cmd, "build_live_drain_broker", lambda: drain)

    guard = registry_cmd._live_exit_guard(conn, repo, "s1", Stage.DORMANT)
    assert isinstance(guard, LiveExitGuard)
    # The account-creds drain path was chosen and audited (not a silent skip).
    actions = [r["action"] for r in audit_log.read(conn, strategy="s1")]
    assert "live_exit_drain_account_creds" in actions

    # Drive the transition with the guard: the resting order is cancelled before the exit commits.
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.HUMAN, reason="bench",
                              exit_guard=guard)
    assert drain.canceled == ["boid-1"]  # THIS strategy's resting order was cancelled
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, sid) is None


def test_no_account_credentials_fails_closed(tmp_path, monkeypatch):
    # Authorization revoked AND no account credentials to drain with -> FAIL CLOSED, never fall open
    # to a positions-only check that ignores the resting order.
    repo, conn, sid = _live_repo(tmp_path)
    _seed_alloc(conn, sid)

    monkeypatch.setattr(live_gate, "verify_live_authorization", _raise_revoked)
    monkeypatch.setattr(registry_cmd, "build_live_drain_broker", lambda: None)

    with pytest.raises(TransitionError, match="live flatten"):
        registry_cmd._live_exit_guard(conn, repo, "s1", Stage.DORMANT)

    actions = [r["action"] for r in audit_log.read(conn, strategy="s1")]
    assert "live_exit_drain_unavailable" in actions
    # Nothing moved: the guard raised before any transition.
    assert repo.get("s1").stage is Stage.LIVE
    assert allocations.active_allocation(conn, sid) is not None


def test_authorized_path_builds_guard_from_authorized_broker(tmp_path, monkeypatch):
    # The happy authorized path: a verified authorization builds the drain from the authorized
    # trading broker (the account-creds fallback is never reached).
    repo, conn, sid = _live_repo(tmp_path)
    sentinel = object()

    def _fallback_forbidden():
        raise AssertionError("account-creds fallback used on the happy authorized path")

    monkeypatch.setattr(live_gate, "verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr(registry_cmd, "build_live_broker", lambda authorization: sentinel)
    # If the fallback were taken this would blow up; it must not be called.
    monkeypatch.setattr(registry_cmd, "build_live_drain_broker", _fallback_forbidden)

    guard = registry_cmd._live_exit_guard(conn, repo, "s1", Stage.DORMANT)
    assert isinstance(guard, LiveExitGuard)
    assert guard._broker is sentinel


def test_non_live_source_returns_none(tmp_path, monkeypatch):
    # The guard only applies to a LIVE-source book-exit; a non-live source is a no-op (None), with
    # no authorization check attempted.
    repo, conn, sid = _live_repo(tmp_path)
    with conn:
        conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (sid,))

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("verify_live_authorization must not be called for a non-live source")

    monkeypatch.setattr(live_gate, "verify_live_authorization", _must_not_be_called)
    assert registry_cmd._live_exit_guard(conn, repo, "s1", Stage.DORMANT) is None
