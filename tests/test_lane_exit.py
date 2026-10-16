"""LiveExitGuard: the broker-backed LIVE-lane drain a book-exit transition runs (#497 F2/H1)."""

import pytest

from algua.contracts.lifecycle import TransitionError
from algua.execution.alpaca_broker import AlpacaLiveDrainBroker
from algua.execution.errors import BrokerError
from algua.execution.lane_exit import LiveExitGuard, build_live_drain_broker
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_broker_order_id,
    fill_cursor,
    record_live_order,
)
from algua.registry.db import connect, migrate


class _FakeBroker:
    """A minimal live broker: canned open orders, a recorded cancel log, and a canned activity feed
    (the buy fill for the resting order, returned once the cancel has run)."""

    def __init__(self, open_orders: list[dict], activities: list[dict], *,
                 publish_after_cancel: bool = False,
                 uncancellable: frozenset[str] = frozenset()) -> None:
        self._open_orders = open_orders
        self._activities = activities
        self._publish_after_cancel = publish_after_cancel
        self._uncancellable = uncancellable
        self.canceled: list[str] = []
        self.activities_after: list[str | None] = []
        self.log: list[str] = []

    def list_open_orders(self) -> list[dict]:
        return [o for o in self._open_orders
                if o["id"] not in self.canceled or o["id"] in self._uncancellable]

    def cancel_order(self, order_id: str) -> None:
        self.log.append(f"cancel:{order_id}")
        self.canceled.append(order_id)

    def account_activities(self, after: str | None = None) -> list[dict]:
        self.log.append("ingest")
        self.activities_after.append(after)
        if self._publish_after_cancel and not self.canceled:
            return []
        return self._activities


def _conn(tmp_path):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    return conn


def test_cancel_and_ingest_cancels_owned_then_ingests(tmp_path):
    conn = _conn(tmp_path)
    # This strategy owns one resting order (coid maps to it, broker id backfilled on accept).
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    broker = _FakeBroker(
        open_orders=[{"id": "boid-1", "client_order_id": "coid-1"},
                     {"id": "boid-sib", "client_order_id": "coid-other"}],
        activities=[{"id": "a1", "activity_type": "FILL", "side": "buy", "qty": "3",
                     "price": "100", "symbol": "AAPL", "order_id": "boid-1",
                     "transaction_time": "2026-01-01T00:00:00Z"}],
        publish_after_cancel=True,  # the fill that raced the cancel appears only after it
    )
    guard = LiveExitGuard(conn, broker, "s1")

    guard.cancel_and_ingest()

    # Only THIS strategy's order was canceled (never the sibling's).
    assert broker.canceled == ["boid-1"]
    # Ingest, cancel, ingest AGAIN (#690): the second pass picked up the raced fill.
    assert broker.log == ["ingest", "cancel:boid-1", "ingest"]
    assert broker.activities_after == [None, None]
    rows = conn.execute("SELECT strategy, symbol, qty FROM live_fills").fetchall()
    assert [(r["strategy"], r["symbol"], float(r["qty"])) for r in rows] == [("s1", "AAPL", 3.0)]
    # The cursor advanced past the ingested activity.
    assert fill_cursor(conn, LedgerKind.LIVE) == "a1"


def test_owned_open_order_ids_scoped_to_strategy(tmp_path):
    conn = _conn(tmp_path)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    broker = _FakeBroker(
        open_orders=[{"id": "boid-1", "client_order_id": "coid-1"},
                     {"id": "boid-sib", "client_order_id": "coid-other"}],
        activities=[],
    )
    guard = LiveExitGuard(conn, broker, "s1")

    # After the drain cancels it, nothing owned remains open (the sibling's never counted).
    guard.cancel_and_ingest()
    assert guard.owned_open_order_ids() == []


def test_a_held_position_skips_the_cancel_and_refuses_under_the_lock(tmp_path):
    """#690: a materially-not-flat strategy keeps its resting liquidation orders."""
    conn = _conn(tmp_path)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    record_live_order(conn, "s1", "AAPL", "sell", None, "coid-off")
    backfill_broker_order_id(conn, "coid-off", "boid-off")
    broker = _FakeBroker(
        open_orders=[{"id": "boid-off", "client_order_id": "coid-off"}],
        activities=[{"id": "a1", "activity_type": "FILL", "side": "buy", "qty": "3",
                     "price": "100", "symbol": "AAPL", "order_id": "boid-1",
                     "transaction_time": "2026-01-01T00:00:00Z"}],
    )
    guard = LiveExitGuard(conn, broker, "s1")
    guard.cancel_and_ingest()
    assert broker.canceled == []  # the offset order survived
    with pytest.raises(TransitionError, match="not flat"):
        guard.owned_open_order_ids()


def test_an_order_the_cancel_could_not_remove_refuses_under_the_lock(tmp_path):
    conn = _conn(tmp_path)
    record_live_order(conn, "s1", "AAPL", "buy", None, "coid-1")
    backfill_broker_order_id(conn, "coid-1", "boid-1")
    broker = _FakeBroker(open_orders=[{"id": "boid-1", "client_order_id": "coid-1"}],
                         activities=[], uncancellable=frozenset({"boid-1"}))
    guard = LiveExitGuard(conn, broker, "s1")
    guard.cancel_and_ingest()
    assert guard.owned_open_order_ids() == ["boid-1"]


def test_the_recheck_cannot_run_before_the_drain(tmp_path):
    guard = LiveExitGuard(_conn(tmp_path), _FakeBroker([], []), "s1")
    with pytest.raises(RuntimeError, match="before its drain"):
        guard.owned_open_order_ids()


# --- account-credential drain broker (#497 H1): drain a de-authorized strategy's resting orders ---

def test_build_live_drain_broker_none_without_credentials(monkeypatch):
    # No live credentials configured -> None, so the caller can FAIL CLOSED (never fall open to a
    # positions-only check that ignores resting OPEN orders).
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_API_SECRET", raising=False)
    assert build_live_drain_broker() is None


def test_build_live_drain_broker_built_from_account_credentials(monkeypatch):
    # With account-level credentials present, the drain broker is built WITHOUT any per-strategy
    # go-live authorization (mirroring `_live_account_equity`'s raw-credential account read).
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    broker = build_live_drain_broker()
    assert isinstance(broker, AlpacaLiveDrainBroker)
    assert broker.api_key == "lk"


def test_drain_broker_refuses_post():
    # A drain cancels and reads only — it must NEVER submit a new order. Every submit path POSTs, so
    # refusing _post proves the cancel-only invariant.
    broker = AlpacaLiveDrainBroker("lk", "ls")
    with pytest.raises(BrokerError, match="cancel-only"):
        broker._post("/v2/orders", {"symbol": "AAPL"})
