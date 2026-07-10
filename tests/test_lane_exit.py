"""LiveExitGuard: the broker-backed LIVE-lane drain a book-exit transition runs (#497 F2/H1)."""

from algua.execution.lane_exit import LiveExitGuard
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

    def __init__(self, open_orders: list[dict], activities: list[dict]) -> None:
        self._open_orders = open_orders
        self._activities = activities
        self.canceled: list[str] = []
        self.activities_after: list[str | None] = []

    def list_open_orders(self) -> list[dict]:
        return [o for o in self._open_orders if o["id"] not in self.canceled]

    def cancel_order(self, order_id: str) -> None:
        self.canceled.append(order_id)

    def account_activities(self, after: str | None = None) -> list[dict]:
        self.activities_after.append(after)
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
    )
    guard = LiveExitGuard(conn, broker, "s1")

    guard.cancel_and_ingest()

    # Only THIS strategy's order was canceled (never the sibling's).
    assert broker.canceled == ["boid-1"]
    # The activity feed was pulled from the current fill cursor and its fill was ingested.
    assert broker.activities_after == [None]
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

    # Before any cancel: our resting order shows as owned-open; the sibling's does not.
    assert guard.owned_open_order_ids() == ["boid-1"]
    # After canceling it (via the drain), nothing owned remains open.
    guard.cancel_and_ingest()
    assert guard.owned_open_order_ids() == []
