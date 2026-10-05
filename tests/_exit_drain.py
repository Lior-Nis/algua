"""Explicit exit-drain venues for tests (Story 2.2 §7, §8).

Every allocation-shedding transition now runs the source lane's real exit drain through
``lane_exit.select_exit_guard``. A test that drives such an edge opts in to ``empty_exit_venues``
(registered as a plugin by ``tests/conftest.py``; never autouse): it replaces both lanes' broker
builders with empty fakes, makes the live authorization read raise unless the test supplies one, and
redirects ``operator.lock`` into ``tmp_path``. The real guards still run against the fakes.

``FakePaperVenue`` is the scriptable paper venue the drain tests use: it implements
``PaperExitDrainBroker``, records every call, answers by-coid lookups with the order's ``id``,
``client_order_id``, ``symbol`` and ``filled_qty``, can fill an order at its cancel, keep a
cancelled order listed (``pending_cancel``), refuse a cancel, and withhold a fill activity until
released.
Its ``cancel_open_orders`` raises: the drain must never use the account-wide cancel.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from algua.contracts.types import LiveAuthorization
from algua.registry.live_gate import LiveAuthorizationError

BASE_TS = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass
class FakeOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str = "buy"
    qty: float = 2.0
    price: float = 100.0
    status: str = "accepted"
    filled_qty: float = 0.0
    cancelable: bool = True
    fill_on_cancel: bool = False      # executes in full at the DELETE (which then answers 422)
    withhold: bool = False            # that fill's activity stays out of the feed until release()
    listed_after_cancel: int = 0      # list calls a cancelled order survives (pending_cancel)


@dataclass
class FakePaperVenue:
    """A paper venue whose clock advances one second per read."""

    now: datetime = BASE_TS
    orders: dict[str, FakeOrder] = field(default_factory=dict)
    feed: list[dict[str, Any]] = field(default_factory=list)
    withheld: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    fail: dict[str, Exception] = field(default_factory=dict)  # method -> raised on every call
    fail_at: dict[str, dict[int, Exception]] = field(default_factory=dict)  # method -> {nth: exc}
    counts: dict[str, int] = field(default_factory=dict)
    answers: list[dict[str, Any] | None] = field(default_factory=list)  # by-coid answers, in order
    clock_value: str | None = None                             # e.g. a tz-naive timestamp
    by_coid_override: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    on_cancel: Callable[[str], None] | None = None

    def add_order(self, oid: str, coid: str, symbol: str = "AAPL", **kw: Any) -> FakeOrder:
        self.orders[oid] = FakeOrder(oid, coid, symbol, **kw)
        return self.orders[oid]

    def release(self) -> None:
        self.feed.extend(self.withheld)
        self.withheld.clear()

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def _maybe_fail(self, method: str) -> None:
        n = self.counts[method] = self.counts.get(method, 0) + 1
        if method in self.fail:
            raise self.fail[method]
        if n in self.fail_at.get(method, {}):
            raise self.fail_at[method][n]

    def clock(self) -> str:
        self.now += timedelta(seconds=1)
        self.calls.append(("clock", self.now.isoformat()))
        self._maybe_fail("clock")
        return self.clock_value or self.now.isoformat()

    def account_activities_window(self, after: str, until: str) -> list[dict[str, Any]]:
        self.calls.append(("activities", after, until))
        self._maybe_fail("account_activities_window")
        lo, hi = _parse(after), _parse(until)
        return sorted((a for a in self.feed if lo < _parse(a["transaction_time"]) <= hi),
                      key=lambda a: a["transaction_time"])

    def list_open_orders(self) -> list[dict[str, Any]]:
        self.calls.append(("list_open_orders",))
        self._maybe_fail("list_open_orders")
        out = []
        for o in self.orders.values():
            if o.status == "pending_cancel":
                if o.listed_after_cancel <= 0:
                    o.status = "canceled"
                    continue
                o.listed_after_cancel -= 1
            elif o.status != "accepted":
                continue
            out.append({"id": o.id, "client_order_id": o.client_order_id, "symbol": o.symbol,
                        "status": o.status})
        return out

    def cancel_order(self, order_id: str) -> None:
        self.calls.append(("cancel", order_id))
        if self.on_cancel is not None:
            self.on_cancel(order_id)
        self._maybe_fail("cancel_order")
        o = self.orders.get(order_id)
        if o is None or o.status != "accepted" or not o.cancelable:
            return  # 404/422: gone, terminal, or not cancelable -- a no-op at the broker
        if o.fill_on_cancel:
            self.execute(o)
        else:
            o.status = "pending_cancel" if o.listed_after_cancel > 0 else "canceled"

    def execute(self, o: FakeOrder) -> None:
        """Fill ``o`` in full now; publish its activity unless the order withholds it."""
        o.status, o.filled_qty = "filled", o.qty
        act = {"id": f"act-{o.id}", "activity_type": "FILL", "side": o.side, "qty": str(o.qty),
               "price": str(o.price), "symbol": o.symbol, "order_id": o.id,
               "transaction_time": self.now.isoformat()}
        (self.withheld if o.withhold else self.feed).append(act)

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        self.calls.append(("by_coid", client_order_id))
        self._maybe_fail("get_order_by_client_order_id")
        answer: dict[str, Any] | None = None
        if client_order_id in self.by_coid_override:
            answer = self.by_coid_override[client_order_id]
        else:
            for o in self.orders.values():
                if o.client_order_id == client_order_id:
                    answer = {"id": o.id, "client_order_id": o.client_order_id,
                              "symbol": o.symbol, "filled_qty": f"{o.filled_qty:g}",
                              "status": o.status}
        self.answers.append(answer)
        return answer

    def cancel_open_orders(self) -> None:
        raise AssertionError("the exit drain must never call the account-wide cancel")


@dataclass
class EmptyLiveVenue:
    """An empty live venue for the live drain: no open orders, no activities."""

    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def list_open_orders(self) -> list[dict[str, Any]]:
        self.calls.append(("list_open_orders",))
        return []

    def cancel_order(self, order_id: str) -> None:
        self.calls.append(("cancel", order_id))

    def account_activities(self, after: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("activities", after))
        return []


@dataclass
class ExitVenues:
    """What ``empty_exit_venues`` installed. Set ``authorization`` to let the live branch build its
    authorized broker; leave it ``None`` and the live branch takes the account-credential drain."""

    paper: FakePaperVenue = field(default_factory=FakePaperVenue)
    live: EmptyLiveVenue = field(default_factory=EmptyLiveVenue)
    authorization: LiveAuthorization | None = None
    builds: list[str] = field(default_factory=list)


@pytest.fixture
def empty_exit_venues(monkeypatch, tmp_path) -> ExitVenues:
    from algua.execution import lane_exit, paper_exit_drain
    from algua.registry import live_gate

    venues = ExitVenues()

    def paper() -> FakePaperVenue:
        venues.builds.append("paper")
        return venues.paper

    def live(authorization: LiveAuthorization) -> EmptyLiveVenue:
        venues.builds.append("live")
        return venues.live

    def live_drain() -> EmptyLiveVenue:
        venues.builds.append("live_drain")
        return venues.live

    def verify(conn, repo, name, allowed_signers_path) -> LiveAuthorization:
        if venues.authorization is None:
            raise LiveAuthorizationError(f"{name}: no live authorization in this test")
        return venues.authorization

    monkeypatch.setattr(paper_exit_drain, "build_paper_drain_broker", paper)
    monkeypatch.setattr(lane_exit, "build_live_broker", live)
    monkeypatch.setattr(lane_exit, "build_live_drain_broker", live_drain)
    monkeypatch.setattr(live_gate, "verify_live_authorization", verify)
    monkeypatch.setattr("algua.operator.deployment_lock._operator_lock_path",
                        lambda: tmp_path / "operator.lock")
    return venues
