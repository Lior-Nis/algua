"""The paper-lane exit drain (Story 2.2, #685).

A strategy that leaves the paper book (``paper -> candidate | dormant | retired``,
``forward_tested -> retired | live``) must not leave a resting paper order behind: if it fills after
the exit, the paper account reconcile sees a material belief on a strategy that is no longer on the
lane, defers the cycle and, after the grace window, halts the whole account. ``PaperExitGuard``
is the ``ExitLaneGuard`` the registry runs for those exits (``lane_exit.select_exit_guard``):

- ``cancel_and_ingest`` runs before the registry write lock. It syncs the venue, skips the cancel
  while the strategy holds a material position (resting liquidation offsets must survive a premature
  exit), cancels only the strategy's own open orders (each cancel request audited before its
  DELETE), re-lists them briefly while a cancel is pending, syncs again, and settles every order any
  drain ever asked the venue to cancel for the strategy against the ledger's fills.
- ``owned_open_order_ids`` runs under the lock. It refuses an exit that did not complete the drain,
  or whose venue fills are not in the ledger yet, and otherwise re-lists the strategy's open orders.

The drain never advances the shared paper fill cursor and never calls the account-wide cancel. This
module does not import the registry; the registry reaches it only through ``transitions.py``'s lazy
import of ``lane_exit``.
"""
from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from algua.audit import log as audit_log
from algua.contracts.lifecycle import TransitionError
from algua.contracts.types import ActivityWindowBroker, OrderLookupBroker, ScopedCancelBroker
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.execution.broker_factory import BrokerKind, maybe_broker
from algua.execution.dust import DUST_SHARES, paper_dust
from algua.execution.errors import BrokerError
from algua.execution.live_ledger import LedgerKind, believed_positions, owned_open_order_ids
from algua.execution.tick_clock import tick_clock
from algua.execution.venue_sync import ingest_paper_venue_keep_cursor, recover_stranded

#: One audit row per order the drain asks the venue to cancel, committed before its DELETE. Its
#: reason is exactly the broker order id: the settle step reads these rows back as its durable set.
CANCEL_REQUESTED = "paper_exit_drain_cancel_requested"
#: Further re-lists while a cancelled order is still listed (Alpaca's ``status=open`` includes
#: ``pending_cancel``), and the pause before each.
RELISTS = 3
RELIST_INTERVAL_S = 1.0


@runtime_checkable
class PaperExitDrainBroker(ScopedCancelBroker, ActivityWindowBroker, OrderLookupBroker, Protocol):
    """What the paper drain needs from the venue: list and cancel by id, the exhaustive activity
    window, the per-coid order lookup, and the venue clock. ``AlpacaPaperBroker`` satisfies it; the
    account-wide ``cancel_open_orders`` is deliberately outside it."""

    def clock(self) -> str: ...


def build_paper_drain_broker() -> AlpacaPaperBroker | None:
    """The paper venue built from the settings credentials, or ``None`` when they are absent (the
    selector then refuses the exit). A malformed setting propagates, as on every paper command."""
    return maybe_broker(BrokerKind.ALPACA_PAPER)


class PaperExitGuard:
    """The paper-lane ``ExitLaneGuard`` (Story 2.2 §3.3-§3.4); ``sleep`` is the re-list seam."""

    def __init__(
        self, conn: sqlite3.Connection, broker: PaperExitDrainBroker, name: str, *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._conn = conn
        self._broker = broker
        self._name = name
        self._sleep = sleep
        self._step = "clock"
        self.state: Literal["new", "skipped", "drained"] = "new"
        self.unsettled: list[str] = []
        self.unpublished: set[str] = set()

    def cancel_and_ingest(self) -> None:
        """Drain before the write lock. A step failure other than ``sqlite3.Error`` is audited as
        ``paper_exit_drain_failed`` and raised as an actionable ``BrokerError``; a database fault
        propagates unaudited, because the audit write would meet the same fault."""
        try:
            self._drain()
        except sqlite3.Error:
            raise
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            audit_log.append(self._conn, actor="system", action="paper_exit_drain_failed",
                             reason=f"{self._step}: {detail}", strategy=self._name)
            raise BrokerError(
                f"cannot exit paper-lane strategy {self._name!r}: its exit drain failed at "
                f"{self._step} ({detail}); its stage and allocation are unchanged; retry when the "
                "paper venue is reachable") from exc

    def owned_open_order_ids(self) -> list[str]:
        """The under-lock re-check. Never writes and never commits: an audit append would commit
        the exit transaction half-way, so an under-lock failure is reported only by the envelope."""
        if self.state == "new":
            raise RuntimeError("paper exit re-check ran before its drain")
        if self.state == "skipped":
            raise TransitionError(
                f"{self._name} is not flat (open paper positions when the exit drain ran); "
                "flatten before this transition")
        if self.unpublished:
            raise TransitionError(
                f"{self._name} is not flat (the paper venue reports fills on order(s) "
                f"{sorted(self.unpublished)} that the paper ledger does not hold yet); retry this "
                "transition once the venue publishes them")
        if self.unsettled:
            return sorted(self.unsettled)
        try:
            return sorted(self._open_orders())
        except sqlite3.Error:
            raise
        except Exception as exc:
            raise BrokerError(
                f"cannot exit paper-lane strategy {self._name!r}: re-listing its open paper orders "
                f"under the registry lock failed ({type(exc).__name__}: {exc}); the transition was "
                "rolled back; retry") from exc

    def _drain(self) -> None:
        self._sync()
        held = believed_positions(self._conn, self._name, LedgerKind.PAPER)
        if any(not paper_dust(self._conn, s, q) for s, q in held.items()):
            self.state = "skipped"  # the store refuses on positions; resting offsets must survive
            return
        self._step = "open_orders"
        owned = self._open_orders()
        if owned:
            for oid in owned:
                self._step = "cancel"
                audit_log.append(self._conn, actor="system", action=CANCEL_REQUESTED, reason=oid,
                                 strategy=self._name)
                self._broker.cancel_order(oid)
            self._step = "recheck"
            unsettled = self._open_orders()
            relists = 0
            while set(unsettled) & set(owned) and relists < RELISTS:
                self._sleep(RELIST_INTERVAL_S)
                unsettled = self._open_orders()
                relists += 1
            self.unsettled = unsettled
        self._sync()  # its `until` is read after the last observation of the strategy's orders
        self._step = "settle"
        self._settle()
        self.state = "drained"

    def _sync(self) -> None:
        """The paper operator's ingest pairing, with the cursor-keeping ingest. Broker clock
        only: a local ``until`` behind the venue's would end the window before its fills."""
        self._step = "clock"
        until, source = tick_clock(self._broker.clock)
        if source != "broker":
            raise BrokerError("the paper venue clock is unusable; the exit drain refuses the "
                              "local-clock fallback")
        self._step = "ingest"
        ingest_paper_venue_keep_cursor(self._conn, self._broker, until)
        recover_stranded(self._conn, self._broker, LedgerKind.PAPER)

    def _open_orders(self) -> list[str]:
        return owned_open_order_ids(self._conn, self._broker, self._name, kind=LedgerKind.PAPER)

    def _settle(self) -> None:
        """Every order any drain asked the venue to cancel for this strategy, and that was not open
        at the last observation, is closed at the venue with a final ``filled_qty``; the ledger must
        hold it. A gap the venue has not published yet refuses under the lock, on every retry."""
        requested = {r["reason"] for r in audit_log.read(
            self._conn, strategy=self._name, actor="system", action=CANCEL_REQUESTED)}
        for oid in sorted(requested - set(self.unsettled)):
            row = self._conn.execute(
                "SELECT client_order_id FROM paper_venue_orders"
                " WHERE broker_order_id = ? AND strategy = ?", (oid, self._name)).fetchone()
            if row is None:
                raise ValueError(f"no paper order row maps broker order {oid!r} to {self._name!r}")
            coid = row[0]
            order = self._broker.get_order_by_client_order_id(coid)
            if order is None:
                raise ValueError(f"the paper venue has no order with client_order_id {coid!r}")
            if str(order.get("id")) != oid:
                raise ValueError(f"the paper venue's order for client_order_id {coid!r} has id "
                                 f"{order.get('id')!r}, expected {oid!r}")
            venue = _filled_qty(order, oid)
            held = abs(float(self._conn.execute(
                "SELECT COALESCE(SUM(qty), 0.0) FROM paper_venue_fills WHERE broker_order_id = ?",
                (oid,)).fetchone()[0]))
            if venue - held > DUST_SHARES:
                self.unpublished.add(oid)
            elif held - venue > DUST_SHARES:
                raise ValueError(f"order {oid!r}: the paper ledger holds {held} shares but the "
                                 f"venue reports {venue} filled")


def _filled_qty(order: dict[str, Any], oid: str) -> float:
    try:
        venue = float(order["filled_qty"])
    except (KeyError, TypeError, ValueError):
        venue = math.nan
    if not math.isfinite(venue) or venue < 0:
        raise ValueError(f"order {oid!r}: bad filled_qty {order.get('filled_qty')!r}")
    return venue
