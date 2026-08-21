"""Broker-backed source-lane drains for book-exit transitions (#497 F2/H1).

When a strategy leaves its operating book (``live -> dormant/paper/retired``), a resting order left
behind at the venue can fill AFTER the exit and orphan a position the source lane's ``run-all`` no
longer iterates. ``LiveExitGuard`` gives the registry transition the same cancel -> ingest ->
recheck ceremony ``live flatten`` uses, injected as an ``ExitLaneGuard`` so the registry layer never
imports a broker or the execution ledger directly (store.py stays behind the data wall)."""
from __future__ import annotations

import sqlite3

from algua.contracts.types import ExitDrainBroker, LiveAuthorization
from algua.execution.alpaca_broker import AlpacaLiveBroker, AlpacaLiveDrainBroker
from algua.execution.broker_factory import BrokerKind, build_broker, maybe_broker
from algua.execution.live_ledger import (
    LedgerKind,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
)


def build_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker:
    """Construct the Alpaca LIVE broker from the settings-configured credentials, bound to a
    verified ``LiveAuthorization``. Single-sourced so ``live_cmd`` and the book-exit drain agree on
    how the real-money broker is built (no drift, no dual path)."""
    return build_broker(BrokerKind.ALPACA_LIVE, authorization)


def build_live_drain_broker() -> AlpacaLiveDrainBroker | None:
    """Construct the CANCEL-ONLY account-credential LIVE drain broker (#497 H1), used to cancel a
    strategy's resting orders on a book-exit when its per-strategy go-live authorization is
    revoked/absent. Built from the SAME settings credentials as ``build_live_broker`` but WITHOUT a
    ``LiveAuthorization`` — the authorization is only a construction tollbooth on the trading broker
    and is never used for REST, exactly like ``live_cmd._live_account_equity``'s raw-credential
    account read.

    Returns ``None`` when the live credentials are not configured, so the caller can FAIL CLOSED
    (block the exit) rather than fall open to a positions-only check that ignores resting orders."""
    return maybe_broker(BrokerKind.ALPACA_LIVE_DRAIN)


class LiveExitGuard:
    """The LIVE-lane ``ExitLaneGuard`` a book-exit transition runs so a resting live order cannot
    outlive the strategy's departure from the live book (#497 F2/H1).

    ``cancel_and_ingest`` runs BEFORE the registry write lock (broker network calls + committing
    ingests): cancel THIS strategy's own open live orders (scoped — never a sibling's), then ingest
    the account activity feed so any just-filled order is reflected in the live ledger the
    under-lock flatness check reads. ``owned_open_order_ids`` runs UNDER the lock to re-list any
    order the cancel failed to remove (a non-cancelable/partial state), so it blocks the revoke+CAS
    rather than orphaning a position."""

    def __init__(
        self, conn: sqlite3.Connection, broker: ExitDrainBroker, strategy: str
    ) -> None:
        self._conn = conn
        self._broker = broker
        self._strategy = strategy

    def cancel_and_ingest(self) -> None:
        for oid in owned_open_order_ids(
            self._conn, self._broker, self._strategy, kind=LedgerKind.LIVE
        ):
            self._broker.cancel_order(oid)
        cursor = fill_cursor(self._conn, LedgerKind.LIVE)
        ingest_activities(
            self._conn, self._broker.account_activities(after=cursor), LedgerKind.LIVE)

    def owned_open_order_ids(self) -> list[str]:
        return owned_open_order_ids(
            self._conn, self._broker, self._strategy, kind=LedgerKind.LIVE)
