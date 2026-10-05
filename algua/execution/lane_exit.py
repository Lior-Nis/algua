"""Broker-backed source-lane drains for book-exit transitions (#497 F2/H1, Story 2.2).

When a strategy leaves its operating book, a resting order left behind at the venue can fill AFTER
the exit and orphan a position the source lane's ``run-all`` no longer iterates. ``LiveExitGuard``
gives a ``live -> dormant/paper/retired`` transition the same cancel -> ingest -> recheck ceremony
``live flatten`` uses; ``paper_exit_drain.PaperExitGuard`` does the same for the paper lane.
``select_exit_guard`` is the one selection for both lanes: ``transition_strategy`` reaches it
through a lazy import, so the registry layer never imports a broker or the execution ledger
directly (store.py stays behind the data wall) and no caller can forget the drain."""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from algua.audit.log import append as audit_append
from algua.contracts.lifecycle import Stage, TransitionError
from algua.contracts.types import ExitDrainBroker, ExitLaneGuard, LiveAuthorization
from algua.execution import paper_exit_drain
from algua.execution.alpaca_broker import AlpacaLiveBroker, AlpacaLiveDrainBroker
from algua.execution.broker_factory import BrokerKind, build_broker, maybe_broker
from algua.execution.live_ledger import (
    LedgerKind,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
)
from algua.registry import live_gate
from algua.registry.live_gate import LiveAuthorizationError

if TYPE_CHECKING:  # annotation only: the selector reads the repository through live_gate
    from algua.registry.repository import StrategyRepository


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


def select_exit_guard(
    repo: StrategyRepository, name: str, source: Stage, target: Stage
) -> ExitLaneGuard:
    """The one exit-drain selection for both lanes (Story 2.2). ``transition_strategy`` calls it,
    inside ``operator.lock`` and after every validation, for each edge in ``_REVOKE_ON_EXIT``, with
    ``source`` the stage the store's compare-and-swap checks, so the guard's lane always matches the
    lane the exit commits from. A live source gets ``LiveExitGuard``; a paper-lane source (go-live
    included: its resting orders are paper orders) gets ``PaperExitGuard``. Neither falls open."""
    conn = getattr(repo, "connection", None)
    if conn is None:
        raise TransitionError("the exit drain needs a sqlite-backed repository")
    if source is Stage.LIVE:
        return _live_exit_guard(conn, repo, name)
    if source in (Stage.PAPER, Stage.FORWARD_TESTED):
        broker = paper_exit_drain.build_paper_drain_broker()  # module attribute: the test seam
        if broker is None:
            audit_append(
                conn, actor="system", action="paper_exit_drain_unavailable",
                reason="Alpaca paper credentials not configured; cannot drain resting paper orders",
                strategy=name)
            raise TransitionError(
                f"cannot exit paper-lane strategy {name!r}: Alpaca paper credentials are not "
                "configured, so its resting paper orders cannot be drained; set "
                "ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET, then retry")
        return paper_exit_drain.PaperExitGuard(conn, broker, name)
    raise ValueError(f"no exit drain for {source.value} -> {target.value}")


def _live_exit_guard(
    conn: sqlite3.Connection, repo: StrategyRepository, name: str
) -> ExitLaneGuard:
    """Build the LIVE-lane exit drain (cancel resting orders -> ingest -> under-lock recheck) for a
    ``live -> paper/dormant/retired`` transition.

    When the per-strategy go-live authorization is revoked/absent, the drain does NOT fall open to a
    positions-only check (which ignores resting OPEN orders — the exact orphan class #451 closed):
    it instead cancels the strategy's resting orders using the ACCOUNT-LEVEL live credentials
    (``build_live_drain_broker`` — no per-strategy authorization needed; the authorization is only a
    construction tollbooth on the trading broker, never used for REST). Only if EVEN the account
    credentials are unavailable (none configured) does the exit FAIL CLOSED — never fall open — with
    a message pointing at the `live flatten` break-glass. Both branches are audited so the drain
    path taken is attributable. (Moved verbatim from ``registry_cmd.py`` by Story 2.2.)"""
    try:
        authorization = live_gate.verify_live_authorization(
            conn, repo, name, live_gate.ALLOWED_SIGNERS_PATH)
    except LiveAuthorizationError as exc:
        # Per-strategy authorization is gone, but a resting live order left behind can still FILL
        # after the strategy leaves the live book -> orphaned position `live run-all` (iterating
        # only Stage.LIVE) never winds down. Drain it via the account-level credentials instead.
        drain = build_live_drain_broker()
        if drain is None:
            # No live credentials configured at all: we cannot reach the venue to cancel the resting
            # orders. FAIL CLOSED — never fall through to a positions-only check that ignores an
            # OPEN resting order. Human break-glass: `live flatten` before benching.
            audit_append(
                conn, actor="system", action="live_exit_drain_unavailable",
                reason=(f"live authorization unavailable ({exc}) AND live credentials not "
                        "configured; cannot drain resting orders"),
                strategy=name)
            raise TransitionError(
                f"cannot exit live strategy {name!r}: its per-strategy authorization is "
                f"unavailable ({exc}) and no live credentials are configured to drain resting "
                "orders at the account level; run `algua live flatten` to clear resting orders, "
                "then retry"
            ) from exc
        audit_append(
            conn, actor="system", action="live_exit_drain_account_creds",
            reason=(f"per-strategy authorization unavailable ({exc}); draining resting orders via "
                    "account-level live credentials"),
            strategy=name)
        return LiveExitGuard(conn, drain, name)
    return LiveExitGuard(conn, build_live_broker(authorization), name)
