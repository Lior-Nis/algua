"""Classifying the Alpaca order rejections that are not failures, and recovering from them.

`client_order_id()` is deterministic over (strategy, decision_ts, symbol) so that a retried submit
reuses the id instead of double-filling. That design assumed the VENUE de-duplicates -- the retry
policy in `alpaca_broker` said so outright: "a retried POST that already landed is de-duplicated by
Alpaca rather than double-filling". It does not. Alpaca answers a repeat with

    422 {"code":42210000,"message":"client_order_id must be unique"}

which `_read` turned into a `BrokerError`. Because `paper run-all` ticks every tenant inside ONE
cycle, a single 422 aborted the cycle for all of them. Measured on the live box before this change:
18 aborts in 6 hours, every one of them this rejection, with the lane producing nothing for days.

So the de-duplication happens here instead: a duplicate rejection is not a failure, it is proof the
order ALREADY LANDED, and the recovery is to fetch that order and return its id. The submit becomes
idempotent in fact rather than by assumption.

Deliberately NOT the alternative fix: minting a fresh random id on rejection would turn one order
into two in the live lane. The duplicate id is the safety mechanism, not the bug.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from algua.execution.errors import BrokerError

# Both halves must match. A bare "must be unique" could be some other field, and a bare mention of
# client_order_id appears in unrelated validation errors (e.g. one that is too long) -- those are
# real failures and must keep raising. Alpaca's own code for this rejection is 42210000, matched
# too so a future message rewording does not silently re-open the abort.
_MENTIONS_COID = re.compile(r"client[ _]?order[ _]?id|42210000", re.IGNORECASE)
_MEANS_DUPLICATE = re.compile(r"unique|duplicate|already exist", re.IGNORECASE)


def is_duplicate_client_order_id(status_code: int, text: str) -> bool:
    """True iff this response is Alpaca rejecting a submit because the id is already in use.

    Narrow ON PURPOSE. A 422 is also how Alpaca reports insufficient buying power and malformed
    orders; treating those as "already landed" would silently drop real orders and report a phantom
    order id, which is far worse than the abort this removes.
    """
    return (
        status_code == 422
        and bool(_MENTIONS_COID.search(text))
        and bool(_MEANS_DUPLICATE.search(text))
    )


def recover_duplicate_order_id(
    lookup: Callable[[str], dict[str, Any] | None],
    client_order_id: str,
    *,
    symbol: str,
    side: str,
    path: str,
) -> str:
    """The broker id of the order already holding `client_order_id`, verified to be OUR order.

    `lookup` is the broker's per-coid read (None on 404), so this module never builds the endpoint.

    The returned payload is a SAFETY BOUNDARY, mirroring #312 stranded-order recovery in
    `live_ledger.recover_stranded_orders`: the returned client_order_id must match exactly, the
    symbol must be the one we submitted, and the id must be a non-empty string (a truthy non-str
    would coerce to a bogus broker id). Without this a mis-addressed lookup hands back an unrelated
    order, and `flatten` would then count someone else's order as this strategy's liquidation.

    Unlike the #312 path, SIDE IS CHECKED here: there the recorded intent side can legitimately
    differ from the delta-derived POSTed side, whereas here we compare against the body we just
    posted, so a mismatch is always wrong. It matters most on the liquidation path, where accepting
    a buy as a sell would misreport a position as closed.

    Size is deliberately NOT compared. The same decision re-run under a changed allocation posts a
    different notional for the same coid, and the order that already landed is still the correct
    idempotent answer; requiring equality would break legitimate recovery.
    """
    order = lookup(client_order_id)
    if order is None:
        # The venue said the id was taken and then 404ed on it. Never invent an id here: the caller
        # would record a fill against an order nobody can reconcile.
        raise BrokerError(
            f"alpaca {path}: client_order_id {client_order_id!r} was rejected as a duplicate but "
            f"no order carries it"
        )
    order_id = order.get("id")
    if (
        order.get("client_order_id") != client_order_id
        or order.get("symbol") != symbol
        or str(order.get("side", "")).lower() != side.lower()
        or not isinstance(order_id, str)
        or not order_id.strip()
    ):
        raise BrokerError(
            f"alpaca {path}: the order returned for client_order_id {client_order_id!r} does not "
            f"match the submitted order (expected {symbol} {side}); refusing to attribute it"
        )
    return order_id
