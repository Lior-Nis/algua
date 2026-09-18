"""Client-side idempotency for Alpaca order submission (#560).

`client_order_id()` is deterministic over (strategy, decision_ts, symbol) so that a retried submit
reuses the id instead of double-filling. That design assumed the VENUE de-duplicates — the retry
policy in `alpaca_broker` says so in as many words ("a retried POST that already landed is
de-duplicated by Alpaca rather than double-filling"). It does not. Alpaca answers a repeat id with

    422 {"code":42210000,"message":"client_order_id must be unique"}

which `_read` turns into a `BrokerError`, which aborts the WHOLE multi-tenant paper cycle. Every
20 minutes, for every tenant behind the one that tripped it. The assumed safety never existed.

So the de-duplication is done here instead: a duplicate rejection is not a failure, it is proof the
order ALREADY LANDED, and the recovery is to fetch that order and return its id. The submit becomes
idempotent in fact rather than by assumption.

Deliberately NOT the alternative fix: minting a fresh random id on rejection would turn one order
into two in the live lane. The duplicate id is the safety mechanism, not the bug.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from algua.execution.errors import BrokerError

BY_CLIENT_ORDER_ID = "/v2/orders:by_client_order_id"

# Both halves must match. A bare "must be unique" could be some other field, and a bare mention of
# client_order_id appears in unrelated validation errors (e.g. one that is too long) -- those are
# real failures and must keep raising. Alpaca's own code for this rejection is 42210000, matched
# too so a future message rewording does not silently re-open the abort.
_MENTIONS_COID = re.compile(r"client[ _]?order[ _]?id|42210000", re.IGNORECASE)
_MEANS_DUPLICATE = re.compile(r"unique|duplicate|already exist", re.IGNORECASE)


def is_duplicate_client_order_id(status_code: int, text: str) -> bool:
    """True iff this response is Alpaca rejecting a submit because the id is already in use.

    Narrow ON PURPOSE. A 422 is also how Alpaca reports insufficient buying power and malformed
    orders; treating those as "already landed" would silently drop real orders and report a
    phantom fill, which is far worse than the abort this fix removes.
    """
    return (
        status_code == 422
        and bool(_MENTIONS_COID.search(text))
        and bool(_MEANS_DUPLICATE.search(text))
    )


def recover_duplicate_order_id(
    get: Callable[[str], Any], client_order_id: str, *, path: str
) -> str:
    """Fetch the broker id of the order that already holds `client_order_id`.

    `get` takes a path and returns the decoded JSON body (the broker's read-and-raise helper), so
    this stays independent of the HTTP client and its retry policy.
    """
    data = get(f"{BY_CLIENT_ORDER_ID}?client_order_id={quote(client_order_id, safe='')}")
    order_id = data.get("id") if isinstance(data, dict) else None
    if not order_id:
        # The venue said the id was taken and then could not produce its order. Never invent an id
        # here: the caller would record a fill against an order nobody can reconcile.
        raise BrokerError(
            f"alpaca {path}: client_order_id {client_order_id!r} rejected as duplicate but "
            f"{BY_CLIENT_ORDER_ID} returned no order id: {data}"
        )
    return str(order_id)
