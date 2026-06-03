from __future__ import annotations

import math
from dataclasses import dataclass

# A signed notional delta whose magnitude is below this is treated as "already on target":
# no order is emitted. Keeps both brokers from churning sub-dollar rebalances.
MIN_NOTIONAL = 1.0


@dataclass(frozen=True)
class SizedOrder:
    """The result of sizing one symbol against a fixed equity snapshot.

    `delta_notional` is the signed dollar amount to trade (+buy / -sell). `delta_shares` is
    that delta expressed in whole shares at `price` (floored toward zero), for brokers that
    trade whole shares rather than notional. `is_noop` is True when the delta is below
    MIN_NOTIONAL (or rounds to zero shares) and no order should be submitted.
    """

    symbol: str
    delta_notional: float
    delta_shares: float
    is_noop: bool


def size_order(
    *,
    symbol: str,
    target_weight: float,
    equity: float,
    current_market_value: float,
    price: float | None = None,
    current_shares: float = 0.0,
) -> SizedOrder:
    """Size ONE symbol against a fixed equity snapshot — the single sizing rule both brokers share.

    The denominator (`equity`) is snapshotted once per tick by the caller and passed in for every
    symbol, so sizing is deterministic and does not drift as earlier orders fill. The target is
    `target_weight * equity`; the order is the difference between that target and the position the
    broker currently holds (valued at `current_market_value`).

    Notional brokers (Alpaca) use `delta_notional`. Whole-share brokers (the in-process sim) pass
    `price`/`current_shares` and use `delta_shares`: the share count is derived from the SAME equity
    snapshot (`floor(target_weight * equity / price) - current_shares`) so the two brokers size the
    same target. A delta below MIN_NOTIONAL — or one that rounds to zero shares — is a no-op.
    """
    if equity <= 0:
        raise ValueError(f"size_order requires positive equity, got {equity!r}")
    target_notional = target_weight * equity
    delta_notional = target_notional - current_market_value

    delta_shares = 0.0
    if price is not None and price > 0:
        target_shares = math.floor(target_notional / price)
        delta_shares = target_shares - current_shares

    is_noop = abs(delta_notional) < MIN_NOTIONAL
    if price is not None:
        # whole-share path: a sub-share delta is also a no-op even if its notional clears the floor
        is_noop = is_noop or delta_shares == 0.0
    return SizedOrder(
        symbol=symbol,
        delta_notional=delta_notional,
        delta_shares=delta_shares,
        is_noop=is_noop,
    )
