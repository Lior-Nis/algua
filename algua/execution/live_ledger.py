"""Per-strategy live books: order recording, crash-safe activity ingestion, and average-cost
P&L / NAV derivations. The broker account is the netted custodian; this ledger is the source of
truth for per-strategy attribution. Pure derivations are kept side-effect-free for testing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionPnl:
    qty: float          # signed net position
    avg_cost: float     # average cost of the open position (0.0 when flat)
    realized: float     # realized P&L across the fill sequence
    unrealized: float   # (mark - avg_cost) * qty  (correct for long AND short)


def position_pnl(fills: list[tuple[float, float]], mark: float) -> PositionPnl:
    """Average-cost P&L from a time-ordered list of (signed_qty, price) fills.

    Same-direction (or from-flat) fills update the average cost; opposite-direction fills realize
    P&L on the closed quantity; a fill crossing through zero closes the old side then opens the new
    side at the fill price. Unrealized uses the signed qty so it is correct for shorts:
    (mark - avg) * qty."""
    qty = 0.0
    avg = 0.0
    realized = 0.0
    for f_qty, price in fills:
        if qty == 0.0 or (qty > 0) == (f_qty > 0):
            # opening or adding in the same direction: weighted-average the cost
            new_qty = qty + f_qty
            avg = (avg * abs(qty) + price * abs(f_qty)) / abs(new_qty) if new_qty != 0.0 else 0.0
            qty = new_qty
        else:
            # reducing / closing the opposite side
            closing = min(abs(f_qty), abs(qty))
            # realized: long close gains (price-avg); short close gains (avg-price)
            realized += (price - avg) * closing if qty > 0 else (avg - price) * closing
            remaining = abs(f_qty) - closing
            qty = qty + f_qty
            if remaining > 0.0:        # crossed through zero -> open the new side at this price
                avg = price
            elif qty == 0.0:
                avg = 0.0
    unrealized = (mark - avg) * qty
    return PositionPnl(qty=qty, avg_cost=avg, realized=realized, unrealized=unrealized)
