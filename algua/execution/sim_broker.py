from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from algua.contracts.types import OrderIntent
from algua.execution.sizing import size_order


@dataclass(frozen=True)
class Fill:
    symbol: str
    qty: float  # signed shares: +buy, -sell (0.0 for a rejected order)
    price: float
    decision_ts: datetime
    fill_ts: datetime
    broker_order_id: str
    status: str = "filled"  # "filled" | "partial" | "rejected"


class SimBroker:
    """In-process paper broker: fills submitted orders at the next bar's open, no slippage. Sells
    are applied before buys so freed cash funds buys; cash never goes negative. Implements the
    contracts Broker surface (submit, get_positions) plus the sim-only equity()/fill_pending() the
    replay loop drives.

    Every pending order yields exactly one Fill (#26): a full fill, a partial fill (buy clamped to
    cash on hand), or a zero-qty "rejected" record (unaffordable, untradeable price, or already on
    target). This models partial-fill / rejection paths instead of silently dropping orders.
    """

    def __init__(self, cash: float) -> None:
        self.cash = float(cash)
        self.positions: dict[str, float] = {}
        self._pending: list[tuple[str, OrderIntent]] = []
        self._seq = 0

    def submit(self, intent: OrderIntent) -> str:
        self._seq += 1
        order_id = f"sim-{self._seq}"
        self._pending.append((order_id, intent))
        return order_id

    def get_positions(self) -> pd.Series:
        return pd.Series(
            {s: q for s, q in self.positions.items() if q != 0.0}, dtype="float64"
        )

    def equity(self, prices: pd.Series) -> float:
        held = sum(q * float(prices.get(s, 0.0)) for s, q in self.positions.items())
        return self.cash + held

    def fill_pending(self, opens: pd.Series, fill_ts: datetime) -> list[Fill]:
        """Fill every pending order against `opens`, returning one Fill per order. The equity
        snapshot is taken once here and used as the sizing denominator for all orders so sizing
        does not drift as earlier orders fill."""
        eq = self.equity(opens)
        # id, intent, intended signed qty, price (price is NaN when the symbol is untradeable)
        planned: list[tuple[str, OrderIntent, float, float]] = []
        rejected: list[Fill] = []
        for order_id, intent in self._pending:
            price = float(opens.get(intent.symbol, float("nan")))
            if not price > 0:
                rejected.append(Fill(intent.symbol, 0.0, 0.0, intent.decision_ts, fill_ts,
                                     order_id, status="rejected"))
                continue
            current = self.positions.get(intent.symbol, 0.0)
            sized = size_order(symbol=intent.symbol, target_weight=intent.target_weight,
                               equity=eq, current_market_value=current * price,
                               price=price, current_shares=current)
            if sized.is_noop:
                rejected.append(Fill(intent.symbol, 0.0, price, intent.decision_ts, fill_ts,
                                     order_id, status="rejected"))
                continue
            planned.append((order_id, intent, sized.delta_shares, price))
        planned.sort(key=lambda p: p[2])  # sells (negative qty) first so they free cash for buys
        fills: list[Fill] = []
        for order_id, intent, qty, price in planned:
            status = "filled"
            if qty > 0:  # buy: clamp to cash on hand -> partial (or rejected if nothing affordable)
                affordable = float(math.floor(self.cash / price))
                if affordable < qty:
                    qty = affordable
                    status = "partial"
                if qty <= 0:
                    fills.append(Fill(intent.symbol, 0.0, price, intent.decision_ts, fill_ts,
                                      order_id, status="rejected"))
                    continue
            self.cash -= qty * price
            self.positions[intent.symbol] = self.positions.get(intent.symbol, 0.0) + qty
            fills.append(Fill(intent.symbol, float(qty), price, intent.decision_ts, fill_ts,
                              order_id, status=status))
        self._pending.clear()
        return fills + rejected
