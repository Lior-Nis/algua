from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from algua.contracts.types import OrderIntent


@dataclass(frozen=True)
class Fill:
    symbol: str
    qty: float  # signed shares: +buy, -sell
    price: float
    decision_ts: datetime
    fill_ts: datetime
    broker_order_id: str


class SimBroker:
    """In-process paper broker: fills submitted orders at the next bar's open, full fill,
    no slippage. Sells are applied before buys so freed cash funds buys; cash never goes
    negative. Implements the contracts Broker surface (submit, get_positions) plus the
    sim-only equity()/fill_pending() the replay loop drives."""

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
        eq = self.equity(opens)
        planned: list[tuple[str, OrderIntent, float, float]] = []  # id, intent, qty, price
        for order_id, intent in self._pending:
            price = float(opens.get(intent.symbol, float("nan")))
            if not price > 0:
                continue
            target_shares = math.floor(intent.target_weight * eq / price)
            qty = target_shares - self.positions.get(intent.symbol, 0.0)
            if qty != 0.0:
                planned.append((order_id, intent, qty, price))
        planned.sort(key=lambda p: p[2])  # sells (negative qty) first
        fills: list[Fill] = []
        for order_id, intent, qty, price in planned:
            if qty > 0:  # buy: clamp to cash on hand
                qty = min(qty, float(math.floor(self.cash / price)))
                if qty <= 0:
                    continue
            self.cash -= qty * price
            self.positions[intent.symbol] = self.positions.get(intent.symbol, 0.0) + qty
            fills.append(
                Fill(intent.symbol, float(qty), price, intent.decision_ts, fill_ts, order_id)
            )
        self._pending.clear()
        return fills
