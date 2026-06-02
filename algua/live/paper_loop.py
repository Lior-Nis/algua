from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import Fill, SimBroker
from algua.strategies.base import LoadedStrategy

_EPS = 1e-6


@dataclass
class PaperRunResult:
    strategy: str
    orders: list[OrderIntent]
    fills: list[Fill]
    final_positions: dict[str, float]
    final_cash: float
    final_equity: float
    reconcile_ok: bool


def build_intents(
    weights: pd.Series,
    positions: pd.Series,
    closes: pd.Series,
    equity: float,
    decision_ts: datetime,
) -> list[OrderIntent]:
    """Emit one OrderIntent per symbol whose target weight differs from its current weight."""
    intents: list[OrderIntent] = []
    symbols = sorted(set(weights.index) | set(positions.index))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        shares = float(positions.get(sym, 0.0))
        current = (shares * float(closes.get(sym, 0.0)) / equity) if equity > 0 else 0.0
        if abs(target - current) > _EPS:
            side = Side.BUY if target > current else Side.SELL
            intents.append(
                OrderIntent(symbol=sym, side=side, target_weight=target, decision_ts=decision_ts)
            )
    return intents


def run_paper(
    strategy: LoadedStrategy,
    broker: SimBroker,
    provider: Any,  # contracts.DataProvider; Any to keep this module import-light
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
) -> PaperRunResult:
    """Replay the strategy bar-by-bar: decide weights on closed bar t (data <= t), submit
    orders, fill at t+1 open. Pure over the injected broker + provider."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    _reset = bars.reset_index()
    opens = _reset.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = _reset.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    ts = list(opens.index)

    orders: list[OrderIntent] = []
    fills: list[Fill] = []
    for i in range(len(ts) - 1):  # only bars with a successor can fill
        t, t_next = ts[i], ts[i + 1]
        view = bars.loc[:t]
        weights = strategy.target_weights(view)
        if len(weights) and bool((weights < 0).any()):
            negative = sorted(weights[weights < 0].index)
            raise ValueError(
                f"long-only: strategy '{strategy.name}' returned negative target weight(s) "
                f"for {negative} at {t}"
            )
        equity = broker.equity(closes.loc[t])
        for intent in build_intents(weights, broker.get_positions(), closes.loc[t], equity, t):
            broker.submit(intent)
            orders.append(intent)
        fills.extend(broker.fill_pending(opens.loc[t_next], fill_ts=t_next))

    final_positions = {s: float(q) for s, q in broker.get_positions().items()}
    final_equity = broker.equity(closes.loc[ts[-1]]) if ts else broker.cash
    derived: dict[str, float] = {}
    for f in fills:
        derived[f.symbol] = derived.get(f.symbol, 0.0) + f.qty
    reconcile_ok = {s: q for s, q in derived.items() if q != 0.0} == final_positions
    return PaperRunResult(
        strategy=strategy.name, orders=orders, fills=fills,
        final_positions=final_positions, final_cash=broker.cash,
        final_equity=final_equity, reconcile_ok=reconcile_ok,
    )
