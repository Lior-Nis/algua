from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import Fill, SimBroker
from algua.risk.limits import check_drawdown, check_gross_exposure, check_long_only
from algua.strategies.base import LoadedStrategy

_EPS = 1e-6


@dataclass(frozen=True)
class OrderRecord:
    """A submitted order paired with the broker order id submit() returned. Persistence reads the
    id from here rather than reconstructing it from list position (#30)."""

    intent: OrderIntent
    broker_order_id: str


@dataclass
class PaperRunResult:
    strategy: str
    orders: list[OrderRecord]
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
    # Equity is the sizing denominator; a non-positive value would silently treat every position as
    # weight 0 and buy against nothing. The drawdown breaker should have halted the run long before
    # equity reaches zero, so reaching here with equity <= 0 is a logic error, not a market state.
    assert equity > 0, f"build_intents requires positive equity, got {equity!r}"
    intents: list[OrderIntent] = []
    symbols = sorted(set(weights.index) | set(positions.index))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        shares = float(positions.get(sym, 0.0))
        current = shares * float(closes.get(sym, 0.0)) / equity
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
    max_drawdown: float = 1.0,
) -> PaperRunResult:
    """Replay the strategy bar-by-bar: decide weights on closed bar t (data <= t), submit
    orders, fill at t+1 open. Pure over the injected broker + provider."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    _reset = bars.reset_index()
    opens = _reset.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = _reset.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    ts = list(opens.index)
    warmup = strategy.execution.warmup_bars
    max_gross = strategy.execution.max_gross_exposure
    peak = broker.equity(closes.loc[ts[0]]) if ts else broker.cash
    bars_seen = 0

    orders: list[OrderRecord] = []
    fills: list[Fill] = []
    for i in range(len(ts) - 1):  # only bars with a successor can fill
        t, t_next = ts[i], ts[i + 1]
        bars_seen += 1
        # Equity/drawdown are tracked every bar (including warm-up) so the breaker sees losses.
        equity = broker.equity(closes.loc[t])
        peak = max(peak, equity)
        check_drawdown(equity, peak, max_drawdown)
        if bars_seen < warmup:
            continue  # warm-up: observe only — no signal evaluation, validation, or orders
        view = bars.loc[:t]
        weights = strategy.target_weights(view)
        check_long_only(weights, strategy.name)
        check_gross_exposure(weights, max_gross)
        for intent in build_intents(weights, broker.get_positions(), closes.loc[t], equity, t):
            order_id = broker.submit(intent)
            orders.append(OrderRecord(intent=intent, broker_order_id=order_id))
        fills.extend(broker.fill_pending(opens.loc[t_next], fill_ts=t_next))

    final_positions = {s: float(q) for s, q in broker.get_positions().items()}
    final_equity = broker.equity(closes.loc[ts[-1]]) if ts else broker.cash
    # The final bar's close is filled-at but never re-checked in-loop; check it before returning
    # so a drawdown on the last bar still trips the breaker rather than persisting as a clean run.
    peak = max(peak, final_equity)
    check_drawdown(final_equity, peak, max_drawdown)
    derived: dict[str, float] = {}
    for f in fills:
        derived[f.symbol] = derived.get(f.symbol, 0.0) + f.qty
    reconcile_ok = {s: q for s, q in derived.items() if q != 0.0} == final_positions
    return PaperRunResult(
        strategy=strategy.name, orders=orders, fills=fills,
        final_positions=final_positions, final_cash=broker.cash,
        final_equity=final_equity, reconcile_ok=reconcile_ok,
    )
