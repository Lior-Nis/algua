from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import Fill, SimBroker
from algua.risk.limits import (
    WEIGHT_TOL,
    RiskBreach,
    check_drawdown,
    validate_decision_weights,
)
from algua.strategies.base import LoadedStrategy


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
    current_weights: dict[str, float],
    decision_ts: datetime,
) -> list[OrderIntent]:
    """Emit one OrderIntent per symbol whose target weight differs from its current weight by more
    than WEIGHT_TOL. `current_weights` is each held symbol's market-value weight (shares*price over
    equity for the sim, market_value/equity for Alpaca); the caller computes it from what it has."""
    intents: list[OrderIntent] = []
    symbols = sorted(set(weights.index) | set(current_weights))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        current = float(current_weights.get(sym, 0.0))
        if abs(target - current) > WEIGHT_TOL:
            side = Side.BUY if target > current else Side.SELL
            intents.append(
                OrderIntent(symbol=sym, side=side, target_weight=target, decision_ts=decision_ts)
            )
    return intents


def decide(
    strategy: LoadedStrategy,
    view: pd.DataFrame,
    current_weights: dict[str, float],
    decision_ts: datetime,
) -> tuple[pd.Series, list[OrderIntent]]:
    """Shared decision core both loops call: evaluate target weights on the closed-bar `view`, run
    the shared decision-weight rails, then build the per-symbol intents against the caller's current
    market-value weights. Broker mechanics (sim fill_pending vs Alpaca submit) stay in each loop;
    only this weights->risk->intents step is shared (#25)."""
    weights = strategy.target_weights(view)
    validate_decision_weights(
        weights, strategy.execution, strategy.name, allowed_symbols=strategy.universe
    )
    intents = build_intents(weights, current_weights, decision_ts)
    return weights, intents


def run_paper(
    strategy: LoadedStrategy,
    broker: SimBroker,
    provider: Any,  # contracts.DataProvider; Any to keep this module import-light
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    max_drawdown: float | None = None,
    on_decision: Callable[[datetime, pd.Series], None] | None = None,
) -> PaperRunResult:
    """Replay the strategy bar-by-bar: decide weights on closed bar t (data <= t), submit
    orders, fill at t+1 open. Pure over the injected broker + provider.

    `on_decision`, if given, is called with (decision_ts, decided_weights) for every bar the
    loop actually decides on (post warm-up). It is a read-only observation seam — it cannot
    alter any decision — used to assert backtest<->paper decision parity."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    _reset = bars.reset_index()
    opens = _reset.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = _reset.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    ts = list(opens.index)
    warmup = strategy.execution.warmup_bars
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
        # warmup_bars = N holds the first N bars flat: bars_seen runs 1..len(ts)-1, so the
        # first DECIDED bar is bars_seen == N+1 (session index N) — identical to the backtest
        # loop's `if i < warmup: continue` (#1: reconcile the historical off-by-one).
        if bars_seen <= warmup:
            continue  # warm-up: observe only — no signal evaluation, validation, or orders
        # Equity is the sizing denominator; a value that is not a positive finite number would
        # ZeroDivision (== 0), flip every weight's sign (< 0), or NaN-poison every weight to a
        # silent no-op (NaN). The drawdown breaker should have halted long before, so this is a
        # logic error, not a market state — but a bare assert is stripped under `python -O`, so
        # enforce a real fail-closed breach (#162). `not (x > 0.0)` rejects NaN, `x <= 0.0` doesn't.
        if not (equity > 0.0):
            raise RiskBreach(
                "non_positive_equity",
                f"run_paper sizing equity {equity} is not a usable (positive, finite) "
                f"denominator — refusing to size against it (sign-flip / divide-by-zero / NaN)",
            )
        positions = broker.get_positions()
        bar_closes = closes.loc[t]
        current_weights = {
            s: float(positions.get(s, 0.0)) * float(bar_closes.get(s, 0.0)) / equity
            for s in positions.index
        }
        weights, intents = decide(strategy, bars.loc[:t], current_weights, t)
        if on_decision is not None:
            on_decision(t, weights)
        for intent in intents:
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
