from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from algua.contracts.types import OrderIntent, Side
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.risk.limits import check_gross_exposure, check_long_only
from algua.strategies.base import LoadedStrategy


@dataclass
class TickResult:
    decision_ts: datetime | None
    target_weights: dict[str, float]
    positions_before: dict[str, float]
    submitted: list[dict[str, Any]]


def run_tick(
    strategy: LoadedStrategy,
    broker: AlpacaPaperBroker,
    provider: Any,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
) -> TickResult:
    """One wall-clock tick: decide on the latest closed session, submit market-order deltas to
    Alpaca (the source of truth). Pure over the injected broker + provider."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    positions_before = {s: float(q) for s, q in broker.get_positions().items()}
    if bars.empty:
        return TickResult(None, {}, positions_before, [])

    t = bars.index.max()
    if bars.index.nunique() < strategy.execution.warmup_bars:
        return TickResult(t, {}, positions_before, [])  # warm-up not met

    weights = strategy.target_weights(bars.loc[:t])
    check_long_only(weights, strategy.name)
    check_gross_exposure(weights, strategy.execution.max_gross_exposure)

    broker.cancel_open_orders()
    submitted: list[dict[str, Any]] = []
    symbols = sorted(set(weights.index) | set(broker.get_positions().index))
    for sym in symbols:
        target = float(weights.get(sym, 0.0))
        side = Side.BUY if target > 0 else Side.SELL
        order_id = broker.submit(OrderIntent(symbol=sym, side=side, target_weight=target,
                                             decision_ts=t))
        if order_id != "noop":
            submitted.append({"symbol": sym, "side": side.value,
                              "target_weight": target, "order_id": order_id})
    return TickResult(t, {s: float(w) for s, w in weights.items()}, positions_before, submitted)
