from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from algua.contracts.types import OrderIntent, fill_reference_column
from algua.execution.sim_broker import Fill, SimBroker
from algua.live.planner import build_intents as build_intents
from algua.live.planner import decide
from algua.risk.limits import RiskBreach, check_drawdown
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


def _require_held_marks(positions: pd.Series, bar_closes: pd.Series) -> None:
    """Fail closed before sizing when a held symbol has no positive finite close on this bar.

    A NaN close used to surface as ``non_positive_equity`` because it poisoned the equity
    denominator. A symbol absent from the close panel used to be valued at 0.0. Both are
    unvaluable marks. Non-positive closes follow the live lane's ``assert_marks_usable`` rule.
    """
    unvaluable: list[object] = []
    for symbol, qty in positions.items():
        if qty == 0.0:
            continue
        if symbol not in bar_closes.index:
            unvaluable.append(symbol)
            continue
        try:
            mark = float(bar_closes[symbol])
        except (TypeError, ValueError):
            unvaluable.append(symbol)
            continue
        if not (math.isfinite(mark) and mark > 0.0):
            unvaluable.append(symbol)
    if unvaluable:
        raise RiskBreach(
            "unvaluable_marks",
            "held symbols have a missing or unvaluable mark: "
            f"{sorted(unvaluable, key=str)} - "
            "refusing to value/size the book off an unvaluable feed",
        )


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
    orders, fill at the contract-pinned t+1 reference price (next-bar open by default; #383).
    Pure over the injected broker + provider.

    `on_decision`, if given, is called with (decision_ts, decided_weights) for every bar the
    loop actually decides on (post warm-up). It is a read-only observation seam — it cannot
    alter any decision — used to assert backtest<->paper decision parity."""
    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    _reset = bars.reset_index()
    opens = _reset.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = _reset.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    # Fill-price basis (issue #383): the ONE resolver both paths consult picks the raw column the
    # next-bar fill references — "open" (default) or "adj_close" (legacy close basis) — so the loop
    # can never silently fill on a different reference than the backtest pins.
    fill_col = fill_reference_column(strategy.execution)
    fill_grid = opens if fill_col == "open" else closes
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
        positions = broker.get_positions()
        bar_closes = closes.loc[t]
        # Held marks before the equity guard: a NaN close poisons equity and would otherwise be
        # reported as non_positive_equity, and a missing close must not be sized at 0.0.
        _require_held_marks(positions, bar_closes)
        # Equity is the sizing denominator; a value that is not a positive finite number would
        # ZeroDivision (== 0), flip every weight's sign (< 0), or NaN-poison every weight to a
        # silent no-op (NaN). The drawdown breaker should have halted long before, so this is a
        # logic error, not a market state — but a bare assert is stripped under `python -O`, so
        # enforce a real fail-closed breach (#162). `not (x > 0.0)` rejects NaN, `x <= 0.0` doesn't.
        if not (equity > 0.0):
            raise RiskBreach(
                "non_positive_equity",
                f"run_paper sizing equity {equity} is not a usable (positive, finite) "
                f"denominator - refusing to size against it (sign-flip / divide-by-zero / NaN)",
            )
        current_weights = {
            s: float(positions.get(s, 0.0)) * float(bar_closes[s]) / equity
            for s in positions.index
        }
        weights, intents = decide(strategy, bars.loc[:t], current_weights, t)
        if on_decision is not None:
            on_decision(t, weights)
        for intent in intents:
            order_id = broker.submit(intent)
            orders.append(OrderRecord(intent=intent, broker_order_id=order_id))
        fills.extend(broker.fill_pending(fill_grid.loc[t_next], fill_ts=t_next))

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
