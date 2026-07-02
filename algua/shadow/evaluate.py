from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.backtest.metrics import metrics_from_returns
from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import SimBroker
from algua.risk.limits import WEIGHT_TOL, validate_decision_weights
from algua.strategies.base import LoadedStrategy

# Shadow evaluation is a DAILY-bar exercise: metrics_from_returns annualizes with the daily
# constant, so a non-1d timeframe would produce a mis-annualized (unfair) Sharpe. Restrict to "1d".
SHADOW_TIMEFRAME = "1d"


@dataclass(frozen=True)
class ShadowResult:
    """The outcome of one shadow replay — hypothetical, paper-accounted, order-free performance.

    `equity_curve` is the per-bar mark-to-market equity series [(iso_ts, equity), ...]; the scalar
    metrics are derived from its daily returns via the shared backtest metric layer, so a shadow
    Sharpe is computed exactly like a backtest Sharpe (same annualization, same risk-free default).
    """

    strategy: str
    final_equity: float
    total_return: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    max_drawdown: float
    n_bars: int
    final_positions: dict[str, float]
    equity_curve: list[tuple[str, float]]


def _current_weights(positions: pd.Series, closes: pd.Series, equity: float) -> dict[str, float]:
    """Market-value weight per held symbol from the sim book (shares*close / equity). Mirrors the
    paper loop's current-weight computation so decisions compare targets against realized holdings.
    """
    return {
        s: float(positions.get(s, 0.0)) * float(closes.get(s, 0.0)) / equity
        for s in positions.index
    }


def _intents(
    weights: pd.Series, current_weights: dict[str, float], decision_ts: datetime,
) -> list[OrderIntent]:
    """One OrderIntent per symbol whose target weight differs from its current by > WEIGHT_TOL.

    This is the SAME rule as algua.live.paper_loop.build_intents, re-composed here so the shadow
    lane imports NOTHING from algua.live (import-linter forbids algua.shadow -> algua.live): the
    structural wall proving a shadow decision can never reach the real submit path is worth it.
    """
    intents: list[OrderIntent] = []
    for sym in sorted(set(weights.index) | set(current_weights)):
        target = float(weights.get(sym, 0.0))
        current = float(current_weights.get(sym, 0.0))
        if abs(target - current) > WEIGHT_TOL:
            side = Side.BUY if target > current else Side.SELL
            intents.append(
                OrderIntent(symbol=sym, side=side, target_weight=target, decision_ts=decision_ts)
            )
    return intents


def shadow_replay(
    strategy: LoadedStrategy,
    provider: Any,  # contracts.DataProvider; Any to keep this module import-light
    start: datetime,
    end: datetime,
    *,
    timeframe: str = SHADOW_TIMEFRAME,
    cash: float,
    now: datetime | None = None,
) -> ShadowResult:
    """Replay a strategy in SHADOW over the provider's bars and return its hypothetical performance.

    Point-in-time discipline is identical to the live/paper loops:
    - Only fully-CLOSED sessions drive a decision: any bar dated on/after `now.date()` is dropped
      before the replay (the live loop's cutoff, `now` injected for tests) so a partial current-bar
      the live champion would reject can never leak into the shadow decision.
    - Decide on closed bar `t` (data <= t), fill at the NEXT bar's open — no future bar is read
      before deciding, so there is no look-ahead.

    Accounting is a pure in-process SimBroker (no real/paper broker, no order submission, no
    allocation, no holdout). `cash` is a caller-supplied SIM notional, NOT a live allocation.
    """
    if timeframe != SHADOW_TIMEFRAME:
        raise ValueError(
            f"shadow replay supports only timeframe {SHADOW_TIMEFRAME!r} (daily) — "
            f"metrics annualization is daily-calibrated; got {timeframe!r}"
        )
    if not cash > 0:
        raise ValueError(f"cash must be a positive sim notional, got {cash}")
    now = now or datetime.now(UTC)

    bars = provider.get_bars(strategy.universe, start, end, timeframe).sort_index()
    if not bars.empty:
        # Drop any bar dated on/after today so a partial current-session bar can't drive a decision
        # (identical to algua.live.live_loop.run_tick's closed-session cutoff).
        cutoff = now.date()
        bars = bars[[ts.date() < cutoff for ts in bars.index]]

    broker = SimBroker(cash=cash)
    equity_curve: list[tuple[str, float]] = []

    if bars.empty:
        return _result(strategy.name, broker, equity_curve, closes=None, last_ts=None)

    reset = bars.reset_index()
    opens = reset.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = reset.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    ts = list(opens.index)
    warmup = strategy.execution.warmup_bars
    bars_seen = 0

    for i in range(len(ts) - 1):  # only bars with a successor can fill
        t, t_next = ts[i], ts[i + 1]
        bars_seen += 1
        bar_closes = closes.loc[t]
        equity = broker.equity(bar_closes)
        # Record the per-bar mark-to-market equity (including warm-up) so the curve is complete.
        equity_curve.append((_iso(t), float(equity)))
        # warmup_bars = N holds the first N bars flat (bars_seen == N+1 is the first decided bar),
        # identical to run_paper / the backtest loop.
        if bars_seen <= warmup:
            continue
        if not equity > 0.0:
            # A non-positive/NaN sizing denominator would divide-by-zero / sign-flip / NaN-poison.
            # In a from-scratch sim replay this is a degenerate wipeout; stop the replay honestly
            # rather than size against a broken denominator.
            break
        current_weights = _current_weights(broker.get_positions(), bar_closes, equity)
        weights = strategy.target_weights(bars.loc[:t])
        validate_decision_weights(
            weights, strategy.execution, strategy.name, allowed_symbols=strategy.universe
        )
        for intent in _intents(weights, current_weights, t):
            broker.submit(intent)
        broker.fill_pending(opens.loc[t_next], fill_ts=t_next)

    # Mark the final bar to market so the last session's P&L is in the curve.
    last_ts = ts[-1]
    equity_curve.append((_iso(last_ts), float(broker.equity(closes.loc[last_ts]))))
    return _result(strategy.name, broker, equity_curve, closes=closes, last_ts=last_ts)


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _result(
    name: str, broker: SimBroker, equity_curve: list[tuple[str, float]],
    *, closes: pd.DataFrame | None, last_ts: Any,
) -> ShadowResult:
    final_positions = {s: float(q) for s, q in broker.get_positions().items()}
    final_equity = (
        float(broker.equity(closes.loc[last_ts])) if closes is not None and last_ts is not None
        else broker.cash
    )
    # Daily returns from the equity curve; metrics_from_returns is the SAME layer the backtester
    # uses, so a shadow Sharpe is directly comparable to a backtest/holdout Sharpe.
    eq = pd.Series([v for _, v in equity_curve], dtype="float64")
    returns = eq.pct_change().dropna() if len(eq) > 1 else pd.Series([], dtype="float64")
    m = metrics_from_returns(returns)
    return ShadowResult(
        strategy=name,
        final_equity=final_equity,
        total_return=float(m["total_return"]),
        ann_return=float(m["ann_return"]),
        ann_volatility=float(m["ann_volatility"]),
        sharpe=float(m["sharpe"]),
        max_drawdown=float(m["max_drawdown"]),
        n_bars=len(equity_curve),
        final_positions=final_positions,
        equity_curve=equity_curve,
    )
