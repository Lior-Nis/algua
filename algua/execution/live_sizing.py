"""The ledger-backed sizing view for a strategy's virtual subaccount (live or paper). Equity is the
SIZING denominator = min(allocation, NAV); NAV (allocation + realized + unrealized) is the drawdown
basis. Marks are the latest closed bar; a held symbol with no usable mark FAILS CLOSED (the loop
skips the strategy) rather than falling back to average cost — which would hide a loss and suppress
the drawdown breaker."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from algua.execution.live_ledger import LedgerKind, believed_positions, fills_table, position_pnl


class LiveSizingError(ValueError):
    """A live strategy cannot be sized this tick (e.g. a held symbol has no usable mark)."""


@dataclass(frozen=True)
class SizingSnapshot:
    """Ledger belief (NOT broker truth — that's TickSnapshot). Same fields run_tick reads:
    equity is the sizing denominator, market_values/qtys are this strategy's believed book."""

    equity: float
    market_values: dict[str, float]
    qtys: dict[str, float]


def _latest_marks(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    return {str(sym): float(c) for sym, c in bars.groupby("symbol")["close"].last().items()}


def build_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
    *,
    kind: LedgerKind,
) -> tuple[SizingSnapshot, float]:
    held = believed_positions(conn, strategy, kind)  # {symbol: signed qty}, nonzero only
    marks = _latest_marks(bars)
    symbols = set(universe) | set(held)

    nav = allocation
    market_values: dict[str, float] = {}
    qtys: dict[str, float] = {}
    for sym in symbols:
        qty = held.get(sym, 0.0)
        qtys[sym] = qty
        mark = marks.get(sym)
        if qty != 0.0 and (mark is None or mark <= 0.0):
            raise LiveSizingError(
                f"{strategy}: held symbol {sym!r} has no usable mark (got {mark!r}) — refusing to "
                "size on a fail-closed mark"
            )
        market_values[sym] = qty * (mark or 0.0)
        if qty != 0.0:
            fills = [
                (float(r["qty"]), float(r["price"]))
                for r in conn.execute(
                    f"SELECT qty, price FROM {fills_table(kind)} WHERE strategy = ? AND symbol = ? "
                    "ORDER BY fill_ts, id",
                    (strategy, sym),
                )
            ]
            assert mark is not None and mark > 0.0       # guard above already raised otherwise
            pnl = position_pnl(fills, mark=mark)
            nav += pnl.realized + pnl.unrealized

    equity = min(allocation, nav)
    if equity <= 0.0:
        # A non-positive sizing denominator would ZeroDivision / invert weights in run_tick — fail
        # closed (skip the strategy) rather than size off it (codex C1 review).
        raise LiveSizingError(f"{strategy}: NAV {nav:.2f} leaves a non-positive sizing equity")
    return SizingSnapshot(equity=equity, market_values=market_values, qtys=qtys), nav


def build_live_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
) -> tuple[SizingSnapshot, float]:
    """The live-lane sizing snapshot (alias over build_sizing_snapshot with LedgerKind.LIVE)."""
    return build_sizing_snapshot(conn, strategy, allocation, bars, universe, kind=LedgerKind.LIVE)


def build_paper_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
) -> tuple[SizingSnapshot, float]:
    """The paper-lane sizing snapshot (alias over build_sizing_snapshot with LedgerKind.PAPER)."""
    return build_sizing_snapshot(conn, strategy, allocation, bars, universe, kind=LedgerKind.PAPER)
