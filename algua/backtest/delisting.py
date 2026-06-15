"""Delisting-aware exit overlay — pure transform on the (timestamp × symbol) execution grid,
applied right before vectorbt `from_orders`. It NEVER invents an exit: it forces a liquidation
only for a position HELD past its last real bar, and only with a confirmed delisting record
(else fail-closed, unless the human-only relaxation). It always kills the post-delisting NaN
tail so `0 × NaN` cannot poison group equity. See the #212 design spec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class DelistingRecord:
    delisting_date: date
    terminal_price: float  # per-share terminal proceeds, in adj_close units; strictly > 0
    source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.terminal_price) or self.terminal_price <= 0:
            raise ValueError(
                "terminal_price must be finite and > 0 (zero-proceeds write-off deferred)"
            )


class DelistingExitError(Exception):
    """Fail-closed condition in the delisting-exit overlay (engine translates to BacktestError)."""


def _resolve_bar(index: pd.DatetimeIndex, d: date) -> pd.Timestamp | None:
    """Greatest bar whose date is <= d (as-of in the panel's own index). None if d precedes
    the first bar. Calendar-free — uses only the panel index."""
    eligible = [ts for ts in index if ts.date() <= d]
    return eligible[-1] if eligible else None


def apply_delisting_exits(
    adj: pd.DataFrame,
    weights_eff: pd.DataFrame,
    records: Mapping[str, list[DelistingRecord]] | None = None,
    *,
    assume_terminal_last_close: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (adj_exec, weights_exec, forced_exits). See module docstring + spec."""
    records = records or {}
    adj_exec = adj.copy()
    weights_exec = weights_eff.copy()
    forced_exits: list[dict] = []
    if len(adj.index) == 0:
        return adj_exec, weights_exec, forced_exits

    panel_end = adj.index[-1]
    panel_end_date = panel_end.date()

    for c in adj.columns:
        col = adj[c]
        T = col.last_valid_index()
        if T is None:
            continue  # never traded in this panel
        first_bar = col.first_valid_index()
        sym_records = list(records.get(c, []))

        # Integrity: any record dated within the panel whose resolved bar has REAL bars after it.
        for r in sym_records:
            if r.delisting_date > panel_end_date:
                continue
            d_bar = _resolve_bar(adj.index, r.delisting_date)
            if d_bar is None:
                continue
            later = col.loc[col.index > d_bar]
            if bool(later.notna().any()):
                raise DelistingExitError(
                    f"{c}: bars exist after stated delisting {r.delisting_date.isoformat()} "
                    f"(resolved bar {d_bar.date().isoformat()})"
                )

        # Applicable record: in [first_bar, panel_end] and resolving exactly to T.
        candidates = [
            r
            for r in sym_records
            if first_bar.date() <= r.delisting_date <= panel_end_date
            and _resolve_bar(adj.index, r.delisting_date) == T
        ]
        if len(candidates) >= 2:
            raise DelistingExitError(
                f"{c}: {len(candidates)} delisting records resolve to the same terminal bar "
                f"{T.date().isoformat()} (ambiguous terminal valuation)"
            )
        record = candidates[0] if candidates else None
        held = bool(weights_eff.loc[T, c] != 0)
        ends_early = T < panel_end

        if record is not None and held:
            weights_exec.loc[T:, c] = 0.0
            adj_exec.loc[T, c] = record.terminal_price
            forced_exits.append(
                {
                    "symbol": c,
                    "bar": T.isoformat(),
                    "terminal_price": float(record.terminal_price),
                    "source": record.source,
                }
            )
        elif ends_early and held and record is None:
            if not assume_terminal_last_close:
                raise DelistingExitError(
                    f"{c}: held position past its last bar {T.date().isoformat()} with no "
                    f"delisting record (provide one or pass --assume-terminal-last-close)"
                )
            weights_exec.loc[T:, c] = 0.0
            forced_exits.append(
                {
                    "symbol": c,
                    "bar": T.isoformat(),
                    "terminal_price": float(col.loc[T]),
                    "source": "assumed_last_close",
                }
            )

        # NaN-poison kill (whenever the column has a dead tail). adj_exec.loc[T, c] is the
        # realized price (overridden above when a record applied), inert at a 0 position.
        if ends_early:
            adj_exec.loc[adj_exec.index > T, c] = adj_exec.loc[T, c]

    return adj_exec, weights_exec, forced_exits
