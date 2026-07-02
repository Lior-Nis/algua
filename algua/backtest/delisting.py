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

# Float-noise tolerance for held-position detection. Kept independent of the engine's
# POSITION_EPS to preserve the pure-module / no-engine-import property of this module.
_WEIGHT_EPS = 1e-12


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
    """Greatest bar in `index` whose date is <= d (as-of). None if d precedes the first bar.
    Calendar-free. Callers pass the SYMBOL's own traded bars (not the union panel index) so a
    vendor delisting_date that lands a day past the last trade — on a date present in the union
    index only because another symbol traded then — still resolves to this symbol's terminal bar.
    `index` is sorted; binary search makes this O(log n)."""
    # bars are daily UTC timestamps; compare against end-of-day d to include a same-date bar.
    # +1 day cutoff with side="left" includes any timestamp strictly before (d+1) 00:00,
    # i.e. all of day d. Handle both tz-aware and tz-naive indexes.
    tz = index.tz
    base = pd.Timestamp(d, tz=tz) if tz is not None else pd.Timestamp(d)
    cutoff = base + pd.Timedelta(days=1)
    pos = index.searchsorted(cutoff, side="left")
    return index[pos - 1] if pos > 0 else None


def apply_delisting_exits(
    adj: pd.DataFrame,
    weights_eff: pd.DataFrame,
    records: Mapping[str, list[DelistingRecord]] | None = None,
    *,
    assume_terminal_last_close: bool = False,
    terminal_price_grid: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (adj_exec, weights_exec, forced_exits). See module docstring + spec.

    `adj` is the ORDINARY fill-price grid (adj_close, or adj_open under issue #383's open-fill
    basis). `terminal_price_grid` (default: `adj`) is the grid the human-only
    `assume_terminal_last_close` fallback reads the last-bar terminal price from — pinned to
    adj_close by the caller so a delisting realizes at the last CLOSE (a close-of-book event),
    never the last open, regardless of the ordinary fill basis. Record-backed terminal prices are
    explicit adj_close-unit values and are unaffected."""
    records = records or {}
    if terminal_price_grid is None:
        terminal_price_grid = adj
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
        sym_index = col.dropna().index  # this symbol's OWN traded bars (T == sym_index[-1])
        sym_records = list(records.get(c, []))

        # Integrity: a record dated within the panel that resolves (against this symbol's own
        # bars) to a bar BEFORE its terminal bar means the symbol kept trading past the stated
        # delisting — a data inconsistency. Fail closed.
        for r in sym_records:
            if r.delisting_date > panel_end_date:
                continue
            d_bar = _resolve_bar(sym_index, r.delisting_date)
            if d_bar is not None and d_bar != T:
                raise DelistingExitError(
                    f"{c}: bars exist after stated delisting {r.delisting_date.isoformat()} "
                    f"(resolved bar {d_bar.date().isoformat()}, last bar {T.date().isoformat()})"
                )

        # Applicable record: in [first_bar, panel_end] and resolving (against this symbol's own
        # bars) exactly to its terminal bar T.
        candidates = [
            r
            for r in sym_records
            if first_bar.date() <= r.delisting_date <= panel_end_date
            and _resolve_bar(sym_index, r.delisting_date) == T
        ]
        if len(candidates) >= 2:
            raise DelistingExitError(
                f"{c}: {len(candidates)} delisting records resolve to the same terminal bar "
                f"{T.date().isoformat()} (ambiguous terminal valuation)"
            )
        record = candidates[0] if candidates else None
        held = bool(abs(weights_eff.loc[T, c]) > _WEIGHT_EPS)
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
            # Last adj_CLOSE (from terminal_price_grid), NOT the ordinary fill grid — a delisting
            # realizes at the close-of-book, never the open (issue #383). Write it into adj_exec so
            # the forced liquidation at bar T actually fills at that close, matching the reported
            # terminal_price; under the legacy close-fill basis terminal_price_grid IS `adj`, so
            # this is a byte-identical no-op there.
            terminal = float(terminal_price_grid.loc[T, c])
            adj_exec.loc[T, c] = terminal
            forced_exits.append(
                {
                    "symbol": c,
                    "bar": T.isoformat(),
                    "terminal_price": terminal,
                    "source": "assumed_last_close",
                }
            )

        # NaN-poison kill + phantom-weight suppression (whenever the column has a dead tail).
        # Zero post-T weights unconditionally: a delisted symbol cannot be traded after its
        # last bar, regardless of held/record/relaxation status. The held-at-T branches above
        # already set weights_exec.loc[T:, c] = 0.0 (which overlaps harmlessly). adj_exec.loc[T, c]
        # is the realized price (overridden above when a record applied), inert at a 0 position.
        if ends_early:
            after_t = adj_exec.index > T
            # No trading a delisted symbol after its last bar.
            weights_exec.loc[after_t, c] = 0.0
            adj_exec.loc[after_t, c] = adj_exec.loc[T, c]  # NaN-poison kill

    return adj_exec, weights_exec, forced_exits
