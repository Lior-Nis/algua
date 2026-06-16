"""Pure constituents transformer: (symbol, add_date, drop_date) intervals → a minimal
point-in-time membership timeline (one membership set per change date).

Convention: ``add_date`` inclusive, ``drop_date`` exclusive (empty drop = open / still a
member). No I/O — the CLI feeds the result to ``DataStore.ingest_universe`` one snapshot per
change date.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ConstituentInterval:
    symbol: str
    add_date: date
    drop_date: date | None  # None = open (still a member)


def _norm_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("symbol must not be empty")
    return s


def parse_constituents_rows(rows: list[dict[str, str]]) -> list[ConstituentInterval]:
    """Canonicalize + validate raw CSV rows into intervals (symbol normalized BEFORE checks).

    Rejects: empty symbol, unparseable dates, ``add_date >= drop_date`` (covers add>drop and the
    degenerate zero-length ``add == drop``). Exact-duplicate rows are de-duplicated.
    """
    seen: set[tuple[str, date, date | None]] = set()
    out: list[ConstituentInterval] = []
    for row in rows:
        symbol = _norm_symbol(str(row.get("symbol", "")))
        add_date = date.fromisoformat(str(row["add_date"]).strip())
        drop_raw = str(row.get("drop_date", "") or "").strip()
        drop_date = date.fromisoformat(drop_raw) if drop_raw else None
        if drop_date is not None and add_date >= drop_date:
            raise ValueError(
                f"{symbol}: add_date must be < drop_date "
                f"({add_date} >= {drop_date}; "
                f"zero-length / add_date == drop_date intervals are rejected)"
            )
        key = (symbol, add_date, drop_date)
        if key in seen:
            continue
        seen.add(key)
        out.append(ConstituentInterval(symbol, add_date, drop_date))
    return out


def constituents_to_snapshots(
    intervals: list[ConstituentInterval],
) -> list[tuple[date, frozenset[str]]]:
    """Intervals → minimal membership timeline. Rejects overlapping intervals per symbol;
    collapses consecutive no-op change dates (membership unchanged)."""
    by_symbol: dict[str, list[ConstituentInterval]] = defaultdict(list)
    for iv in intervals:
        by_symbol[iv.symbol].append(iv)
    for symbol, ivs in by_symbol.items():
        ivs.sort(key=lambda i: i.add_date)
        for prev, nxt in zip(ivs, ivs[1:], strict=False):
            if prev.drop_date is None or nxt.add_date < prev.drop_date:
                raise ValueError(f"{symbol}: overlapping membership intervals")

    change_dates = sorted(
        {iv.add_date for iv in intervals}
        | {iv.drop_date for iv in intervals if iv.drop_date is not None}
    )

    timeline: list[tuple[date, frozenset[str]]] = []
    prev_members: frozenset[str] | None = None
    for d in change_dates:
        members = frozenset(
            iv.symbol
            for iv in intervals
            if iv.add_date <= d and (iv.drop_date is None or d < iv.drop_date)
        )
        if members == prev_members:
            continue
        timeline.append((d, members))
        prev_members = members
    return timeline
