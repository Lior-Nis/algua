"""Resolve-or-ingest bars for a lane tick (#556).

The paper/live daily tick must decide on bars that are (a) present for every symbol it needs,
(b) dated exactly the session it decides on, and (c) deep enough for each strategy's history
need. A static snapshot promises none of these, so the lane calls :func:`refresh_bars` each cycle.

ONE coverage predicate is applied both to a candidate snapshot considered for reuse and to a fresh
provider response: a same-key snapshot that no longer satisfies the CURRENT requirement (an older
accepted session, truncated history, or altered content) is never reused. The sequence
resolve -> fetch -> validate -> ingest runs under a request-keyed flock so two concurrent callers
(the 20-minute timer re-fire racing a manual cycle) cannot both miss and mint.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pandas as pd

from algua.data.contracts import BarProvider, BarRequest
from algua.data.files import logical_bars_hash
from algua.data.models import Dataset, SnapshotRecord
from algua.data.schema import empty_bars, to_bar_schema
from algua.data.store import DataStore, normalize_symbols
from algua.primitives import flock

__all__ = ["RefreshError", "refresh_bars"]

#: Never swept by any staging/GC sweep: a lock file here is only ever held or idle.
LOCK_DIRNAME = "refresh-locks"


class RefreshError(RuntimeError):
    """The bars cannot back a tick. ``kind``: ``"missing"`` (no rows), ``"stale"`` (newest bar
    older than ``require_bar_on``), ``"misdated"`` (newest bar NEWER than ``require_bar_on`` — a
    non-session row), ``"short_history"`` (fewer rows than the symbol's floor). Raised BEFORE any
    ingest, so nothing is minted and the next cycle re-queries the provider."""

    def __init__(self, kind: str, symbols: list[str], *, require_bar_on: str) -> None:
        self.kind = kind
        self.symbols = symbols
        self.require_bar_on = require_bar_on
        super().__init__(f"bars refresh: {kind} on {require_bar_on} for {symbols}")


def _to_canon(bars: pd.DataFrame) -> pd.DataFrame:
    """Bar-schema frame (timestamp index) -> the ``ts``-column canon the store hashes."""
    return bars.reset_index().rename(columns={"timestamp": "ts"})


def _coverage_error(
    canon: pd.DataFrame, symbols: list[str], require_bar_on: str, min_rows: Mapping[str, int],
) -> RefreshError | None:
    """The single coverage predicate (spec §1). ``canon`` has a tz-aware ``ts`` column."""
    want = pd.Timestamp(require_bar_on, tz="UTC").date()
    present: set[str] = set(canon["symbol"]) if not canon.empty else set()
    missing = [s for s in symbols if s not in present]
    if missing:
        return RefreshError("missing", missing, require_bar_on=require_bar_on)
    dates = pd.to_datetime(canon["ts"], utc=True).dt.date
    stale, misdated, short = [], [], []
    for s in symbols:
        mask = canon["symbol"] == s
        newest = dates[mask].max()
        if newest < want:
            stale.append(s)
        elif newest > want:
            misdated.append(s)
        if int(mask.sum()) < int(min_rows.get(s, 0)):
            short.append(s)
    if stale:
        return RefreshError("stale", stale, require_bar_on=require_bar_on)
    if misdated:
        return RefreshError("misdated", misdated, require_bar_on=require_bar_on)
    if short:
        return RefreshError("short_history", short, require_bar_on=require_bar_on)
    return None


def _matches(rec: SnapshotRecord, *, provider_name: str, symbols: list[str], start: str,
             end: str, timeframe: str, adjustment: str) -> bool:
    m = rec.metadata
    return (
        rec.dataset is Dataset.BARS
        and m.provider == provider_name
        and m.timeframe == timeframe
        and m.adjustment == adjustment
        and list(m.symbols) == symbols
        and m.start == start
        and m.end == end
    )


def _reusable(
    store: DataStore, rec: SnapshotRecord, symbols: list[str], require_bar_on: str,
    min_rows: Mapping[str, int],
) -> bool:
    """Re-validate a same-key candidate: content hash, then the CURRENT coverage predicate."""
    try:
        canon = _to_canon(store.read_bars(rec.snapshot_id))
    except Exception:  # noqa: BLE001 — unreadable payload: never reuse
        return False
    if logical_bars_hash(canon) != rec.content_hash:
        return False
    return _coverage_error(canon, symbols, require_bar_on, min_rows) is None


def refresh_bars(
    store: DataStore,
    provider: BarProvider,
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    require_bar_on: str,
    min_rows: Mapping[str, int] | None = None,
    timeframe: str = "1d",
    adjustment: str = "none",
) -> tuple[SnapshotRecord, bool]:
    """Return ``(snapshot, refreshed)``. ``refreshed`` is True iff the provider was queried and a
    new snapshot minted; False means the NEWEST same-key snapshot passed re-validation and was
    reused (no network). Provider identity is ``provider.name``.

    ``require_bar_on`` is the ISO date every symbol must have its newest bar on — the session the
    tick decides on. ``min_rows`` floors the in-window row count per symbol (absent = 0). The
    caller owns the calendar and the strategy contracts; this layer is free of both."""
    syms = normalize_symbols(list(symbols))
    floors: Mapping[str, int] = {k.upper(): int(v) for k, v in (min_rows or {}).items()}
    digest = hashlib.sha256(
        json.dumps([provider.name, syms, start, end, timeframe, adjustment]).encode()
    ).hexdigest()[:24]
    lock_dir = store.data_dir / LOCK_DIRNAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = flock.acquire(lock_dir / f"{digest}.lock", verify_inode=True)
    try:
        newest = next((r for r in reversed(store.list_snapshots(Dataset.BARS))
                       if _matches(r, provider_name=provider.name, symbols=syms, start=start,
                                   end=end, timeframe=timeframe, adjustment=adjustment)), None)
        if newest is not None and _reusable(store, newest, syms, require_bar_on, floors):
            return newest, False
        result = provider.get_bars(BarRequest(symbols=tuple(syms), start=start, end=end,
                                              timeframe=timeframe, adjustment=adjustment))
        if result.frame.empty:
            bars = empty_bars()
        else:
            bars = to_bar_schema(result.frame, timeframe=timeframe)
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        bars = bars[(bars.index >= lo) & (bars.index < hi)]
        canon = _to_canon(bars)
        err = _coverage_error(canon, syms, require_bar_on, floors)
        if err is not None:
            raise err
        rec = store.ingest_bars(
            provider=provider.name, symbols=syms, start=start, end=end,
            as_of=datetime.now(UTC).isoformat(), source=provider.name, frame=canon,
            timeframe=timeframe, adjustment=adjustment,
            source_metadata=result.source_metadata,
        )
        return rec, True
    finally:
        flock.release(fd)
