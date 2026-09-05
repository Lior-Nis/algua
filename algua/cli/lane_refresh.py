"""Cycle plan + lane bars refresh for ``paper run-all --refresh`` / ``live run-all --refresh``
(#556).

Composition only — no policy the data layer or the registry doesn't already enforce. It answers
two questions for a run-all cycle: WHICH strategies are in this cycle and what bars they need
(the plan), and WHICH snapshot the cycle ticks against (the refresh).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from algua.calendar.factory import get_calendar
from algua.cli._common import SYSTEMIC_SETUP_EXCEPTIONS
from algua.config.settings import Settings, get_settings
from algua.data.providers import get_provider
from algua.data.refresh import refresh_bars
from algua.data.store import DataStore
from algua.execution.live_ledger import LedgerKind, believed_positions
from algua.registry.universe_binding import resolve_operational_universe
from algua.strategies.loader import load_tradable_strategy

__all__ = ["CyclePlan", "build_cycle_plan", "cycle_start", "lane_symbols",
           "refresh_lane_snapshot"]

#: The systemic setup faults PLUS a filesystem fault (e.g. an unreadable universe-snapshot file):
#: both are shared-infrastructure problems, not one tenant's, so they propagate raw to abort the
#: cycle rather than being isolated per-strategy (mirrors live_cmd/paper_cmd's own boundary).
_ABORT_CYCLE_EXCEPTIONS: tuple[type[BaseException], ...] = (*SYSTEMIC_SETUP_EXCEPTIONS, OSError)


@dataclass(frozen=True)
class CyclePlan:
    """The strategies this cycle will tick: each one's gate-bound operational universe, its
    ledger-believed held symbols, and a per-symbol history floor (rows the deepest strategy on
    that symbol needs); ``skipped`` carries per-tenant setup faults in the shape run-all emits."""

    universes: dict[str, list[str]]
    held: dict[str, list[str]]
    min_rows: dict[str, int]
    skipped: list[dict]

    @property
    def names(self) -> list[str]:
        return list(self.universes)


def build_cycle_plan(
    conn: sqlite3.Connection, *, names: list[str], kind: LedgerKind, data_dir: Path,
) -> CyclePlan:
    """Resolve each strategy's operational universe (#559: the gate-bound one, never the CONFIG
    template), held symbols, and history need. Admission is ``load_tradable_strategy`` — the SAME
    path the tick helpers use — so a tenant cannot pass planning and fail only at tick time. A
    per-strategy failure is ISOLATED (excluded from the plan, listed in ``skipped``) so one
    tenant's bad state never blocks its siblings; a systemic fault (``SYSTEMIC_SETUP_EXCEPTIONS``
    — a locked sqlite — or an ``OSError`` from the filesystem) propagates raw to abort the
    cycle."""
    universes: dict[str, list[str]] = {}
    held: dict[str, list[str]] = {}
    min_rows: dict[str, int] = {}
    skipped: list[dict] = []
    for name in names:
        try:
            strategy = load_tradable_strategy(name)
            symbols, _source = resolve_operational_universe(
                conn, data_dir, name, list(strategy.universe))
        except (KeyboardInterrupt, SystemExit):
            raise
        except _ABORT_CYCLE_EXCEPTIONS:
            raise
        except Exception as exc:  # noqa: BLE001 — a per-tenant setup fault, isolated by design
            skipped.append({"strategy": name, "traded": False, "skipped": f"cycle plan: {exc}"})
            continue
        lookback = strategy.config.feature_lookback
        if lookback is None:
            # UNDECLARED is not zero: the strategy cannot state its history need, so the wall
            # cannot be sized for it. Declare it (even 0) — the agent promote path already
            # requires this.
            skipped.append({"strategy": name, "traded": False,
                            "skipped": "cycle plan: undeclared_feature_lookback"})
            continue
        # Every bar-consuming contract sets the floor: the signal's lookback, the warm-up, AND
        # the capacity model's ADV window — a window too short for the ADV estimate silently
        # zeroes capacity and would force a held book flat.
        capacity = strategy.execution.capacity
        adv_window = int(capacity.adv_window_bars) if capacity is not None else 0
        need = max(int(lookback), int(strategy.execution.warmup_bars), adv_window) + 1
        universes[name] = sorted(symbols)
        for sym in universes[name]:
            min_rows[sym] = max(min_rows.get(sym, 0), need)
        held[name] = sorted(believed_positions(conn, name, kind))
    return CyclePlan(universes=universes, held=held, min_rows=min_rows, skipped=skipped)


def lane_symbols(plan: CyclePlan, broker_net: dict[str, float]) -> list[str]:
    """Every symbol the cycle can touch: plan universes ∪ ledger-held ∪ broker-truth net positions
    (orphan / residual / inherited holdings the book breakers and the mark wall value)."""
    out: set[str] = set()
    for syms in plan.universes.values():
        out.update(syms)
    for syms in plan.held.values():
        out.update(syms)
    out.update(s for s, q in broker_net.items() if q != 0.0)
    return sorted(out)


#: Heuristic seed for the start search (252 sessions/yr -> 1.45 days/session; 1.6 over-covers),
#: then an EXACT session count against the configured calendar decides.
_DAYS_PER_SESSION = 1.6
_START_PAD_DAYS = 10
_SESSION_PAD = 5          # extra sessions beyond the deepest floor (a vendor gap or two)
_WIDEN_STEP_DAYS = 30
#: Mirrors ``algua.cli._common.resolve_wall_clock_window``'s ``LIVE_WINDOW_LOOKBACK_DAYS`` (~275
#: sessions). NOT delegated to that helper: it defaults an unset *start* off real wall-clock "now"
#: regardless of the ``end`` it is given, which is correct for its own callers (an omitted
#: --start/--end pair on a live/paper run) but wrong here — a cycle's default window must be
#: anchored to ITS OWN ``end`` (deterministic and reproducible for a historical/backtest-shaped
#: cycle, not silently drifting relative to whatever day the process happens to run on).
_DEFAULT_WINDOW_DAYS = 400


def cycle_start(*, end: str, min_rows: dict[str, int]) -> str:
    """The cycle window start: the EARLIER of the default rolling start and the calendar date
    that yields ``max(min_rows) + _SESSION_PAD`` ACTUAL exchange sessions through the session the
    tick decides on. The default 400-day window holds ~275 sessions; a strategy declaring
    ``feature_lookback=336`` could never pass the wall — nor decide — without this. Exact, not a
    ratio: a holiday-dense span or another exchange cannot wedge the lane on the same
    insufficient start forever."""
    end_date = date.fromisoformat(end)
    default_start = (end_date - timedelta(days=_DEFAULT_WINDOW_DAYS)).isoformat()
    deepest = max(min_rows.values(), default=0)
    if deepest <= 0:
        return default_start
    cal = get_calendar()
    expected = cal.previous_session(end_date)
    need = deepest + _SESSION_PAD
    candidate = expected - timedelta(days=math.ceil(deepest * _DAYS_PER_SESSION) + _START_PAD_DAYS)
    while len(cal.sessions_in_range(candidate, expected)) < need:
        candidate -= timedelta(days=_WIDEN_STEP_DAYS)
    return min(default_start, candidate.isoformat())


def _lane_provider_name(kind: LedgerKind, settings: Settings) -> str:
    """PAPER uses the research-convenience default; LIVE must be named explicitly by a human —
    real-money decision data is never inherited from a default."""
    if kind is LedgerKind.LIVE:
        if not settings.bars_refresh_provider_live:
            raise ValueError(
                "live_refresh_provider_unset: set ALGUA_BARS_REFRESH_PROVIDER_LIVE to the "
                "approved live bars provider before `live run-all --refresh`")
        return settings.bars_refresh_provider_live
    return settings.bars_refresh_provider


def refresh_lane_snapshot(
    symbols: list[str], *, end: str, min_rows: dict[str, int], kind: LedgerKind,
) -> dict:
    """Resolve-or-ingest the lane's bars for ``[cycle_start, end)``, requiring every symbol's
    newest bar to fall on the session the tick decides on — the latest completed session
    strictly before ``end`` — and every decision-universe symbol to carry its history floor.
    Returns the window it used (``start``/``end``): the caller ticks over the SAME window.
    Raises (RefreshError / provider errors / an unset live provider) rather than returning a
    stale id; the caller fails the cycle closed."""
    settings = get_settings()
    provider = get_provider(_lane_provider_name(kind, settings), settings)
    start = cycle_start(end=end, min_rows=min_rows)
    require_bar_on = get_calendar().previous_session(date.fromisoformat(end)).isoformat()
    rec, refreshed = refresh_bars(
        DataStore(settings.data_dir), provider, symbols=symbols, start=start, end=end,
        require_bar_on=require_bar_on, min_rows=min_rows,
    )
    return {"id": rec.snapshot_id, "refreshed": refreshed, "symbols": len(symbols),
            "require_bar_on": require_bar_on, "provider": provider.name,
            "start": start, "end": end}
