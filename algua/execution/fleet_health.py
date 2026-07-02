"""Fleet-wide, read-only health aggregation (#400).

One place that turns PERSISTED operability state into a per-strategy health rollup, and rolls that
up across the whole fleet ranked worst-offender-first. This is the shared engine behind both
``paper show`` (single strategy) and ``fleet status`` (every strategy) so the two can never drift
apart (DRY).

Pure reads only — SELECTs against the registry DB, no broker call, no writes, no locks. Lives in
``algua.execution`` (not a cli command module) because a ``cli->cli`` sibling import is forbidden
by the import-linter ``independence`` contract; ``execution -> registry/risk/calendar`` is
permitted and introduces no load-time cycle (registry's execution import is lazy/in-function).

Liveness (#399): ``health`` is NOT derived from a row merely existing. ``staleness_sessions`` counts
completed market sessions since the newest tick; a non-idle strategy whose newest tick is older than
``STALE_AFTER_SESSIONS`` — or whose ``tick_ts`` is unparseable/tz-naive/future — is reported
``stale``, never a silent ``ok``. A never-ticked strategy stays ``idle``. So ``ok`` requires actual,
fresh, parseable, non-future tick evidence.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any, Protocol

from algua.contracts.lifecycle import Stage
from algua.execution.live_ledger import (
    LedgerKind,
    believed_positions,
    paper_believed_positions,
)
from algua.execution.order_state import (
    count_orders,
    count_venue_orders,
    derive_positions,
    get_nav_peak,
    get_peak_equity,
    latest_tick_snapshot,
)
from algua.registry.repository import StrategyRecord
from algua.risk import global_halt, kill_switch

# Newest admissible tick may be at most this many completed sessions old before a non-idle strategy
# is flagged ``stale`` (matches the forward gate's MAX_STALENESS_SESSIONS). A dead operator loop is
# one of the most dangerous silent failures — positions held with no risk-checking tick running.
STALE_AFTER_SESSIONS = 5

# Worst-first severity ordering for both the ``health`` verdict and the fleet ranking.
_SEVERITY = {"halted": 0, "drift": 1, "stale": 2, "idle": 3, "ok": 4}


class _Calendar(Protocol):
    def sessions_between(self, a: Any, b: Any) -> int: ...


def _safe_latest_tick(conn: sqlite3.Connection, name: str) -> tuple[dict | None, str | None]:
    """The newest tick snapshot, or ``(None, error)`` if reading it raised.

    ``latest_tick_snapshot`` eagerly ``json.loads`` the persisted ``positions`` column, so ONE
    strategy with a corrupt row would otherwise crash the whole ``fleet status`` aggregate and hide
    every OTHER strategy's health. Catch the read here so a broken row degrades to a per-strategy
    fail-closed verdict (``stale`` + ``last_tick_error``) instead of taking down the fleet view."""
    try:
        return latest_tick_snapshot(conn, name), None
    except Exception as exc:  # corrupt positions JSON / any read error: fail closed, don't crash
        return None, f"{type(exc).__name__}: {exc}"


def _parse_utc(value: object) -> datetime | None:
    """ISO-8601 -> aware-UTC datetime, or ``None`` on anything unparseable — INCLUDING a tz-naive
    string. Every legitimate tick writer stamps an explicit offset, so a naive/garbage value can
    only be a raw-write fabrication and is rejected fail-closed (a non-idle strategy with such a
    tick surfaces as ``stale``, never ``ok``). Mirrors ``forward_promotion._parse_dt`` without
    importing that CODEOWNERS-protected module."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo is not None else None


def strategy_health(
    conn: sqlite3.Connection,
    rec: StrategyRecord,
    calendar: _Calendar,
    *,
    halted_globally: bool,
    now: datetime,
) -> dict:
    """One strategy's operability rollup from persisted state (no broker call).

    Position/peak/order readers are chosen per lane exactly like ``paper show``: a LIVE strategy
    reads the ledger + NAV peak; a strategy with paper-venue orders reads the venue ledger + equity
    peak; otherwise the sim-derived positions + equity peak.

    ``health`` precedence (worst first): ``halted`` (kill-switch or global halt) > ``drift`` (newest
    tick failed reconcile) > ``stale`` (non-idle but newest tick is unparseable/future/older than
    ``STALE_AFTER_SESSIONS``) > ``idle`` (never ticked) > ``ok``.
    """
    if rec.stage is Stage.LIVE:
        positions = believed_positions(conn, rec.name, LedgerKind.LIVE)
        peak = get_nav_peak(conn, rec.name)
        n_orders = count_orders(conn, rec.name)
    elif conn.execute(
        "SELECT 1 FROM paper_venue_orders WHERE strategy = ? LIMIT 1", (rec.name,)
    ).fetchone() is not None:
        positions = paper_believed_positions(conn, rec.name)
        peak = get_peak_equity(conn, rec.name)
        n_orders = count_venue_orders(conn, rec.name)
    else:
        positions = derive_positions(conn, rec.name)
        peak = get_peak_equity(conn, rec.name)
        n_orders = count_orders(conn, rec.name)

    ks = kill_switch.get(conn, rec.name)
    tripped = ks is not None
    last, tick_error = _safe_latest_tick(conn, rec.name)
    # A row that exists but can't be read (corrupt positions JSON) is NOT idle: the strategy has
    # ticked but is unmonitorable, so it must fail closed to ``stale`` — never ``idle``/``ok``.
    has_unreadable_tick = tick_error is not None
    last_equity = last["equity"] if last else None
    drawdown = (
        1.0 - last_equity / peak
        if last_equity is not None and peak is not None and peak > 0 else None
    )

    # Staleness: completed sessions since the newest tick. A non-idle strategy whose tick_ts is
    # unparseable/tz-naive/future — or whose row is unreadable — fails closed to sentinel staleness
    # so it can never read as ``ok``.
    staleness_sessions: int | None = None
    if has_unreadable_tick:
        staleness_sessions = STALE_AFTER_SESSIONS + 1  # fail closed -> stale
    elif last is not None:
        tick_dt = _parse_utc(last["tick_ts"])
        if tick_dt is None or tick_dt > now:
            staleness_sessions = STALE_AFTER_SESSIONS + 1  # fail closed -> stale
        else:
            staleness_sessions = calendar.sessions_between(tick_dt.date(), now.date())

    if tripped or halted_globally:
        health = "halted"
    elif last is not None and not last["reconcile_ok"]:
        health = "drift"
    elif last is None and not has_unreadable_tick:
        health = "idle"
    elif staleness_sessions is not None and staleness_sessions > STALE_AFTER_SESSIONS:
        health = "stale"
    else:
        health = "ok"

    return {
        "strategy": rec.name,
        "stage": rec.stage.value,
        "health": health,
        "staleness_sessions": staleness_sessions,
        "stale_after_sessions": STALE_AFTER_SESSIONS,
        "last_tick_error": tick_error,
        "kill_switch": {
            "tripped": tripped,
            "reason": ks["reason"] if ks else None,
            "global_halt": halted_globally,
        },
        "drawdown": {"peak_equity": peak, "last_equity": last_equity, "drawdown": drawdown},
        "last_tick": last,
        "positions": positions,
        "n_orders": n_orders,
    }


def fleet_status(conn: sqlite3.Connection, calendar: _Calendar, *, now: datetime) -> list[dict]:
    """Every strategy's health rollup, ranked worst-offender-first.

    Reads ``global_halt`` ONCE (account-wide) and ``list_strategies()`` across ALL stages, then
    rolls each up via ``strategy_health``. Ties (same severity) break by name for a stable order.
    """
    from algua.registry.store import SqliteStrategyRepository  # lazy: avoid load-time reach-around

    halted_globally = global_halt.is_engaged(conn)
    rows = [
        strategy_health(conn, rec, calendar, halted_globally=halted_globally, now=now)
        for rec in SqliteStrategyRepository(conn).list_strategies()
    ]
    rows.sort(key=lambda r: (_SEVERITY.get(r["health"], 99), r["strategy"]))
    return rows
