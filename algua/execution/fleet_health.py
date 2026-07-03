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

# The complete set of verdicts ``strategy_health`` can emit. Any OTHER value on a row is a
# corrupt/unknown verdict and the active liveness gate fails closed on it (never a silent ``ok``).
_KNOWN_HEALTHS = frozenset(_SEVERITY)

# Every legal lifecycle stage value. A row whose ``stage`` is outside this set is corrupt and the
# active gate fails closed on it (never a silent ``ok``).
_ALL_STAGES = frozenset(s.value for s in Stage)

# Stages an operator loop actually ticks — the ONLY stages for which a dead/stalled/never-started
# loop is a real alert. VERIFIED against origin/main: ``registry.gating.load_gated_strategy`` gates
# every paper tick surface (``paper run`` / ``paper trade-tick``) to PAPER or FORWARD_TESTED, and
# ``live run-all`` ticks LIVE. A strategy in any other stage (idea/backtested/candidate/dormant/
# retired) is NOT run by a loop, so its old/absent tick is expected — not a heartbeat failure. The
# ``test_operational_stages_match_gating`` drift-guard pins this to the real tick surface so it can
# never silently diverge from the loop it watches (#399).
OPERATIONAL_STAGES = frozenset({Stage.LIVE.value, Stage.PAPER.value, Stage.FORWARD_TESTED.value})

# For an OPERATIONAL strategy these verdicts each mean a loop that is stopped, stalled, drifted, or
# never started — every one an actively-alertable liveness failure. ``idle`` (never ticked) alerts
# here because a live/paper loop that never produced a tick is a loop that never started; on a
# NON-operational stage ``idle`` is correctly quiet (see :func:`fleet_alert`). ``halted`` alerts
# because a stopped, unmonitored operational loop is exactly the silent failure #399 targets.
_ALERT_HEALTHS_OPERATIONAL = frozenset({"stale", "drift", "idle", "halted"})


class _Calendar(Protocol):
    def sessions_between(self, a: Any, b: Any) -> int: ...


def fleet_alert(
    rows: list[dict], *, halted_globally: bool,
    operational_stages: frozenset[str] = OPERATIONAL_STAGES,
) -> list[dict]:
    """The rows that should ALERT an external liveness watchdog, worst-offender-first (#399).

    This is the ACTIVE half of loop-liveness monitoring: #400 computes the fail-closed staleness
    verdict per strategy (``strategy_health``); this turns that standing view into a heartbeat
    ALARM an external supervisor (systemd ``OnFailure=``, cron, k8s liveness) can poll via the
    ``fleet health`` exit code. Pure — no I/O, classifies pre-computed ``fleet_status`` rows.

    A row alerts iff ANY of:

    * ``halted_globally`` — an account-wide global halt makes EVERY row alert (the whole book is
      stopped); it is also surfaced at the command level so an engaged halt trips the gate even
      with zero strategy rows.
    * its ``health`` is UNKNOWN, or its ``stage`` is UNKNOWN — a corrupt/malformed row is never a
      silent pass for an active gate; it fails closed to an alert.
    * its ``stage`` is operational AND its ``health`` is one of ``stale``/``drift``/``idle``/
      ``halted`` — a stopped, stalled, drifted, or never-started operator loop.

    A per-strategy kill-switch (``halted``) on a NON-operational strategy (e.g. a retired strategy
    with a lingering kill-switch) does NOT alert via the row rule — only a global halt does — so a
    benched strategy can never wedge the watchdog permanently red. An ``idle`` non-operational
    strategy (never ticked, not being run) is correctly quiet.
    """
    def _alerts(row: dict) -> bool:
        if halted_globally:
            return True
        health = row.get("health")
        stage = row.get("stage")
        if health not in _KNOWN_HEALTHS or stage not in _ALL_STAGES:  # corrupt row -> fail closed
            return True
        return stage in operational_stages and health in _ALERT_HEALTHS_OPERATIONAL

    alerting = [r for r in rows if _alerts(r)]
    alerting.sort(
        key=lambda r: (_SEVERITY.get(str(r.get("health")), 99), str(r.get("strategy", "")))
    )
    return alerting


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
