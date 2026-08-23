from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.calendar.factory import get_calendar
from algua.cli._common import ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.execution.fleet_health import (
    OPERATIONAL_STAGES,
    STALE_AFTER_SESSIONS,
    fleet_alert,
    fleet_status,
)
from algua.execution.order_state import tick_snapshot_series
from algua.registry.db import registry_conn
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt

fleet_app = typer.Typer(help="Fleet-wide observability across all strategies", no_args_is_help=True)
app.add_typer(fleet_app, name="fleet")

_SERIES_LANES = ("live", "paper")


def _norm_since(value: str) -> datetime:
    """Parse a user-supplied --since bound to an aware-UTC datetime (ISO-8601; naive assumed UTC,
    offset-aware converted to the true UTC instant). Unparseable input raises ``ValueError``
    (rendered as the JSON error envelope by @json_errors). Local copy of ``audit_cmd._norm_ts``
    semantics — cli command modules never import each other (composition-root rule, see main.py)."""
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


@fleet_app.command("status")
@json_errors
def status() -> None:
    """Fleet-wide health rollup: every strategy's stage, kill-switch/global-halt, drawdown, last
    tick, tick staleness, and health verdict in ONE read, ranked worst-offender-first. A pure read
    of persisted state (no broker call) — the standing observability view that replaces an O(N)
    sweep of ``paper show``. Emits a bare JSON array (like ``registry list``)."""
    # registry_conn() is the sanctioned read-path opener shared by every read command (paper show,
    # registry list, data inspect): its migrate() is idempotent and touches no strategy/trading
    # state, so "read-only" holds at the domain level (no order/kill-switch/allocation mutation).
    with registry_conn() as conn:
        rows = fleet_status(conn, get_calendar(), now=datetime.now(UTC))
    emit(rows)


@fleet_app.command("series")
@json_errors
def series(
    name: str = typer.Argument(..., help="strategy name"),
    since: str = typer.Option(
        None, "--since",
        help="inclusive lower bound on tick_ts (ISO-8601, assumed UTC if naive); "
             "filtering happens BEFORE the newest-N cut"),
    limit: int = typer.Option(
        500, "--limit", min=1, max=5000, help="max rows kept per lane (the newest N)"),
    lane: str = typer.Option(None, "--lane", help="restrict to one lane: paper|live"),
) -> None:
    """Per-lane tick-snapshot time series for ONE strategy (the equity curve behind the fleet
    rollup), event-time-ordered ascending. A pure read of persisted state (no broker call) — the
    drill-down companion to ``fleet status``'s one-row-per-strategy view. Legacy pre-v21 rows
    (no strategy_id/lane) are excluded and surfaced as ``n_legacy_excluded``; unparseable/
    invalid-lane rows are skipped and counted, never silently dropped."""
    if lane is not None and lane not in _SERIES_LANES:
        raise ValueError(f"lane must be one of {list(_SERIES_LANES)!r}, got {lane!r}")
    since_dt = _norm_since(since) if since is not None else None
    with registry_conn() as conn:
        # get() raises StrategyNotFound (a LookupError) for an unknown name -> {ok:false,
        # code:"not_found"} via @json_errors.
        rec = SqliteStrategyRepository(conn).get(name)
        result = tick_snapshot_series(
            conn, rec.id, rec.name, lane=lane, since=since_dt, limit=limit)
    emit(ok({"strategy": name, "lane_filter": lane, **result}))


@fleet_app.command("health")
@json_errors
def health() -> None:
    """Loop-liveness / heartbeat GATE for an external watchdog (#399). Runs the same fleet health
    rollup as ``fleet status`` and EXITS NON-ZERO iff any operator loop is dead/stalled/drifted/
    never-started (an operational strategy that is ``stale``/``drift``/``idle``/``halted``) OR the
    account is globally halted OR a fleet row is corrupt. A dead operator loop is one of the most
    dangerous silent failures — positions held with no risk-checking tick — so this is the signal a
    supervisor (systemd ``OnFailure=``, cron, k8s liveness) polls to distinguish 'market closed,
    correctly quiet' from 'loop dead, silently unmonitored'.

    DOMAIN read-only (like ``fleet status``): no broker call, no order/kill-switch/allocation
    mutation. It opens the DB via ``registry_conn()``, whose idempotent ``migrate()`` may run
    schema DDL on a stale DB — so it is not byte-literally read-only, only trading-state
    read-only. Cadence is measured in COMPLETED sessions of the CONFIGURED exchange
    (``ALGUA_EXCHANGE``, default XNYS) since the last tick (via
    ``strategy_health``), never wall-clock, so a weekend/holiday gap does not false-alarm. Emits a
    stable summary object AND exits 0 (healthy) / 1 (alerting); ``@json_errors`` turns even a
    status-engine crash into ``{ok:false}`` + exit 1 (fail closed)."""
    with registry_conn() as conn:
        # Sample the account-wide global halt ONCE and thread that SAME value through both the
        # rollup and the alert decision, so an engage/clear landing between two reads can't produce
        # an inconsistent verdict (rows say ok while the gate says halted, or vice-versa).
        halted_globally = global_halt.is_engaged(conn)
        rows = fleet_status(conn, get_calendar(), now=datetime.now(UTC),
                            halted_globally=halted_globally)
    alerting = fleet_alert(rows, halted_globally=halted_globally)
    by_health: dict[str, int] = {}
    for r in alerting:
        by_health[r["health"]] = by_health.get(r["health"], 0) + 1
    ok = not alerting and not halted_globally
    emit({
        "ok": ok,
        "global_halt": halted_globally,
        "alerting": alerting,
        "summary": {
            "total": len(rows),
            "alerting": len(alerting),
            "by_health": by_health,
        },
        "stale_after_sessions": STALE_AFTER_SESSIONS,
        "operational_stages": sorted(OPERATIONAL_STAGES),
        "rows": rows,
    })
    raise typer.Exit(code=0 if ok else 1)
