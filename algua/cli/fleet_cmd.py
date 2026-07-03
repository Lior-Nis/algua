from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.calendar.market_calendar import MarketCalendar
from algua.cli._common import registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.execution.fleet_health import (
    OPERATIONAL_STAGES,
    STALE_AFTER_SESSIONS,
    fleet_alert,
    fleet_status,
)
from algua.risk import global_halt

fleet_app = typer.Typer(help="Fleet-wide observability across all strategies", no_args_is_help=True)
app.add_typer(fleet_app, name="fleet")


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
        rows = fleet_status(conn, MarketCalendar(), now=datetime.now(UTC))
    emit(rows)


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

    Pure read of persisted state (no broker call, no writes) — cadence is measured in COMPLETED
    NYSE sessions since the last tick (via ``strategy_health``), never wall-clock, so a weekend/
    holiday gap does not false-alarm. Emits a stable summary object AND exits 0 (healthy) / 1
    (alerting); ``@json_errors`` turns even a status-engine crash into ``{ok:false}`` + exit 1
    (fail closed)."""
    with registry_conn() as conn:
        halted_globally = global_halt.is_engaged(conn)
        rows = fleet_status(conn, MarketCalendar(), now=datetime.now(UTC))
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
