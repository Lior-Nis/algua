from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.calendar.market_calendar import MarketCalendar
from algua.cli._common import registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.execution.fleet_health import fleet_status

fleet_app = typer.Typer(help="Fleet-wide observability across all strategies", no_args_is_help=True)
app.add_typer(fleet_app, name="fleet")


@fleet_app.command("status")
@json_errors
def status() -> None:
    """Fleet-wide health rollup: every strategy's stage, kill-switch/global-halt, drawdown, last
    tick, tick staleness, and health verdict in ONE read, ranked worst-offender-first. A pure read
    of persisted state (no broker call) — the standing observability view that replaces an O(N)
    sweep of ``paper show``. Emits a bare JSON array (like ``registry list``)."""
    with registry_conn() as conn:
        rows = fleet_status(conn, MarketCalendar(), now=datetime.now(UTC))
    emit(rows)
