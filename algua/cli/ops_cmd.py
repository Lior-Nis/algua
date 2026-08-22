"""`algua ops status` and `algua book status` — the two read-only surfaces the monitor's triage
screen is built on (spec: 2026-08-15-monitor-dashboard-redesign).

Both are pure reads: no broker call, no writes, no locks. Thin command bodies — the rollup logic
lives in ``algua.operator.loop_health`` and ``algua.execution.book`` so it is unit-testable without
a CLI, matching the domain-extraction convention (#165).
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.calendar.factory import get_calendar
from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.execution.book import book_status
from algua.operator.loop_health import loop_status

ops_app = typer.Typer(help="Machine liveness: are the autonomous loops still running?",
                      no_args_is_help=True)
book_app = typer.Typer(help="Book capital: who holds a slice, and who should but doesn't",
                       no_args_is_help=True)
app.add_typer(ops_app, name="ops")
app.add_typer(book_app, name="book")


@ops_app.command("status")
@json_errors
def ops_status() -> None:
    """Liveness of every autonomous loop (research / paper / merge-back), worst-first.

    Answers the question the fleet rollup structurally CANNOT: a dead research loop produces no
    strategy rows to be unhealthy about, so ``fleet health`` stays green while the top of the funnel
    is stopped. Reads only the durable artifacts the loops already write — never systemd.

    Exits 0 always: this is an operator VIEW, not a watchdog gate (``fleet health`` is the gate).
    ``ok: false`` in the payload carries the verdict.
    """
    settings = get_settings()
    emit(ok(loop_status(settings.data_dir, get_calendar(), now=datetime.now(UTC))))


@book_app.command("status")
@json_errors
def book_status_cmd() -> None:
    """Active capital slices, plus every operational strategy holding NO slice.

    An operational strategy without an allocation is skipped by the operator loop forever and
    surfaces only as an unexplained ``idle`` — the standing condition that stranded
    ``liquidity_stable_quality_momentum`` after a `dormant` round-trip.

    Capital headroom is deliberately absent: it needs the account equity, and the only path to that
    is a broker call this view must not make. Count headroom is reported instead.
    """
    with registry_conn() as conn:
        payload = book_status(conn, capacity=get_settings().paper_book_capacity)
    emit(ok(payload))
