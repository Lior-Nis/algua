"""`algua data refresh-bars` — the lane-refresh primitive as a CLI seam (#556). Carved out of
data_cmd.py so that module stays under its size pin.

Owns its own `_store`/`_bar_provider` seam (mirrors data_cmd.py's, but does not import that
module — a cli->cli sibling import is exactly what
`tests/test_lint_imports.py::"cli command modules are independent of one another"` forbids). It
registers on its own `refresh_app`, which `algua.cli.main` merges flat onto `data_cmd.data_app`
at the composition root (the same pattern main.py already uses to mount `idea_app` under
`research_app`), so `algua data refresh-bars ...` is unchanged.
"""
from __future__ import annotations

import typer

from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.contracts import BarProvider
from algua.data.providers import get_provider
from algua.data.refresh import refresh_bars
from algua.data.store import DataStore

refresh_app = typer.Typer()


def _store() -> DataStore:
    return DataStore(get_settings().data_dir)


def _bar_provider(name: str) -> BarProvider:
    return get_provider(name, get_settings())


@refresh_app.command("refresh-bars")
@json_errors
def refresh_bars_cmd(
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    start: str = typer.Option(..., "--start", help="inclusive start date"),
    end: str = typer.Option(..., "--end", help="exclusive end date (half-open [start, end))"),
    require_bar_on: str = typer.Option(
        ..., "--require-bar-on",
        help=(
            "ISO date every symbol's NEWEST bar must fall on (the session a tick decides on); "
            "older = stale, newer = misdated, absent = missing — each fails closed, nothing minted"
        ),
    ),
    min_rows: int = typer.Option(
        0, "--min-rows", help="minimum in-window rows per symbol (history floor); 0 = none"),
    provider: str = typer.Option(
        None, "--provider", help="bar provider name (default: settings.bars_refresh_provider)"),
    timeframe: str = typer.Option("1d", "--timeframe"),
) -> None:
    """Resolve-or-ingest bars for a lane tick (#556): reuse the newest same-request snapshot if
    it still passes the coverage wall, else fetch, clip, validate, and mint a new one."""
    name = provider or get_settings().bars_refresh_provider
    syms = symbols.split(",")
    rec, refreshed = refresh_bars(
        _store(), _bar_provider(name), symbols=syms, start=start, end=end,
        require_bar_on=require_bar_on,
        min_rows={s.strip().upper(): min_rows for s in syms} if min_rows > 0 else None,
        timeframe=timeframe,
    )
    emit(ok({"snapshot": rec.to_dict(), "refreshed": refreshed}))
