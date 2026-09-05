"""`algua data refresh-bars` — the lane-refresh primitive as a CLI seam (#556). Carved out of
data_cmd.py so that module stays under its size pin; registered onto the same `data_app`.

Resolves `_bar_provider`/`_store` through the `algua.cli.data_cmd` module attribute at call time
(not a direct name import) so `tests/test_cli_data.py`'s monkeypatch of `data_cmd._bar_provider`
still reaches this command.
"""
from __future__ import annotations

import typer

from algua.cli import data_cmd
from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.data_cmd import data_app
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.refresh import refresh_bars


@data_app.command("refresh-bars")
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
        data_cmd._store(), data_cmd._bar_provider(name), symbols=syms, start=start, end=end,
        require_bar_on=require_bar_on,
        min_rows={s.strip().upper(): min_rows for s in syms} if min_rows > 0 else None,
        timeframe=timeframe,
    )
    emit(ok({"snapshot": rec.to_dict(), "refreshed": refreshed}))
