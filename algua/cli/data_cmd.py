from __future__ import annotations

from pathlib import Path

import typer

from algua.cli._common import now_iso
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.contracts import BarProvider, BarRequest
from algua.data.providers.alpaca import AlpacaBarProvider
from algua.data.providers.yfinance import YFinanceBarProvider
from algua.data.store import DataStore

data_app = typer.Typer(help="Point-in-time data snapshots", no_args_is_help=True)
app.add_typer(data_app, name="data")

FROM_FILE_OPTION = typer.Option(..., "--from-file", help="local data file to snapshot")


def _store() -> DataStore:
    return DataStore(get_settings().data_dir)


def _bar_provider(name: str) -> BarProvider:
    settings = get_settings()
    if name == "yfinance":
        return YFinanceBarProvider()
    if name == "alpaca":
        if settings.alpaca_api_key is None or settings.alpaca_api_secret is None:
            raise ValueError(
                "alpaca provider requires ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET"
            )
        return AlpacaBarProvider(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            base_url=settings.alpaca_data_url,
        )
    raise ValueError(f"unsupported bar provider: {name}")


@data_app.command("ingest")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest(
    dataset: str,
    provider: str = typer.Option(..., "--provider"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    start: str = typer.Option(..., "--start", help="inclusive YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="inclusive YYYY-MM-DD"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option(..., "--source", help="source/provenance label"),
    from_file: Path = FROM_FILE_OPTION,
) -> None:
    """Register a local immutable data snapshot."""
    rec = _store().ingest_file(
        dataset=dataset,
        provider=provider,
        symbols=symbols.split(","),
        start=start,
        end=end,
        as_of=as_of,
        source=source,
        file_path=from_file,
    )
    emit({"ok": True, "snapshot": rec.to_dict()})


@data_app.command("ingest-bars")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_bars(
    provider: str = typer.Option("yfinance", "--provider"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    start: str = typer.Option(..., "--start", help="inclusive provider start date/datetime"),
    end: str = typer.Option(..., "--end", help="exclusive/ provider end date/datetime"),
    timeframe: str = typer.Option("1d", "--timeframe"),
    adjustment: str = typer.Option("none", "--adjustment"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
) -> None:
    """Fetch provider bars and persist a reproducible parquet snapshot."""
    clean_symbols = symbols.split(",")
    request = BarRequest(
        symbols=tuple(s.strip().upper() for s in clean_symbols if s.strip()),
        start=start,
        end=end,
        timeframe=timeframe,
        adjustment=adjustment,
    )
    result = _bar_provider(provider).get_bars(request)
    rec = _store().ingest_bars(
        provider=provider,
        symbols=list(request.symbols),
        start=start,
        end=end,
        as_of=as_of or now_iso(),
        source=provider,
        frame=result.frame,
        timeframe=timeframe,
        adjustment=adjustment,
        source_metadata=result.source_metadata,
    )
    emit({"ok": True, "snapshot": rec.to_dict()})


@data_app.command("ingest-universe")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_universe(
    universe: str,
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    effective_date: str = typer.Option(..., "--effective-date", help="YYYY-MM-DD"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option("manual", "--source"),
) -> None:
    """Persist a point-in-time universe membership snapshot."""
    rec = _store().ingest_universe(
        universe=universe,
        symbols=symbols.split(","),
        effective_date=effective_date,
        as_of=as_of or now_iso(),
        source=source,
    )
    emit({"ok": True, "snapshot": rec.to_dict()})


@data_app.command("inspect")
@json_errors(ValueError, LookupError, FileNotFoundError)
def inspect(
    dataset: str = typer.Option(None, "--dataset", help="filter by dataset"),
    snapshot_id: str = typer.Option(None, "--snapshot-id", help="show one snapshot"),
    summary: bool = typer.Option(False, "--summary", help="summarize available datasets"),
) -> None:
    """Inspect recorded point-in-time data snapshots."""
    ds = _store()
    if summary:
        emit(ds.summary())
        return
    if snapshot_id is not None:
        emit(ds.get_snapshot(snapshot_id).to_dict())
        return
    emit([rec.to_dict() for rec in ds.list_snapshots(dataset)])
