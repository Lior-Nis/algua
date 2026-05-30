from __future__ import annotations

from pathlib import Path

import typer

from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.store import DataStore

data_app = typer.Typer(help="Point-in-time data snapshots", no_args_is_help=True)
app.add_typer(data_app, name="data")

FROM_FILE_OPTION = typer.Option(..., "--from-file", help="local data file to snapshot")


def _store() -> DataStore:
    return DataStore(get_settings().data_dir)


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


@data_app.command("inspect")
@json_errors(ValueError, LookupError, FileNotFoundError)
def inspect(
    dataset: str = typer.Option(None, "--dataset", help="filter by dataset"),
    snapshot_id: str = typer.Option(None, "--snapshot-id", help="show one snapshot"),
) -> None:
    """Inspect recorded point-in-time data snapshots."""
    store = _store()
    if snapshot_id is not None:
        emit(store.get_snapshot(snapshot_id).to_dict())
        return
    emit([rec.to_dict() for rec in store.list_snapshots(dataset)])
