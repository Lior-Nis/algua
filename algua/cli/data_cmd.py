from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import typer

from algua.cli._common import now_iso, ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.contracts import BarProvider, BarRequest, ImportRequest
from algua.data.hindsight import query_fundamentals
from algua.data.importers import get_importer
from algua.data.providers import get_provider
from algua.data.store import IMPORT_WARN_ROWS, DataStore, normalize_symbols

data_app = typer.Typer(help="Point-in-time data snapshots", no_args_is_help=True)
app.add_typer(data_app, name="data")

FROM_FILE_OPTION = typer.Option(..., "--from-file", help="local data file to snapshot")


def _store() -> DataStore:
    return DataStore(get_settings().data_dir)


def _bar_provider(name: str) -> BarProvider:
    return get_provider(name, get_settings())


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
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("ingest-bars")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_bars(
    provider: str = typer.Option("yfinance", "--provider"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    start: str = typer.Option(..., "--start", help="inclusive provider start date/datetime"),
    end: str = typer.Option(
        ...,
        "--end",
        help=(
            "end date/datetime. Canonical convention is half-open [start, end). NOTE: vendors "
            "differ — yfinance treats end exclusive (matches the convention); Alpaca treats end "
            "inclusive, so an Alpaca snapshot may include the end bar."
        ),
    ),
    timeframe: str = typer.Option("1d", "--timeframe"),
    adjustment: str = typer.Option("none", "--adjustment"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
) -> None:
    """Fetch provider bars and persist a reproducible parquet snapshot."""
    request = BarRequest(
        symbols=tuple(normalize_symbols(symbols.split(","))),
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
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("import-bars")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_bars(
    vendor: str = typer.Option(..., "--vendor", help="bulk-file vendor, e.g. firstrate"),
    raw_dir: Path = typer.Option(..., "--raw-dir", help="dir of unadjusted per-symbol files"),
    adjusted_dir: Path = typer.Option(
        ..., "--adjusted-dir", help="dir of adjusted per-symbol files (supplies adj_close)"
    ),
    timeframe: str = typer.Option("1d", "--timeframe"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    adjustment: str = typer.Option(
        "split_div", "--adjustment", help="operator-declared adjusted-file flavor (recorded as-is)"
    ),
    start: str = typer.Option(None, "--start", help="optional requested coverage start YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="optional requested coverage end YYYY-MM-DD"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """Import local vendor bar files into one consolidated, normalized bars snapshot."""
    importer = get_importer(vendor)
    request = ImportRequest(
        raw_dir=raw_dir,
        adjusted_dir=adjusted_dir,
        timeframe=timeframe,
        as_of=as_of,
        adjustment=adjustment,
        symbols=tuple(normalize_symbols(symbols.split(","))) if symbols else None,
    )
    store = _store()
    store.clear_staging()
    chunks = importer.import_bars(request)
    # seen_symbols is populated lazily by _tracked() as chunks stream; ingest_bars_streamed reads
    # it only after exhausting the stream (it builds metadata post-loop), so it is complete by then.
    seen_symbols: list[str] = []

    def _tracked() -> Iterator[object]:
        for chunk in chunks:
            seen_symbols.extend(str(s) for s in chunk.frame["symbol"].unique())
            yield chunk.frame

    rec = store.ingest_bars_streamed(
        provider=vendor,
        symbols=seen_symbols,
        as_of=as_of,
        source=f"{vendor}-import",
        chunks=_tracked(),
        timeframe=timeframe,
        adjustment=adjustment,
        start=start,
        end=end,
        source_metadata={
            "vendor": importer.vendor_label,
            "raw_dir": raw_dir.name,
            "adjusted_dir": adjusted_dir.name,
        },
    )
    if rec.row_count is not None and rec.row_count >= IMPORT_WARN_ROWS:
        # stderr: non-fatal advisory; the snapshot is valid, just not servable by the current
        # read path
        typer.echo(
            f"warning: imported {rec.row_count} rows; snapshot not servable by the current "
            f"read path until #130 (marked servable=deferred-130)",
            err=True,
        )
    emit(ok({"snapshot": rec.to_dict()}))


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
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("ingest-fundamentals")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_fundamentals(
    provider: str = typer.Option(..., "--provider"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option(..., "--source", help="source/provenance label"),
    from_file: Path = FROM_FILE_OPTION,
) -> None:
    """Ingest a local tidy fundamentals file (CSV/parquet) as one validated snapshot."""
    path = from_file.expanduser()
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    rec = _store().ingest_fundamentals(
        provider=provider,
        symbols=normalize_symbols(symbols.split(",")),
        as_of=as_of,
        source=source,
        frame=raw,
    )
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("query-fundamentals")
@json_errors(ValueError, LookupError, FileNotFoundError)
def query_fundamentals_cmd(
    snapshot_id: str = typer.Option(..., "--snapshot-id"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """HINDSIGHT fundamentals read (full history) — the agent's post-mortem/analysis surface."""
    syms = normalize_symbols(symbols.split(",")) if symbols else None
    frame = query_fundamentals(_store(), snapshot_id, symbols=syms)
    records = [
        {
            "symbol": row.symbol,
            "fiscal_period_end": row.fiscal_period_end.isoformat(),
            "metric": row.metric,
            "value": None if pd.isna(row.value) else float(row.value),
            "knowable_at": row.knowable_at.isoformat(),
            "source": row.source,
        }
        for row in frame.itertuples(index=False)
    ]
    emit(records)


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
