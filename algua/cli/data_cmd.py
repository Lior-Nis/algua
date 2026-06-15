from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import typer

from algua.cli._common import now_iso, ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.data.constituents import constituents_to_snapshots, parse_constituents_rows
from algua.data.contracts import (
    BarProvider,
    BarRequest,
    DatabentoImportRequest,
    FirstRateImportRequest,
    ImportRequest,
)
from algua.data.hindsight import query_fundamentals, query_news
from algua.data.importers import get_importer
from algua.data.providers import get_provider
from algua.data.store import DataStore, normalize_symbols

data_app = typer.Typer(help="Point-in-time data snapshots", no_args_is_help=True)
app.add_typer(data_app, name="data")

FROM_FILE_OPTION = typer.Option(..., "--from-file", help="local data file to snapshot")


def _store() -> DataStore:
    return DataStore(get_settings().data_dir)


def _bar_provider(name: str) -> BarProvider:
    return get_provider(name, get_settings())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
        None, "--adjusted-dir", help="firstrate: dir of adjusted per-symbol files (adj_close)"
    ),
    corp_actions: Path = typer.Option(
        None, "--corp-actions",
        help="databento: parquet of split/dividend events (computes adj_close)",
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
    sym_tuple = tuple(normalize_symbols(symbols.split(","))) if symbols else None
    if vendor == "firstrate":
        if adjusted_dir is None:
            raise ValueError("firstrate import requires --adjusted-dir")
        if corp_actions is not None:
            raise ValueError("firstrate import does not use --corp-actions")
        request: ImportRequest = FirstRateImportRequest(
            raw_dir=raw_dir, adjusted_dir=adjusted_dir, timeframe=timeframe, as_of=as_of,
            adjustment=adjustment, symbols=sym_tuple,
        )
        source_metadata = {
            "vendor": importer.vendor_label,
            "raw_dir": raw_dir.name,
            "adjusted_dir": adjusted_dir.name,
        }
    elif vendor == "databento":
        if corp_actions is None:
            raise ValueError("databento import requires --corp-actions")
        if adjusted_dir is not None:
            raise ValueError("databento import does not use --adjusted-dir")
        request = DatabentoImportRequest(
            raw_dir=raw_dir, corp_actions_path=corp_actions, timeframe=timeframe, as_of=as_of,
            adjustment=adjustment, symbols=sym_tuple,
        )
        source_metadata = {
            "vendor": importer.vendor_label,
            "raw_dir": raw_dir.name,
            "corp_actions_file": corp_actions.name,
            "corp_actions_sha256": _sha256(corp_actions),
            "ca_schema_version": "1",
        }
    else:
        raise ValueError(f"vendor {vendor!r} has no import-bars flag wiring")
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
        source_metadata=source_metadata,
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


@data_app.command("import-universe")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_universe(
    universe: str,
    file: Path = typer.Option(..., "--file", help="constituents CSV: symbol,add_date,drop_date"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option("bulk-import", "--source"),
) -> None:
    """Bulk-import a constituents CSV into the universe-snapshot timeline (one snapshot per change
    date; add inclusive, drop exclusive, multiple rows/symbol for re-additions). Universes are
    IMMUTABLE: a same-date membership conflict aborts before any write (corrections need a new
    name). Empty-membership change dates are rejected (deferred limitation)."""
    with file.expanduser().open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    intervals = parse_constituents_rows(rows)
    timeline = constituents_to_snapshots(intervals)
    stamp = as_of or now_iso()
    store = _store()
    symbols_seen: set[str] = set()
    for effective_date, members in timeline:
        if not members:
            raise ValueError(
                f"universe {universe!r}: membership is empty on {effective_date.isoformat()} "
                "— empty-membership snapshots are a deferred limitation"
            )
        symbols_seen.update(members)
        store.ingest_universe(
            universe=universe,
            symbols=sorted(members),
            effective_date=effective_date.isoformat(),
            as_of=stamp,
            source=source,
            require_immutable=True,
        )
    emit(ok({
        "universe": universe,
        "snapshots_written": len(timeline),
        "change_dates": [d.isoformat() for d, _ in timeline],
        "symbols_seen": sorted(symbols_seen),
    }))


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


@data_app.command("ingest-news")
@json_errors(ValueError, LookupError, FileNotFoundError)
def ingest_news(
    provider: str = typer.Option(..., "--provider"),
    as_of: str = typer.Option(..., "--as-of", help="point-in-time ISO datetime"),
    from_file: Path = FROM_FILE_OPTION,
) -> None:
    """Ingest a local news file (CSV/parquet) as one validated snapshot (hindsight lane).

    Rows carry: source, article_id, symbols, published_at, knowable_at, headline, [url], [body].
    `source` is a required per-row column and the covered symbol/source sets are derived, so there
    is no --source/--symbols flag; --provider is the ingest label."""
    path = from_file.expanduser()
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    rec = _store().ingest_news(provider=provider, as_of=as_of, frame=raw)
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("import-delistings")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_delistings(
    file: Path = typer.Option(..., "--file", help="CSV: symbol,delisting_date,delisting_value"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option("vendor", "--source"),
) -> None:
    """Import a delistings CSV (delisting_value = per-share terminal price in adj_close units,
    strictly > 0) as one point-in-time delistings snapshot."""
    frame = pd.read_csv(file.expanduser())
    rec = _store().ingest_delistings(frame=frame, as_of=as_of or now_iso(), source=source)
    emit(ok({"snapshot": rec.to_dict()}))


@data_app.command("query-news")
@json_errors(ValueError, LookupError, FileNotFoundError)
def query_news_cmd(
    snapshot_id: str = typer.Option(..., "--snapshot-id"),
    symbols: str = typer.Option(None, "--symbols", help="optional comma-separated subset"),
) -> None:
    """HINDSIGHT news read (full history) — the agent's post-mortem/analysis surface."""
    syms = normalize_symbols(symbols.split(",")) if symbols else None
    frame = query_news(_store(), snapshot_id, symbols=syms)
    records = [
        {
            "source": row.source,
            "article_id": row.article_id,
            "symbol": row.symbol,
            "published_at": row.published_at.isoformat(),
            "knowable_at": row.knowable_at.isoformat(),
            "headline": row.headline,
            "url": None if pd.isna(row.url) else str(row.url),
            "body": None if pd.isna(row.body) else str(row.body),
        }
        for row in frame.itertuples(index=False)
    ]
    emit(records)


@data_app.command("verify")
@json_errors(ValueError, LookupError, FileNotFoundError)
def verify(
    snapshot_id: str = typer.Option(None, "--snapshot-id", help="verify one snapshot"),
) -> None:
    """Power-loss backstop (#184): read each snapshot's payload back from disk and check it
    against its record. Reports one row per snapshot and exits non-zero if any failed."""
    results = _store().verify_snapshots(snapshot_id)
    failed = sum(1 for r in results if not r["ok"])
    emit(
        {
            "ok": failed == 0,
            "verified": len(results),
            "failed": failed,
            "snapshots": results,
        }
    )
    raise typer.Exit(code=0 if failed == 0 else 1)


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
