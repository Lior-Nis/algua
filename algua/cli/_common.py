"""Shared CLI helpers — the one place command modules reach for connection, time, and
evaluation-input boilerplate.

This exists so command modules stop importing each other's private helpers (a cross-module
private-import smell): the public names here are the sanctioned shared surface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime

from algua.backtest._sample import SyntheticProvider
from algua.config.settings import get_settings
from algua.contracts.types import DataProvider
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.registry.db import connect, migrate
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_strategy


def ok(data: dict) -> dict:
    """Stamp a success payload with the ``ok: true`` discriminator.

    CLI JSON-envelope convention: every object-shaped *success* payload carries ``"ok": true`` as
    its first key, mirroring the ``{"ok": false, "error": ...}`` failure envelope (see
    ``cli.errors.json_errors`` and ``cli.main.main``). Callers that emit a JSON *array*
    (``registry list``, ``data inspect``) are the explicit exception: they stay bare arrays.
    """
    return {"ok": True, **data}


@contextmanager
def registry_conn() -> Iterator[sqlite3.Connection]:
    """Yield a migrated registry connection, closed on exit.

    The single idiom for opening the registry DB: connect + migrate + auto-close. Replaces the
    two competing forms (a bare ``_conn()`` and inline ``closing(connect(...))`` + ``migrate``).
    """
    conn = connect(get_settings().db_path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()


def now_iso() -> str:
    """Current UTC instant as an ISO-8601 string — the shared 'now' for persisted timestamps."""
    return datetime.now(UTC).isoformat()


def utc(date_str: str) -> datetime:
    """Parse an ISO date/datetime string and stamp it UTC."""
    return datetime.fromisoformat(date_str).replace(tzinfo=UTC)


def select_provider(demo: bool, snapshot: str | None) -> DataProvider:
    """Pick the data provider from the mutually-exclusive --demo / --snapshot flags."""
    if demo and snapshot:
        raise ValueError("pass only one of --demo or --snapshot")
    if demo:
        return SyntheticProvider(seed=0)
    if snapshot:
        return StoreBackedProvider(DataStore(get_settings().data_dir), snapshot)
    raise ValueError("pass one of --demo (synthetic) or --snapshot <id> (real data)")


def resolve_eval_inputs(
    name: str, demo: bool, snapshot: str | None, start: str, end: str
) -> tuple[LoadedStrategy, DataProvider, datetime, datetime]:
    """Resolve the shared backtest-family preamble: load the strategy, pick the provider, and
    parse the period. Returns ``(strategy, provider, start_dt, end_dt)``."""
    strategy = load_strategy(name)
    provider = select_provider(demo, snapshot)
    return strategy, provider, utc(start), utc(end)


def resolve_delisting_inputs(
    delistings_name: str | None, end_dt: datetime
) -> tuple[Mapping[str, list] | None, str | None]:
    """Resolve opt-in delisting records as-of end_dt (mirror of resolve_universe_inputs).

    ``delistings_name is None`` (no ``--delistings``) => ``(None, None)``.
    Returns ``(records, snapshot_id)`` where ``snapshot_id`` is the ACTUAL snapshot selected
    (not the user-supplied name label) for truthful provenance stamping.
    Raises ``ValueError`` if no delistings snapshot is effective on or before ``end_dt``.
    """
    if delistings_name is None:
        return None, None
    store = DataStore(get_settings().data_dir)
    records = store.read_delistings(as_of=end_dt.isoformat())
    if not records:
        raise ValueError(
            f"--delistings {delistings_name!r}: no delistings snapshot effective on or before "
            f"{end_dt.date().isoformat()}"
        )
    snapshot_id = store.latest_delistings_snapshot_id(as_of=end_dt.isoformat())
    return records, snapshot_id


def resolve_universe_inputs(
    universe_name: str | None, start_dt: datetime, end_dt: datetime
) -> tuple[Mapping[date, Collection[str]] | None, list[dict[str, str]] | None]:
    """Resolve the opt-in point-in-time universe for a backtest-family command.

    `universe_name is None` (no `--universe`) => static mode: returns ``(None, None)`` and the
    engine fetches/shows the strategy's declared universe unchanged.

    Otherwise reads the named universe's membership timeline from the `DataStore`, restricts it to
    snapshots effective on or before `end_dt` (so the union fetched for bars never includes a
    member that only becomes effective after the backtest window — and the as-of resolution at any
    `t <= end_dt` is unaffected), and returns:
      * a sparse ``{effective_date: symbols}`` map the engine resolves as-of-t (greatest
        effective_date <= t; empty before the earliest), and
      * the provenance list ``[{"snapshot_id", "effective_date"}, ...]`` for the result JSON.
    Raises ``ValueError`` if the universe has no membership effective by `end_dt`.
    """
    if universe_name is None:
        return None, None
    timeline = DataStore(get_settings().data_dir).read_universe(universe_name)
    end_date = end_dt.date()
    in_window = [snap for snap in timeline if snap.effective_date <= end_date]
    if not in_window:
        raise ValueError(
            f"universe {universe_name!r} has no membership effective on or before "
            f"{end_date.isoformat()}; ingest a snapshot with --effective-date <= end"
        )
    universe_by_date: dict[date, Collection[str]] = {
        snap.effective_date: snap.symbols for snap in in_window
    }
    provenance = [
        {"snapshot_id": snap.snapshot_id, "effective_date": snap.effective_date.isoformat()}
        for snap in in_window
    ]
    return universe_by_date, provenance
