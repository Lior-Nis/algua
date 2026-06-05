"""Shared CLI helpers — the one place command modules reach for connection, time, and
evaluation-input boilerplate.

This exists so command modules stop importing each other's private helpers (a cross-module
private-import smell): the public names here are the sanctioned shared surface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

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
