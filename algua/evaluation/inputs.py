"""Input resolvers for the shared backtest-family evaluation preamble.

These four functions (``select_provider``, ``resolve_eval_inputs``, ``resolve_delisting_inputs``,
``resolve_universe_inputs``) used to live in ``algua/cli/_common.py``. They import ``data``,
``strategies``, ``backtest._sample``, ``config``, ``contracts`` and nothing from ``cli`` or
``registry``, so they were domain code living in the CLI package, and ``registry``/``evaluation``
code cannot reach them where they were. Moving them here lets both ``cli`` and ``registry`` import
them without either importing the other.

``resolve_eval_inputs`` used ``algua.cli._common.utc`` to parse its ``start``/``end`` strings; that
helper stays in ``_common`` (it's genuine shared CLI infrastructure used well beyond these four
functions), so importing it here would pull ``algua.cli`` — and, transitively, ``algua.registry``,
since ``_common`` itself imports ``algua.registry.db`` — right back into this package, defeating the
reason it was split out. ``_utc`` below is a private, byte-identical duplicate of that one two-line
parse used only internally by ``resolve_eval_inputs``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, date, datetime

from algua.backtest._sample import SyntheticProvider
from algua.config.settings import get_settings
from algua.contracts.types import DataProvider
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_strategy


def _utc(date_str: str) -> datetime:
    """Parse an ISO date/datetime string and stamp it UTC (private duplicate of
    ``algua.cli._common.utc`` — see module docstring for why this isn't imported instead)."""
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
    name: str, demo: bool, snapshot: str | None, start: str, end: str, *, reload: bool = False
) -> tuple[LoadedStrategy, DataProvider, datetime, datetime]:
    """Resolve the shared backtest-family preamble: load the strategy, pick the provider, and
    parse the period. Returns ``(strategy, provider, start_dt, end_dt)``.

    ``reload=True`` force-reloads the strategy module (see ``load_strategy``) — used by the
    long-lived ``research run-all`` batch worker (#326) so a warm process does not carry a
    strategy's own module-level state from one task into the next."""
    strategy = load_strategy(name, reload=reload)
    provider = select_provider(demo, snapshot)
    return strategy, provider, _utc(start), _utc(end)


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
    # Single manifest read: records and snapshot_id come from the SAME selected snapshot, so a
    # concurrent ingest can never make the stamped provenance disagree with the loaded records.
    records, snapshot_id = store.read_delistings_with_snapshot(as_of=end_dt.isoformat())
    if not records:
        raise ValueError(
            f"--delistings {delistings_name!r}: no delistings snapshot effective on or before "
            f"{end_dt.date().isoformat()}"
        )
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
