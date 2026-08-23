"""The ``backtest sweep`` task body, shared with the ``research run-all`` batch worker (#326) and
the merge-back authoritative-evidence seam (#485/#550) — see ``algua.registry.mergeback_intake``.

Moved out of ``algua.cli.backtest_cmd`` so ``paper_cmd``'s merge-back saga can reach the REAL sweep
body via a legal static import instead of a dynamic ``importlib`` dodge around the cli-independence
contract (issue #165): this package is importable by both ``cli`` and ``registry`` without either
importing the other (see ``algua/evaluation/__init__.py``), so it must not import ``algua.cli``
itself. ``sweep_task`` opens its own registry connection and records tracking exactly as it did in
``backtest_cmd``, via the shared ``algua.registry.db.registry_conn`` and
``algua.tracking.record.record_tracking`` — the owning leaves — so neither idiom is duplicated here.
"""

from __future__ import annotations

from algua.backtest.sweep import parse_grid, sweep
from algua.config.settings import get_settings
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.evaluation.inputs import (
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.registry.db import registry_conn
from algua.registry.search_breadth import record_search_breadth
from algua.registry.store import SqliteStrategyRepository
from algua.tracking.factory import get_tracker
from algua.tracking.record import record_tracking


def sweep_task(  # noqa: PLR0913
    name: str, *, start: str = "2023-01-01", end: str = "2023-12-31", demo: bool = False,
    snapshot: str | None = None, universe: str | None = None, windows: int = 4,
    holdout_frac: float = 0.2, param: list[str] | None = None, rank_by: str = "mean_sharpe",
    top: int = 20, fundamentals_snapshot: str | None = None, news_snapshot: str | None = None,
    delistings: str | None = None, assume_terminal_last_close: bool = False, track: bool = False,
    reload: bool = False,
) -> dict:
    """Sweep a strategy across a parameter grid and return the (untruncated-summary) payload dict —
    the body of ``backtest sweep``, shared with the ``research run-all`` batch worker (#326).

    Opens+closes its own ``registry_conn()`` (no caller-owned connection). The ``--summary``
    projection stays in the typer wrapper; ``top`` truncation lives here (it is part of the
    recorded payload shape). ``reload`` force-reloads the strategy module (warm-worker hygiene)."""
    if top < 1:
        raise ValueError("--top must be >= 1")
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(
        name, demo, snapshot, start, end, reload=reload)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError(
            "--fundamentals-snapshot was given but the strategy does not declare needs_fundamentals"
        )
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError(
            "--news-snapshot was given but the strategy does not declare needs_news"
        )
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot
        else None
    )
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot
        else None
    )
    delisting_records, _delisting_prov = resolve_delisting_inputs(delistings, end_dt)
    grid = parse_grid(param or [])
    result = sweep(strategy, provider, start_dt, end_dt,
                   grid=grid, windows=windows, holdout_frac=holdout_frac, rank_by=rank_by,
                   universe_by_date=universe_by_date,
                   universe_name=universe, universe_snapshots=universe_prov,
                   fundamentals_provider=fundamentals_provider,
                   news_provider=news_provider,
                   delisting_records=delisting_records,
                   assume_terminal_last_close=assume_terminal_last_close)
    with registry_conn() as conn:
        recorded = record_search_breadth(SqliteStrategyRepository(conn), name, result)
    payload = result.to_dict()
    payload["ranked"] = payload["ranked"][:top]
    # Surface the MEASURED breadth this sweep contributed (this sweep's n_combos) and the
    # cumulative family total now on record, so the operator sees what promotion will read back.
    # Recorded by strategy NAME, so even a sweep of an UNREGISTERED strategy counts.
    payload["recorded_breadth"] = recorded
    if track:
        record_tracking(payload, lambda: get_tracker().log_sweep(
            result, tracking_uri=get_settings().mlflow_tracking_uri
        ))
    return payload
