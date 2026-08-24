"""``run_backtest_task`` -- the ``backtest run`` task body, shared with the ``research run-all``
batch worker (#326) and the merge-back authoritative-evidence seam (#485/#550) — see
``algua.registry.mergeback_intake``.

Moved out of ``algua.cli.backtest_cmd`` so ``paper_cmd``'s merge-back saga can reach the REAL
backtest body via a legal static import instead of a dynamic ``importlib`` dodge around the
cli-independence contract (issue #165): this package is importable by both ``cli`` and ``registry``
without either importing the other (see ``algua/evaluation/__init__.py``), so it must not import
``algua.cli`` itself. ``run_backtest_task`` opens its own registry connection, re-syncs the kb doc,
writes the series file, and records tracking exactly as it did in ``backtest_cmd``, via the shared
``algua.registry.db.registry_conn``, ``algua.registry.kb_sync.sync_kb_doc``,
``algua.evaluation.series.emit_series_file``, and ``algua.tracking.record.record_tracking`` — the
owning leaves — so none of those idioms is duplicated here.
"""

from __future__ import annotations

from pathlib import Path

from algua.backtest.engine import run as run_backtest
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.evaluation.inputs import (
    resolve_delisting_inputs,
    resolve_eval_inputs,
    resolve_universe_inputs,
)
from algua.evaluation.series import emit_series_file
from algua.registry.db import registry_conn
from algua.registry.kb_sync import sync_kb_doc
from algua.registry.runs import record_backtest_run
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from algua.tracking.factory import get_tracker
from algua.tracking.record import record_tracking


def run_backtest_task(  # noqa: PLR0913
    name: str, *, start: str = "2023-01-01", end: str = "2023-12-31", demo: bool = False,
    snapshot: str | None = None, universe: str | None = None,
    fundamentals_snapshot: str | None = None, news_snapshot: str | None = None,
    delistings: str | None = None, assume_terminal_last_close: bool = False,
    register: bool = False, emit_series: str | None = None, track: bool = False,
    reload: bool = False,
) -> dict:
    """Backtest a strategy and return the result payload dict (the body of ``backtest run``).

    Shared by the ``backtest run`` typer command and the ``research run-all`` batch worker (#326)
    so there is exactly ONE backtest code path. Opens+closes its own ``registry_conn()`` and takes
    NO caller-owned connection (each call is a self-contained unit — the batch worker never wraps
    the loop in one connection, so per-task transaction contracts stay intact). ``track`` runs the
    best-effort MLflow log in-place (the batch worker never passes it, so a warm run-all never
    leaks an active MLflow run). ``reload`` force-reloads the strategy module (warm-worker state
    hygiene, #326)."""
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(
        name, demo, snapshot, start, end, reload=reload)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    delisting_records, delisting_snapshot_id = resolve_delisting_inputs(delistings, end_dt)
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
    result = run_backtest(
        strategy, provider, start_dt, end_dt,
        universe_by_date=universe_by_date,
        universe_name=universe, universe_snapshots=universe_prov,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
        delisting_records=delisting_records,
        delisting_snapshot=delisting_snapshot_id,
        assume_terminal_last_close=assume_terminal_last_close,
    )

    # Record the evaluation as a first-class run row. UNCONDITIONAL — including for a
    # not-yet-registered strategy, the same rationale record_search_breadth documents: keying by
    # name means pre-registration evidence still counts. Own transaction, like the sibling writes.
    with registry_conn() as conn:
        record_backtest_run(
            SqliteStrategyRepository(conn), name, result, params=strategy.config.params)

    if register:
        with registry_conn() as conn:
            repo = SqliteStrategyRepository(conn)
            existing = {s.name for s in repo.list_strategies()}
            if name not in existing:
                repo.add(name)
            reason = (
                f"backtest sharpe={result.metrics['sharpe']:.2f} "
                f"ret={result.metrics['total_return']:.2%}"
            )
            transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT, reason)
        # Re-sync the kb doc to the new `backtested` stage (#331): best-effort, out-of-transaction.
        sync_kb_doc(name)

    # Persist return series for the return-correlation clustering axis (#222, Task 7).
    # Only persists for registered strategies; silently skips otherwise.
    if result.returns is not None:
        with registry_conn() as conn:
            repo = SqliteStrategyRepository(conn)
            try:
                repo.get(name)
            except Exception:  # noqa: BLE001 — strategy not yet registered, skip
                pass
            else:
                repo.persist_backtest_returns(
                    name,
                    start_dt.date().isoformat(),
                    end_dt.date().isoformat(),
                    result.returns,
                )

    payload = result.to_dict()
    if emit_series:
        payload["series"] = emit_series_file(result, Path(emit_series))
    if track:
        record_tracking(payload, lambda: get_tracker().log_backtest(
            result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
        ))
    return payload
