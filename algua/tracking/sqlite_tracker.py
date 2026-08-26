"""SQLite-backed MLflow tracker (mlflow filestore deprecation, #605).

MLflow's filesystem tracking backend (a bare ``./mlruns`` directory) is deprecated as of
February 2026 and prints a ``FutureWarning`` on every run — see
``mlflow.store.tracking.file_store.FileStore.__init__``. MLflow's own migration guidance is to
move to a database backend, e.g. ``sqlite:///mlflow.db``.

:class:`SqliteMlflowTracker` is a second, additive :class:`~algua.tracking.base.ExperimentTracker`
implementation, selected the same way ``mlflow`` and ``noop`` are (``ALGUA_TRACKING_BACKEND``). It
reuses the exact same MLflow logging logic as :class:`~algua.tracking.mlflow_tracker.MlflowTracker`
— the two differ only in what ``tracking_uri`` string ends up passed to ``mlflow.set_tracking_uri``.
A bare, schemeless ``tracking_uri`` (the ``ALGUA_MLFLOW_TRACKING_URI`` default, ``"mlruns"``) is
adapted into a same-stemmed ``sqlite:///*.db`` URI; a ``tracking_uri`` that already names a scheme
(``sqlite://``, ``postgresql://``, ``http(s)://``, ...) is passed through unchanged, so an operator
who has already pointed the setting at a database backend is never second-guessed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algua.backtest.result import BacktestResult
from algua.backtest.sweep import SweepResult
from algua.backtest.walkforward import WalkForwardResult
from algua.tracking.mlflow_tracker import log_backtest, log_sweep, log_walk_forward


def _sqlite_tracking_uri(tracking_uri: str) -> str:
    """Adapt a bare filesystem ``tracking_uri`` into a ``sqlite:///`` URI.

    ``"mlruns"`` -> ``"sqlite:///mlruns.db"`` (the FileStore replacement MLflow's own deprecation
    warning recommends). Already-schemed values (anything containing ``"://"``) are returned as-is.
    """
    if "://" in tracking_uri:
        return tracking_uri
    path = Path(tracking_uri)
    db_path = path if path.suffix == ".db" else path.with_name(path.name + ".db")
    return f"sqlite:///{db_path}"


class SqliteMlflowTracker:
    """The SQLite-backed :class:`~algua.tracking.base.ExperimentTracker`. Selected with
    ``ALGUA_TRACKING_BACKEND=mlflow-sqlite`` — replaces the deprecated MLflow FileStore with
    MLflow's SQLite store without changing any of the logging behaviour ``mlflow`` already has."""

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return log_backtest(result, params, tracking_uri=_sqlite_tracking_uri(tracking_uri))

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str:
        return log_sweep(result, tracking_uri=_sqlite_tracking_uri(tracking_uri))

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return log_walk_forward(result, params, tracking_uri=_sqlite_tracking_uri(tracking_uri))
