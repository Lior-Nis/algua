"""Tracking-URI adaptation, kept as a pure leaf.

The sqlite backend rewrites a bare filesystem ``tracking_uri`` on the way IN, so anything that
READS the store back has to apply the same rewrite or it opens the wrong path -- and, since MLflow
3.15 turned the FileStore into a hard error, raises rather than returning nothing.

Lives in its own module because its consumers sit on opposite sides of a layering wall:
``algua.tracking.sqlite_tracker`` (which imports the backtest result types) and
``algua.knowledge.metrics`` (which is contractually forbidden from reaching ``algua.backtest``).
A pure string function has no reason to carry the backtest dependency to the second caller.

Imports nothing from algua.
"""

from __future__ import annotations

from pathlib import Path

#: The backend key whose tracking URI is rewritten. See ``algua/tracking/factory.py``.
SQLITE_BACKEND = "mlflow-sqlite"


def sqlite_tracking_uri(tracking_uri: str) -> str:
    """Adapt a bare filesystem ``tracking_uri`` into a ``sqlite:///`` URI.

    ``"mlruns"`` -> ``"sqlite:///mlruns.db"`` (the FileStore replacement MLflow's own deprecation
    warning recommends). Already-schemed values (anything containing ``"://"``) are returned as-is,
    so an operator who already points the setting at a database backend is never second-guessed.
    """
    if "://" in tracking_uri:
        return tracking_uri
    path = Path(tracking_uri)
    db_path = path if path.suffix == ".db" else path.with_name(path.name + ".db")
    return f"sqlite:///{db_path}"


def resolved_tracking_uri(tracking_uri: str, backend: str) -> str:
    """The tracking URI a READER must open to find what ``backend`` wrote."""
    return sqlite_tracking_uri(tracking_uri) if backend == SQLITE_BACKEND else tracking_uri
