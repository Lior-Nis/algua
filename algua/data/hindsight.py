from __future__ import annotations

import pandas as pd

from algua.data.store import DataStore

# The HINDSIGHT lane (issue #132). Returns FULL history regardless of any decision time — for agent
# post-mortems and idea sourcing ONLY. It is never wired into the backtest engine, and the import
# wall (pyproject.toml) forbids algua.backtest / algua.features / algua.contracts / algua.strategies
# from importing this module: hindsight must be structurally unable to reach compute_weights.


def query_fundamentals(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Full-hindsight fundamentals read (no as-of masking). Stable canonical row order for
    reproducible agent diffs (read_fundamentals already returns the canonical sort)."""
    return store.read_fundamentals(snapshot_id, symbols=symbols)


def query_news(
    store: DataStore, snapshot_id: str, symbols: list[str] | None = None
) -> pd.DataFrame:
    """Full-hindsight news read (no as-of masking) — the agent's post-mortem/analysis surface.
    Wraps store.read_news, which returns the canonical sort for reproducible agent diffs."""
    return store.read_news(snapshot_id, symbols=symbols)
