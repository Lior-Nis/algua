"""Point-in-time (PIT) masking views for the backtest engine.

Given a timestamp, these functions expose only the universe membership, fundamentals, and
news that were knowable at that instant. This IS the anti-look-ahead enforcement: an as-of
mask is the difference between a legitimate backtest and one that peeks. Moved verbatim out
of `algua.backtest.engine` (which stays the CODEOWNERS-protected consumer); this module is
independently CODEOWNERS-protected for the same reason (see `INTEGRITY_CRITICAL_MODULES` in
`tests/test_repo_hygiene.py`).

`BacktestError` lives in `algua.backtest.errors` — the lane's error leaf — because it is raised
and caught across five packages; see that module for why it is not defined in a module that also
does work.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date

import pandas as pd

from algua.backtest.errors import BacktestError
from algua.contracts.types import (
    FUNDAMENTALS_AS_OF_KEY,
    FUNDAMENTALS_COLUMNS,
    FUNDAMENTALS_KNOWABLE_AT,
    NEWS_AS_OF_KEY,
    NEWS_COLUMNS,
    NEWS_KNOWABLE_AT,
    NEWS_RETRACTED,
)
from algua.strategies.base import LoadedStrategy


def members_as_of(
    universe_by_date: Mapping[date, Collection[str]], t: pd.Timestamp
) -> frozenset[str]:
    """As-of-t membership: the snapshot with the greatest effective_date <= t.date().

    The map is keyed by effective_date and may be sparse (the CLI wiring passes one entry per
    universe snapshot, not per session); the as-of rule holds either way. Empty before the
    earliest effective date. Uses only dates <= t, so membership at t can never see a later
    snapshot — no look-ahead.
    """
    target = t.date()
    eligible = [d for d in universe_by_date if d <= target]
    if not eligible:
        return frozenset()
    return frozenset(universe_by_date[max(eligible)])



def _assert_fundamentals_shape(frame: pd.DataFrame) -> None:
    """Structural defense at the engine seam (no algua.data import): a foreign
    FundamentalsProvider must hand back contract-shaped, UTC, unique-keyed data.
    Store-backed reads already validate; this fails closed for any other provider (spec §2.1)."""
    missing = [c for c in FUNDAMENTALS_COLUMNS if c not in frame.columns]
    if missing:
        raise BacktestError(f"fundamentals frame missing columns {missing}")
    ka = frame[FUNDAMENTALS_KNOWABLE_AT]
    if not isinstance(ka.dtype, pd.DatetimeTZDtype) or str(ka.dt.tz) != "UTC":
        raise BacktestError("fundamentals 'knowable_at' must be tz-aware UTC")
    if ka.isna().any():
        raise BacktestError("fundamentals 'knowable_at' must not be null")
    if str(frame["value"].dtype) != "float64":
        raise BacktestError("fundamentals 'value' must be float64")
    key = [*FUNDAMENTALS_AS_OF_KEY, FUNDAMENTALS_KNOWABLE_AT]
    if frame[key].duplicated().any():
        raise BacktestError(
            "fundamentals has duplicate (symbol, fiscal_period_end, metric, knowable_at) rows"
        )


def _fundamentals_as_of(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """As-of-t fundamentals: of the rows with knowable_at <= t, keep for each
    (symbol, fiscal_period_end, metric) the row with the greatest knowable_at (latest revision
    knowable by t). knowable_at is unique per key within a snapshot, so the pick is deterministic.
    Uses only knowable_at <= t -> no look-ahead. Empty in/empty out (returns a 0-row slice, never a
    view into future rows)."""
    if t.tz is None:
        raise BacktestError("fundamentals as-of mask requires a tz-aware (UTC) timestamp t")
    visible = frame[frame[FUNDAMENTALS_KNOWABLE_AT] <= t]
    if visible.empty:
        return frame.iloc[0:0].copy()
    ordered = visible.sort_values(FUNDAMENTALS_KNOWABLE_AT, kind="stable")
    latest = ordered.drop_duplicates(subset=list(FUNDAMENTALS_AS_OF_KEY), keep="last")
    return latest.reset_index(drop=True)


def _assert_news_shape(frame: pd.DataFrame) -> None:
    """Structural defense at the engine seam (no algua.data import): a foreign NewsProvider must
    hand back contract-shaped, UTC, unique-keyed data. Store-backed reads already validate; this
    fails closed for any other provider (spec §5)."""
    missing = [c for c in NEWS_COLUMNS if c not in frame.columns]
    if missing:
        raise BacktestError(f"news frame missing columns {missing}")
    for col in (NEWS_KNOWABLE_AT, "published_at"):
        ts = frame[col]
        if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
            raise BacktestError(f"news {col!r} must be tz-aware UTC")
        if ts.isna().any():
            raise BacktestError(f"news {col!r} must not be null")
    if (frame[NEWS_KNOWABLE_AT].to_numpy() < frame["published_at"].to_numpy()).any():
        raise BacktestError("news 'knowable_at' must be >= 'published_at'")
    if str(frame[NEWS_RETRACTED].dtype) != "bool":
        raise BacktestError("news 'retracted' must be non-nullable bool")
    key = [*NEWS_AS_OF_KEY, NEWS_KNOWABLE_AT]
    if frame[key].duplicated().any():
        raise BacktestError("news has duplicate (source, article_id, symbol, knowable_at) rows")


def _news_as_of(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """As-of-t news: of the rows with knowable_at <= t, keep for each (source, article_id, symbol)
    the latest revision (greatest knowable_at), then DROP retraction tombstones. knowable_at is
    unique per key within a snapshot, so the pick is deterministic. Uses only knowable_at <= t ->
    no look-ahead. Empty-in/empty-out returns a 0-row slice (preserves dtypes)."""
    if t.tz is None:
        raise BacktestError("news as-of mask requires a tz-aware (UTC) timestamp t")
    visible = frame[frame[NEWS_KNOWABLE_AT] <= t]
    if visible.empty:
        return frame.iloc[0:0].copy()
    ordered = visible.sort_values(NEWS_KNOWABLE_AT, kind="stable")
    latest = ordered.drop_duplicates(subset=list(NEWS_AS_OF_KEY), keep="last")
    live = latest[~latest[NEWS_RETRACTED]]
    return live.reset_index(drop=True)


def _static_operating_view(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project the strategy-visible STATIC view to the operating universe (declared AND available)
    so a misbehaving provider's undeclared symbols never reach the loop view, the fast-path
    signal_panel, the weights/grid, or the fundamentals/news sidecars (observation parity, #208).

    Fails closed when no declared symbol has fetched price data (empty operating universe) — this
    absorbs #179's empty-intersection guard AND the empty-declared-universe case. The projection on
    `adj` is COLUMN-ONLY (adj.index is untouched), so holdout_window's grid and the #192 single-use
    holdout identity are unaffected. No-op for a compliant provider (adj.columns within universe).
    """
    universe = set(strategy.universe)
    # Order-preserving intersection: keep adj's existing column order so a compliant provider is a
    # STRICT no-op (no reorder, no NaN/reindex-fill since operating is a subset of adj.columns).
    operating = [c for c in adj.columns if c in universe]
    if not operating:
        raise BacktestError(
            f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
            f"universe {sorted(strategy.universe)} (fetched columns: "
            f"{sorted(map(str, adj.columns))})"
        )
    return bars[bars["symbol"].isin(operating)], adj.loc[:, operating]
