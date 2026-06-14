from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date, datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

from algua.backtest.metrics import portfolio_metrics
from algua.backtest.result import BacktestResult, config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.types import (
    FUNDAMENTALS_AS_OF_KEY,
    FUNDAMENTALS_COLUMNS,
    FUNDAMENTALS_KNOWABLE_AT,
    NEWS_AS_OF_KEY,
    NEWS_COLUMNS,
    NEWS_KNOWABLE_AT,
    NEWS_RETRACTED,
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
)
from algua.risk.limits import WEIGHT_TOL, RiskBreach, validate_decision_weights
from algua.strategies.base import LoadedStrategy

_SUPPORTED_CADENCES = {"1d"}  # this slice rebalances on every daily bar only

# Fail-closed runtime parity guard: number of post-warmup bars at which the fast path is
# re-verified against the canonical per-bar definition on every run that uses a signal_panel_fn.
# Bounded + deterministic (evenly spread across the evaluated span) so the guard costs O(_PARITY_
# SAMPLE) per-bar evaluations rather than O(n_bars) — preserving the speedup while still catching a
# panel fn that disagrees with its per-bar twin. Full per-bar parity is asserted in CI (test suite).
_PARITY_SAMPLE = 16


class BacktestError(RuntimeError):
    pass


def _members_as_of(
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


def _decision_weights(
    strategy: LoadedStrategy,
    bars: pd.DataFrame,
    adj: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the per-bar decision loop and return raw target weights (pre-lag).

    Path-dependent fallback. A strategy whose `target_weights` only depends on the latest
    fully-closed bar could be vectorized over the full panel; this loop is the general
    contract for path-dependent strategies. It walks the index once, slicing the history
    up to each bar `t` via positional `iloc` (a cheap expanding window) rather than
    re-copying a growing `.loc[:t]` prefix each step.

    The first `warmup_bars` bars are held flat: a strategy may need that much history
    before its signal is meaningful, so the first window excludes the warmup span. This matches
    the paper loop's warm-up exactly (warmup_bars = N => the same N initial bars are flat).

    Each evaluated bar runs the SAME long-only + gross-exposure risk checks as the paper/live
    decision core, so a decision that passes the backtest is one paper/live will also accept.

    Point-in-time universe (`universe_by_date`): when provided, the strategy only ever sees the
    symbols that were as-of-t members (the snapshot with the greatest effective_date <= t),
    eliminating survivorship bias. `None` reproduces the original static behavior exactly (the
    full fetched panel is visible every bar). Empty as-of membership => that bar is flat. A weight
    returned for a non-member is a strategy bug and raises `BacktestError`. As-of membership at t
    uses only snapshots dated <= t, so the masking introduces no look-ahead.
    """
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
    # Static operating universe = declared AND available: a declared symbol with no fetched price
    # column can't be traded here (reindex drops it anyway), and an undeclared column a provider
    # wrongly returned is rejected — so the validated set is provably a subset of strategy.universe.
    static_universe = set(strategy.universe) & set(columns)

    weights = pd.DataFrame(0.0, index=adj.index, columns=columns)
    # Sort the raw bars by timestamp ONCE and precompute, per session, the integer end of the
    # history up to and including that bar. Each step is then a positional iloc slice
    # (cheap) rather than a label-based re-slice of a growing prefix (#36).
    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")
    for i, (t, stop) in enumerate(zip(adj.index, end_pos, strict=True)):
        if i < warmup:
            continue
        view = bars_sorted.iloc[:stop]
        if universe_by_date is not None:
            members = _members_as_of(universe_by_date, t)
            if not members:
                continue  # before the earliest effective date -> flat
            view = view[view["symbol"].isin(members)]
            if view.empty:
                continue  # members exist but no bar data for any of them yet -> flat
        if fundamentals is not None:
            f_asof = _fundamentals_as_of(fundamentals, t)
            allowed = members if universe_by_date is not None else set(columns)
            f_asof = f_asof[f_asof["symbol"].isin(allowed)]
            w = strategy.target_weights(view, fundamentals=f_asof)
        elif news is not None:
            n_asof = _news_as_of(news, t)
            allowed = members if universe_by_date is not None else set(columns)
            n_asof = n_asof[n_asof["symbol"].isin(allowed)]
            w = strategy.target_weights(view, news=n_asof)
        else:
            w = strategy.target_weights(view)
        if len(w) == 0:
            continue
        # The shared checks raise RiskBreach; re-raise as BacktestError for the backtest CLI/error
        # contract while preserving the breach (and its `.kind`) as the cause.
        try:
            decision_universe = members if universe_by_date is not None else static_universe
            validate_decision_weights(
                w, strategy.execution, strategy.name, allowed_symbols=decision_universe
            )
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
        # Label-aligned assignment: place each symbol's weight in its own column rather than
        # trusting positional order (#42).
        weights.loc[t, row.index] = row.to_numpy()
    return weights


def _canonical_row(
    strategy: LoadedStrategy, bars_sorted: pd.DataFrame, stop: int, columns: pd.Index
) -> pd.Series:
    """The canonical per-bar weights = construct(signal(view), view) over the expanding history
    slice ending at (and including) that bar, reindexed onto `columns` and zero-filled. This is the
    SAME computation the loop performs per bar — INCLUDING the shared risk rails — so the fast-path
    parity guard compares against the loop's own definition, not a re-derivation. Running the full
    `validate_decision_weights` here (not just one check) keeps the proxy a FAITHFUL loop-twin with
    identical check ordering, so e.g. an out-of-universe per-bar weight fails closed instead of
    being silently reindex-dropped before the comparison."""
    view = bars_sorted.iloc[:stop]
    w = strategy.target_weights(view)
    if len(w) == 0:
        return pd.Series(0.0, index=columns)
    try:
        validate_decision_weights(
            w, strategy.execution, strategy.name,
            allowed_symbols=set(strategy.universe) & set(columns),
        )
    except RiskBreach as breach:
        raise BacktestError(breach.detail) from breach
    return w.reindex(columns).fillna(0.0)


def _fast_weights(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast-path WEIGHTS, without the bounded runtime parity guard. Calls the strategy's
    `signal_panel` ONCE for the whole period to get the SCORES matrix, then applies the construction
    policy PER BAR (with the same expanding `view_t` the loop uses) + the shared risk walls. Static
    universe only; pre-lag, like the loop.

    The scores matrix is NOT NaN-filled before construction — a missing score means 'no opinion' and
    the policy drops it; only the FINAL weights are zero-filled to flat. The parity guard is applied
    by the caller (`_decision_weights_fast` for the bounded runtime check;
    `verify_signal_panel_parity` for the exhaustive promotion gate), so this function never falls
    back silently."""
    panel = strategy.signal_panel(bars)
    assert panel is not None  # caller guarantees signal_panel_fn is set
    if not isinstance(panel, pd.DataFrame):
        raise BacktestError(
            f"strategy {strategy.name!r} signal_panel returned "
            f"{type(panel).__name__}, expected a DataFrame"
        )
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
    # Static operating universe = declared AND available: a declared symbol with no fetched price
    # column can't be traded here (reindex drops it anyway), and an undeclared column a provider
    # wrongly returned is rejected — so the validated set is provably a subset of strategy.universe.
    static_universe = set(strategy.universe) & set(columns)
    # Reindex the SCORES onto the simulation grid WITHOUT filling NaN (missing score != 0 score).
    scores = panel.reindex(index=adj.index, columns=columns)

    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")

    weights = pd.DataFrame(0.0, index=adj.index, columns=columns)
    for i, (t, stop) in enumerate(zip(adj.index, end_pos, strict=True)):
        if i < warmup:
            continue  # warm-up: held flat by SKIPPING construction (weights stay 0)
        view_t = bars_sorted.iloc[:stop]
        scores_row = scores.iloc[i].dropna()  # drop missing/NaN; policy also drops non-finite
        w = strategy.construct(scores_row, view_t)
        if len(w) == 0:
            continue
        try:
            validate_decision_weights(
                w, strategy.execution, strategy.name, allowed_symbols=static_universe
            )
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
        weights.loc[t, row.index] = row.to_numpy()
    return weights


def _decision_weights_fast(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast path used by ordinary backtests: `_fast_weights` followed by the fail-closed
    WEIGHT-level parity guard on a bounded deterministic sample (`_assert_parity`). The fast path is
    never trusted without that guard and never silently falls back. The promotion gate uses
    `verify_signal_panel_parity` instead, which checks EVERY bar."""
    weights = _fast_weights(strategy, bars, adj)
    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")
    _assert_parity(strategy, bars_sorted, end_pos, weights, strategy.execution.warmup_bars)
    return weights


def _parity_sample_positions(warmup: int, n: int) -> list[int]:
    """Deterministic, bounded sample of evaluated-bar positions (>= warmup, < n) spread across the
    period — first, last, and an evenly-spaced interior. Empty when there are no evaluated bars."""
    if n <= warmup:
        return []
    lo, hi = warmup, n - 1
    if hi == lo:
        return [lo]
    k = min(_PARITY_SAMPLE, hi - lo + 1)
    if k == 1:
        return [lo]
    step = (hi - lo) / (k - 1)
    return sorted({lo + round(j * step) for j in range(k)})


def _assert_parity(
    strategy: LoadedStrategy,
    bars_sorted: pd.DataFrame,
    end_pos: np.ndarray,
    weights: pd.DataFrame,
    warmup: int,
) -> None:
    """Fail-closed parity guard. Compares the fast-path weights row against the canonical per-bar
    construct(signal(view), view) on a bounded deterministic sample of evaluated bars, with
    tolerance `WEIGHT_TOL` (rtol=0). A discontinuous policy near-tie that a signal-level check could
    miss is caught here because we compare final WEIGHTS. Any mismatch RAISES `BacktestError` naming
    the disagreement — the fast path is never trusted without this check, and never silently falls
    back."""
    columns = weights.columns
    n = len(weights.index)
    for i in _parity_sample_positions(warmup, n):
        t = weights.index[i]
        stop = int(end_pos[i])
        # A rail breach raised inside the proxy (e.g. an out-of-universe per-bar weight) carries no
        # bar context; append ` at {t}` here so its message matches the loop / non-guard fast path.
        # Chain from the underlying RiskBreach (not the proxy's BacktestError) so `__cause__`
        # stays a direct RiskBreach, matching the loop / fast-path convention.
        try:
            canonical = _canonical_row(strategy, bars_sorted, stop, columns)
        except BacktestError as exc:
            cause = exc.__cause__ if isinstance(exc.__cause__, RiskBreach) else exc
            raise BacktestError(f"{exc} at {t}") from cause
        fast = pd.Series(weights.iloc[i].to_numpy(), index=columns)
        diff = (canonical - fast).abs()
        if bool((diff > WEIGHT_TOL).any()):
            offenders = sorted(diff[diff > WEIGHT_TOL].index)
            raise BacktestError(
                f"fast-path parity check FAILED for strategy {strategy.name!r} at {t}: "
                f"signal_panel disagrees with the per-bar construct(signal(view), view) on "
                f"symbol(s) {offenders} "
                f"(per-bar={canonical[offenders].to_dict()}, "
                f"panel={fast[offenders].to_dict()}, tol={WEIGHT_TOL})"
            )


def _decision_weights_fast_or_loop(
    strategy: LoadedStrategy,
    bars: pd.DataFrame,
    adj: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Selector for the decision step. Returns the same pre-lag weights matrix the loop returns.

    - No `signal_panel_fn` -> the canonical per-bar loop (`_decision_weights`), unchanged.
    - PIT mode (`universe_by_date is not None`) -> FORCE the loop. The per-bar as-of masking cannot
      be reproduced by a generic whole-period panel fn (membership at t depends on t), and we do not
      attempt a vectorized PIT mask here (deferred). The fast path is for the static-universe case.
    - Fundamentals (`fundamentals is not None`) -> FORCE the loop. No vectorized fundamentals fast
      path yet (issue #132).
    - News (`news is not None`) -> FORCE the loop. The per-bar as-of news mask (latest revision
      <= t, tombstones dropped) is path-dependent on t; no vectorized news fast path (issue #132).
    - Otherwise -> the vectorized fast path, gated by the fail-closed parity guard.
    """
    if (
        strategy.signal_panel_fn is None
        or universe_by_date is not None
        or fundamentals is not None
        or news is not None
    ):
        return _decision_weights(
            strategy, bars, adj,
            universe_by_date=universe_by_date, fundamentals=fundamentals, news=news,
        )
    return _decision_weights_fast(strategy, bars, adj)


def verify_signal_panel_parity(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> None:
    """EXHAUSTIVE fail-closed parity gate for promotion: assert `signal_panel` agrees with its
    per-bar `signal` twin on EVERY evaluated bar (not the bounded runtime sample).

    Verifies a CODE property — that the vectorized fast path equals the canonical per-bar
    `construct(signal(view), view)` — in STATIC mode over the strategy's declared universe. The
    agent promote backtest runs under PIT (which forces the loop and never exercises the panel) or,
    if `--universe` is omitted, may run the fast path; either way the panel must be checked here
    directly, where the fast path is the thing under test. No-op when `signal_panel_fn is None`.
    Raises `BacktestError` naming the first divergent bar + offending symbol(s)."""
    if strategy.signal_panel_fn is None:
        return  # nothing to verify

    try:
        bars = provider.get_bars(strategy.universe, start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error during parity check: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the signal_panel parity check")

    # A panel/construct that throws must fail the gate CLOSED rather than crash the JSON CLI
    # with an arbitrary exception type (e.g. AssertionError/KeyError/TypeError from user code).
    # Re-raise an existing BacktestError unchanged (divergence message preserved verbatim);
    # convert anything else to BacktestError so @json_errors always sees a known type.
    try:
        adj = _adj_grid(bars)
        # Same fail-closed guard as simulate(): an empty declared∩available universe would otherwise
        # surface as a confusing out-of-universe `(allowed: [])` breach instead of this clear cause.
        if strategy.universe and not (set(strategy.universe) & set(adj.columns)):
            raise BacktestError(
                f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
                f"universe {sorted(strategy.universe)} (fetched columns: "
                f"{sorted(map(str, adj.columns))})"
            )

        fast = _fast_weights(strategy, bars, adj)
        # static: universe_by_date=None, fundamentals=None
        loop = _decision_weights(strategy, bars, adj)

        # Identical grid by construction (both built on adj.index/columns); assert before comparing.
        if not (fast.index.equals(loop.index) and fast.columns.equals(loop.columns)):
            raise BacktestError(
                f"signal_panel parity check for {strategy.name!r}: fast/loop weight grids differ "
                f"(fast {fast.shape} vs loop {loop.shape})"
            )

        # NaN-safe, every-bar comparison. Both paths zero-fill final weights so a NaN cannot
        # survive; the isna() guard is defensive belt-and-suspenders against a future path.
        nan_mismatch = fast.isna() != loop.isna()
        diff = (loop - fast).abs()
        bad = nan_mismatch | (diff > WEIGHT_TOL)
        if bool(bad.to_numpy().any()):
            first = next(t for t in fast.index if bool(bad.loc[t].any()))
            offenders = sorted(bad.columns[bad.loc[first].to_numpy()])
            raise BacktestError(
                f"signal_panel exhaustive parity check FAILED for strategy {strategy.name!r} at "
                f"{first}: signal_panel disagrees with the per-bar construct(signal(view), view) on "  # noqa: E501
                f"symbol(s) {offenders} "
                f"(per-bar={loop.loc[first, offenders].to_dict()}, "
                f"panel={fast.loc[first, offenders].to_dict()}, tol={WEIGHT_TOL})"
            )
    except BacktestError:
        raise
    except Exception as exc:
        raise BacktestError(
            f"signal_panel parity check for {strategy.name!r} failed to run: {exc}"
        ) from exc


def _fetch_symbols(
    strategy: LoadedStrategy, universe_by_date: Mapping[date, Collection[str]] | None
) -> list[str]:
    """Symbols to fetch bars for.

    Static mode: the strategy's declared universe. PIT mode: the UNION of every symbol ever
    effective across the membership timeline — so price data exists for any ever-member, including
    membership active at `start` that derives from a snapshot dated before it. (The wiring layer
    already restricts the timeline to snapshots effective <= end_date.)
    """
    if universe_by_date is None:
        return strategy.universe
    union: set[str] = set()
    for members in universe_by_date.values():
        union.update(members)
    return sorted(union)


def _adj_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """The simulation grid: adj_close pivoted to (timestamp index x symbol columns), sorted by
    time. This index IS the bar date-index `vectorbt` simulates on and `pf.returns()` carries, so
    it is the single source of truth for both `build_portfolio` and `holdout_window`."""
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return adj.sort_index()


def holdout_window(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    holdout_frac: float,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
) -> tuple[str, str]:
    """The exact OOS holdout interval [start, end] (ISO dates) `walk_forward` would carve as the
    last `holdout_frac` of the simulation grid — computed from the bar date-index WITHOUT running
    the strategy. Reproduces `build_portfolio`'s grid (identical `n`), so the boundary is identical
    to `walk_forward`'s `holdout_metrics`. Computed at reserve time so the single-use guard can
    match on the bars that will actually be burned (issue #192).

    Degenerate inputs (no bars, or holdout rounds to <1 bar) return the conservative full
    grid/period: the subsequent `walk_forward` raises and the reservation is released, so the value
    is immaterial but stays fail-closed (a superset of any real tail)."""
    if not 0.0 < holdout_frac < 1.0:
        raise BacktestError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    try:
        bars = provider.get_bars(_fetch_symbols(strategy, universe_by_date), start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        return start.date().isoformat(), end.date().isoformat()
    idx = _adj_grid(bars).index
    n = len(idx)
    holdout_n = int(n * holdout_frac)
    if holdout_n < 1:
        return idx[0].date().isoformat(), idx[-1].date().isoformat()
    train_n = n - holdout_n
    return idx[train_n].date().isoformat(), idx[-1].date().isoformat()


def simulate(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Fetch bars, compute pre-lag decision weights (per-bar loop, or the vectorized fast path when
    the strategy exposes a parity-guarded `signal_panel_fn` — see `_decision_weights_fast_or_loop`),
    enforcing the shared long-only + gross-exposure risk checks; then apply the t->t+1 shift and
    simulate. Returns (portfolio, effective-weights). The shift lives ONLY here — the panel fn (like
    the loop) returns DECISION-time weights, never executable ones.

    This is the public simulation step: bars -> (portfolio, effective weights). Metrics are
    computed separately (see algua.backtest.metrics). Shared by run() and walk_forward().

    Point-in-time universe (`universe_by_date`): when provided, bars are fetched for the UNION of
    all ever-effective members and the per-bar decision is masked to as-of-t membership (see
    `_decision_weights`). `None` is the original static behavior — fetch the declared universe."""
    cadence = strategy.execution.rebalance_frequency.lower()
    if cadence not in _SUPPORTED_CADENCES:
        raise BacktestError(
            f"rebalance_frequency {strategy.execution.rebalance_frequency!r} not supported; "
            f"this slice rebalances daily only ({sorted(_SUPPORTED_CADENCES)})"
        )
    try:
        bars = provider.get_bars(_fetch_symbols(strategy, universe_by_date), start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    adj = _adj_grid(bars)

    if universe_by_date is None:
        operating_universe = set(strategy.universe) & set(adj.columns)
        if strategy.universe and not operating_universe:
            raise BacktestError(
                f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
                f"universe {sorted(strategy.universe)} (fetched columns: "
                f"{sorted(map(str, adj.columns))})"
            )

    fundamentals: pd.DataFrame | None = None
    if strategy.config.needs_fundamentals:
        if fundamentals_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_fundamentals but no "
                f"fundamentals_provider was supplied (fail closed)"
            )
        fundamentals = fundamentals_provider.get_fundamentals(
            _fetch_symbols(strategy, universe_by_date), end
        )
        _assert_fundamentals_shape(fundamentals)

    news: pd.DataFrame | None = None
    if strategy.config.needs_news:
        if news_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_news but no news_provider was "
                f"supplied (fail closed)"
            )
        news = news_provider.get_news(_fetch_symbols(strategy, universe_by_date), end)
        _assert_news_shape(news)

    weights = _decision_weights_fast_or_loop(
        strategy, bars, adj,
        universe_by_date=universe_by_date, fundamentals=fundamentals, news=news,
    )

    lag = strategy.execution.decision_lag_bars
    weights_eff = weights.shift(lag).fillna(0.0)
    pf = vbt.Portfolio.from_orders(
        close=adj,
        size=weights_eff,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )
    return pf, weights_eff


# build_portfolio is the explicit public alias of the simulation step. walk_forward and
# sweep import this (not a private helper).
build_portfolio = simulate


def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
) -> BacktestResult:
    pf, weights_eff = simulate(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
    )
    metrics = portfolio_metrics(pf, weights_eff)
    stamps = runtime_stamps()
    prov = provenance(provider, seed)
    return BacktestResult(
        strategy=strategy.name,
        metrics=metrics,
        config_hash=config_hash(strategy),
        timeframe="1d",
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        universe_name=universe_name,
        universe_snapshots=universe_snapshots,
        fundamentals_snapshot=getattr(fundamentals_provider, "snapshot_id", None),
        news_snapshot=(
            getattr(news_provider, "snapshot_id", None)
            if strategy.config.needs_news else None
        ),
        **prov,
    )
