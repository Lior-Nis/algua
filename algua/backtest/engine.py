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
    DataProvider,
    FundamentalsProvider,
)
from algua.risk.limits import WEIGHT_TOL, RiskBreach, validate_decision_weights
from algua.strategies.base import LoadedStrategy

_SUPPORTED_CADENCES = {"1d"}  # this slice rebalances on every daily bar only

# Fail-closed runtime parity guard: number of post-warmup bars at which the fast path is
# re-verified against the canonical per-bar definition on every run that uses a panel_fn.
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


def _decision_weights(
    strategy: LoadedStrategy,
    bars: pd.DataFrame,
    adj: pd.DataFrame,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals: pd.DataFrame | None = None,
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
            w = strategy.target_weights(view, f_asof)
        else:
            w = strategy.target_weights(view)
        if len(w) == 0:
            continue
        if universe_by_date is not None:
            non_members = [s for s in w.index[w != 0.0] if s not in members]
            if non_members:
                raise BacktestError(
                    f"strategy {strategy.name!r} returned weight for non-member symbol(s) "
                    f"{sorted(non_members)} at {t} (as-of members: {sorted(members)})"
                )
        # The shared checks raise RiskBreach; re-raise as BacktestError for the backtest CLI/error
        # contract while preserving the breach (and its `.kind`) as the cause.
        try:
            validate_decision_weights(w, strategy.execution, strategy.name)
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
    SAME computation the loop performs per bar — reused by the fast-path parity guard so the guard
    compares the fast path against the loop's own definition, not a re-derivation."""
    view = bars_sorted.iloc[:stop]
    w = strategy.target_weights(view)
    if len(w) == 0:
        return pd.Series(0.0, index=columns)
    return w.reindex(columns).fillna(0.0)


def _decision_weights_fast(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast path: call the strategy's `signal_panel` ONCE for the whole period to get the
    SCORES matrix, then apply the construction policy PER BAR (with the same expanding `view_t` the
    loop uses) + the shared risk walls. A fail-closed WEIGHT-level parity guard then confirms the
    result equals the canonical per-bar `construct(signal(view), view)` on a bounded sample. Static
    universe only; pre-lag, like the loop.

    The speedup is computing the signal once instead of recomputing it on the expanding view each
    bar; construction stays per-bar (cheap for the view-independent starter policies). The scores
    matrix is NOT NaN-filled before construction — a missing score means 'no opinion' and the policy
    drops it; only the FINAL weights are zero-filled to flat.
    """
    panel = strategy.signal_panel(bars)
    assert panel is not None  # caller guarantees signal_panel_fn is set
    if not isinstance(panel, pd.DataFrame):
        raise BacktestError(
            f"strategy {strategy.name!r} signal_panel returned "
            f"{type(panel).__name__}, expected a DataFrame"
        )
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
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
            validate_decision_weights(w, strategy.execution, strategy.name)
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
        weights.loc[t, row.index] = row.to_numpy()

    _assert_parity(strategy, bars_sorted, end_pos, weights, warmup)
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
        canonical = _canonical_row(strategy, bars_sorted, stop, columns)
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
) -> pd.DataFrame:
    """Selector for the decision step. Returns the same pre-lag weights matrix the loop returns.

    - No `signal_panel_fn` -> the canonical per-bar loop (`_decision_weights`), unchanged.
    - PIT mode (`universe_by_date is not None`) -> FORCE the loop. The per-bar as-of masking cannot
      be reproduced by a generic whole-period panel fn (membership at t depends on t), and we do not
      attempt a vectorized PIT mask here (deferred). The fast path is for the static-universe case.
    - Fundamentals (`fundamentals is not None`) -> FORCE the loop. No vectorized fundamentals fast
      path yet (issue #132).
    - Otherwise -> the vectorized fast path, gated by the fail-closed parity guard.
    """
    if (
        strategy.signal_panel_fn is None
        or universe_by_date is not None
        or fundamentals is not None
    ):
        return _decision_weights(
            strategy, bars, adj, universe_by_date=universe_by_date, fundamentals=fundamentals
        )
    return _decision_weights_fast(strategy, bars, adj)


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


def simulate(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Fetch bars, compute pre-lag decision weights (per-bar loop, or the vectorized fast path when
    the strategy exposes a parity-guarded `panel_fn` — see `_decision_weights_fast_or_loop`),
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

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

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

    weights = _decision_weights_fast_or_loop(
        strategy, bars, adj, universe_by_date=universe_by_date, fundamentals=fundamentals
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
) -> BacktestResult:
    pf, weights_eff = simulate(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, fundamentals_provider=fundamentals_provider,
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
        **prov,
    )
