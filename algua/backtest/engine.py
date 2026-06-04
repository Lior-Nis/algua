from __future__ import annotations

from datetime import datetime

import pandas as pd
import vectorbt as vbt

from algua.backtest._constants import GROSS_EXPOSURE_TOL
from algua.backtest.metrics import portfolio_metrics
from algua.backtest.result import BacktestResult, config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_SUPPORTED_CADENCES = {"1d"}  # this slice rebalances on every daily bar only


class BacktestError(RuntimeError):
    pass


def _decision_weights(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Run the per-bar decision loop and return raw target weights (pre-lag).

    Path-dependent fallback. A strategy whose `target_weights` only depends on the latest
    fully-closed bar could be vectorized over the full panel; this loop is the general
    contract for path-dependent strategies. It walks the index once, slicing the history
    up to each bar `t` via positional `iloc` (a cheap expanding window) rather than
    re-copying a growing `.loc[:t]` prefix each step.

    The first `warmup_bars` bars are held flat: a strategy may need that much history
    before its signal is meaningful, so the first window excludes the warmup span.
    """
    columns = adj.columns
    max_gross = strategy.execution.max_gross_exposure
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
        w = strategy.target_weights(view)
        if len(w) == 0:
            continue
        row = w.reindex(columns).fillna(0.0)
        gross = float(row.abs().sum())
        if gross > max_gross + GROSS_EXPOSURE_TOL:
            raise BacktestError(
                f"strategy '{strategy.name}' targeted gross exposure {gross:.4f} at {t} "
                f"exceeding max_gross_exposure={max_gross}"
            )
        # Label-aligned assignment: place each symbol's weight in its own column rather than
        # trusting positional order (#42).
        weights.loc[t, row.index] = row.to_numpy()
    return weights


def simulate(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Fetch bars, run the per-bar decision loop (enforcing gross exposure), apply the
    t->t+1 shift, and simulate. Returns (portfolio, effective-weights).

    This is the public simulation step: bars -> (portfolio, effective weights). Metrics are
    computed separately (see algua.backtest.metrics). Shared by run() and walk_forward()."""
    cadence = strategy.execution.rebalance_frequency.lower()
    if cadence not in _SUPPORTED_CADENCES:
        raise BacktestError(
            f"rebalance_frequency {strategy.execution.rebalance_frequency!r} not supported; "
            f"this slice rebalances daily only ({sorted(_SUPPORTED_CADENCES)})"
        )
    try:
        bars = provider.get_bars(strategy.universe, start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    weights = _decision_weights(strategy, bars, adj)

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
) -> BacktestResult:
    pf, weights_eff = simulate(strategy, provider, start, end)
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
        **prov,
    )
