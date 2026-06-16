"""Standalone factor evaluation (issue #140 slice B): wrap a single catalogued, signal-shaped
factor as an ephemeral synthetic strategy, run it through the existing backtest engine, and
compute construction-free rank IC/IR. Factors are NEVER registered, gate-tokened, or live-pathed:
the synthetic name uses the reserved `__factor__:` prefix and nothing here touches the registry."""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.engine import run as run_backtest
from algua.contracts.types import DataProvider, ExecutionContract
from algua.features.catalogue import FactorSpec, load_factor_callable
from algua.portfolio.construction import get_construction_policy, validate_construction_params
from algua.strategies.base import LoadedStrategy, StrategyConfig

SYNTHETIC_PREFIX = "__factor__:"


def build_factor_strategy(
    spec: FactorSpec,
    *,
    symbols: list[str],
    params: dict[str, Any],
    construction: str,
    construction_params: dict[str, Any],
    execution: ExecutionContract | None = None,
) -> LoadedStrategy:
    """Wrap a standalone-evaluable factor as a synthetic LoadedStrategy.

    Construction is required (no default) so factor eval imposes no hidden weighting bias.
    Rejects a non-standalone factor.
    """
    if not spec.standalone:
        raise ValueError(
            f"factor {spec.name!r} is not standalone-evaluable (not signal-shaped); "
            f"only standalone factors can be evaluated on their own"
        )
    validate_construction_params(construction, construction_params)
    fn = load_factor_callable(spec)
    config = StrategyConfig(
        name=f"{SYNTHETIC_PREFIX}{spec.name}",
        universe=symbols,
        execution=execution or ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params=params,
        construction=construction,
        construction_params=construction_params,
    )
    return LoadedStrategy(
        config=config,
        construct_fn=get_construction_policy(construction),
        signal_fn=fn,
    )


def factor_ic(
    score_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_cross_section: int = 3,
) -> dict[str, Any]:
    """Cross-sectional rank (Spearman) Information Coefficient summary.

    Per timestamp: Spearman correlation between the factor scores and the forward returns over the
    symbols finite in both. Bars with a cross-section narrower than `min_cross_section`, or a
    degenerate (zero-variance -> NaN) correlation, are skipped. Aggregates: mean IC, sample IC std
    (ddof=1), IR = mean/std, t-stat = IR*sqrt(n), hit rate (share of IC>0), n_obs. A run with
    < 2 usable bars (or zero IC variance) returns explicit None rather than a misleading number.

    NOT multiple-testing corrected — the t-stat is raw (FDR accounting is #140 slice E)."""
    ics: list[float] = []
    common = score_panel.index.intersection(forward_returns.index)
    for t in common:
        pair = pd.DataFrame(
            {"s": score_panel.loc[t], "r": forward_returns.loc[t]}
        )
        pair = pair[np.isfinite(pair["s"]) & np.isfinite(pair["r"])]
        if len(pair) < min_cross_section:
            continue
        ic = pair["s"].corr(pair["r"], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    n = len(ics)
    base: dict[str, Any] = {
        "method": "spearman",
        "n_obs": n,
        "min_cross_section": min_cross_section,
        "fdr_corrected": False,
    }
    if n < 2:
        return {**base, "mean_ic": None, "ic_std": None, "ir": None,
                "t_stat": None, "hit_rate": None}
    arr = np.array(ics, dtype=float)
    mean_ic = float(arr.mean())
    ic_std = float(arr.std(ddof=1))
    ir = mean_ic / ic_std if ic_std > 0 else None
    t_stat = ir * math.sqrt(n) if ir is not None else None
    return {
        **base,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ir": ir,
        "t_stat": t_stat,
        "hit_rate": float((arr > 0).mean()),
    }


def _adj_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """adj_close pivoted to (sorted unique timestamp index x symbol columns) — the decision grid."""
    return (
        bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )


def score_panel(strategy: LoadedStrategy, bars: pd.DataFrame) -> pd.DataFrame:
    """The factor's cross-sectional scores at every decision bar, PIT (data <= t only).

    For each timestamp t on the grid, calls the factor over the expanding window ending at t (the
    same expanding `view` the engine's per-bar loop uses), so a score at t can never see a bar
    after t. Returns a (timestamp x symbol) frame; bars before the factor has enough history
    contribute an all-NaN row."""
    bars_sorted = bars.sort_index()
    grid = _adj_grid(bars_sorted).index
    end_pos = bars_sorted.index.searchsorted(grid, side="right")
    rows: dict[pd.Timestamp, pd.Series] = {}
    for t, stop in zip(grid, end_pos, strict=True):
        rows[t] = strategy.signal(bars_sorted.iloc[:stop])
    panel = pd.DataFrame.from_dict(rows, orient="index")
    return panel.reindex(columns=_adj_grid(bars_sorted).columns)


def forward_returns(adj: pd.DataFrame, *, lag: int, horizon: int) -> pd.DataFrame:
    """Per-symbol forward return realized AFTER the decision lag: a score known at t is tradable at
    t+lag, so the label is adj_{t+lag+horizon} / adj_{t+lag} - 1. The trailing (lag+horizon) rows
    have no future bar and are NaN (skipped by the IC cross-section filter)."""
    entry = adj.shift(-lag)
    exit_ = adj.shift(-(lag + horizon))
    return exit_ / entry - 1.0


@dataclass
class FactorEvalResult:
    factor: str
    standalone: bool
    params: dict[str, Any]
    construction: str
    construction_params: dict[str, Any]
    horizon: int
    backtest: dict[str, Any]
    ic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "standalone": self.standalone,
            "params": self.params,
            "construction": self.construction,
            "construction_params": self.construction_params,
            "horizon": self.horizon,
            "backtest": self.backtest,
            "ic": self.ic,
        }


def evaluate_factor(
    spec: FactorSpec,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    symbols: list[str],
    params: dict[str, Any],
    construction: str,
    construction_params: dict[str, Any],
    horizon: int = 1,
    execution: ExecutionContract | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
) -> FactorEvalResult:
    """Evaluate one standalone factor: a real PIT backtest (existing engine) + rank IC/IR over the
    same fetched bars. IC is computed over the declared `symbols` (static); the backtest block
    honors `--universe` PIT membership via the engine. Touches no registry/holdout/gate."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    strategy = build_factor_strategy(
        spec, symbols=symbols, params=params, construction=construction,
        construction_params=construction_params, execution=execution,
    )
    bt = run_backtest(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, universe_name=universe_name,
        universe_snapshots=universe_snapshots,
    )
    bars = provider.get_bars(sorted(set(symbols)), start, end, "1d")
    panel = score_panel(strategy, bars)
    fwd = forward_returns(
        _adj_grid(bars), lag=strategy.execution.decision_lag_bars, horizon=horizon
    )
    ic = factor_ic(panel, fwd)
    return FactorEvalResult(
        factor=spec.name,
        standalone=spec.standalone,
        params=params,
        construction=construction,
        construction_params=construction_params,
        horizon=horizon,
        backtest=bt.to_dict(),
        ic=ic,
    )
