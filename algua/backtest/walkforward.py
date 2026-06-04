from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.engine import BacktestError, build_portfolio
from algua.backtest.metrics import metrics_from_returns
from algua.backtest.result import config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_MIN_WINDOW_BARS = 5


def _segment_bounds(
    n: int, windows: int, holdout_frac: float
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Partition n bars (by index) into K equal windows + a final holdout, as half-open ranges.

    Holdout = the last int(n*holdout_frac) bars. The remaining bars split into `windows` equal
    pieces; any integer-division remainder goes to the LAST window.
    """
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1)")
    holdout_n = int(n * holdout_frac)
    if holdout_n < 1:
        raise BacktestError(
            f"holdout_frac {holdout_frac} of {n} bars rounds to a 0-bar holdout; "
            f"increase --holdout-frac or widen the period so the holdout is non-empty"
        )
    train_n = n - holdout_n
    base = train_n // windows
    if base < _MIN_WINDOW_BARS:
        raise BacktestError(
            f"not enough bars: {train_n} train bars / {windows} windows is "
            f"< {_MIN_WINDOW_BARS} bars/window; widen the period or lower --windows"
        )
    bounds: list[tuple[int, int]] = []
    s = 0
    for i in range(windows):
        e = train_n if i == windows - 1 else s + base
        bounds.append((s, e))
        s = e
    return bounds, (train_n, n)


@dataclass
class WalkForwardResult:
    strategy: str
    config_hash: str
    data_source: str
    snapshot_id: str | None
    timeframe: str
    seed: int | None
    period: dict[str, str]
    windows: int
    holdout_frac: float
    window_metrics: list[dict[str, Any]]
    holdout_metrics: dict[str, Any]
    stability: dict[str, float]
    code_hash: str | None = None
    dependency_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _segment_record(returns: pd.Series, start_i: int, end_i: int) -> dict[str, Any]:
    seg = returns.iloc[start_i:end_i]
    rec: dict[str, Any] = {
        "start": str(seg.index[0].date()),
        "end": str(seg.index[-1].date()),
        "n_bars": int(len(seg)),
    }
    rec.update(metrics_from_returns(seg))
    return rec


def walk_forward(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    windows: int = 4,
    holdout_frac: float = 0.2,
    seed: int | None = None,
) -> WalkForwardResult:
    """Run the strategy once, then segment its return series into K windows + a final holdout."""
    pf, _weights = build_portfolio(strategy, provider, start, end)
    returns = pf.returns()
    bounds, holdout = _segment_bounds(len(returns), windows, holdout_frac)

    window_metrics = [
        {"index": i, **_segment_record(returns, s, e)} for i, (s, e) in enumerate(bounds)
    ]
    holdout_metrics = _segment_record(returns, holdout[0], holdout[1])

    sharpes = [w["sharpe"] for w in window_metrics]
    positive = sum(1 for w in window_metrics if w["total_return"] > 0)
    stability = {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "pct_positive_windows": float(positive / len(window_metrics)),
    }
    stamps = runtime_stamps()
    prov = provenance(provider, seed)

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=config_hash(strategy),
        timeframe="1d",
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
        **prov,
    )
