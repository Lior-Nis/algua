from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.engine import BacktestError, _build_portfolio, _config_hash
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_ANN = 252  # trading days/year
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


def metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    """Return-based metrics for one segment. Safe on empty / zero-vol input (-> zeros)."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "total_return": 0.0, "ann_return": 0.0, "ann_volatility": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
        }
    total_return = float((1.0 + r).prod() - 1.0)
    ann_return = float(r.mean() * _ANN)
    ann_vol = float(r.std() * np.sqrt(_ANN))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    equity = (1.0 + r).cumprod()
    # Floor the running peak at starting capital (1.0) so a loss on the first bar counts as
    # drawdown (otherwise the peak would start at the already-depressed first equity value).
    peak = equity.cummax().clip(lower=1.0)
    max_drawdown = float((equity / peak - 1.0).min())
    return {
        "total_return": total_return, "ann_return": ann_return, "ann_volatility": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_drawdown,
    }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "snapshot_id": self.snapshot_id,
            "timeframe": self.timeframe,
            "seed": self.seed,
            "period": self.period,
            "windows": self.windows,
            "holdout_frac": self.holdout_frac,
            "window_metrics": self.window_metrics,
            "holdout_metrics": self.holdout_metrics,
            "stability": self.stability,
        }


def _segment_record(returns: pd.Series, start_i: int, end_i: int) -> dict[str, Any]:
    seg = returns.iloc[start_i:end_i]
    rec: dict[str, Any] = {
        "start": str(seg.index[0].date()) if len(seg) else None,
        "end": str(seg.index[-1].date()) if len(seg) else None,
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
) -> WalkForwardResult:
    """Run the strategy once, then segment its return series into K windows + a final holdout."""
    pf, _weights = _build_portfolio(strategy, provider, start, end)
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

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=_config_hash(strategy),
        data_source=type(provider).__name__,
        snapshot_id=getattr(provider, "snapshot_id", None),
        timeframe="1d",
        seed=getattr(provider, "seed", None),
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
    )
