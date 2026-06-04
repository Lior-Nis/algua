"""Public backtest surface.

Stable names other lanes (walk-forward, sweep, and a stacked follow-up PR) import from the
package root instead of reaching into private engine internals (#38).
"""
from __future__ import annotations

from algua.backtest.engine import BacktestError, build_portfolio, run, simulate
from algua.backtest.metrics import (
    avg_gross_exposure,
    metrics_from_returns,
    portfolio_metrics,
    weights_turnover,
)
from algua.backtest.result import BacktestResult, config_hash, provenance

__all__ = [
    "BacktestError",
    "BacktestResult",
    "avg_gross_exposure",
    "build_portfolio",
    "config_hash",
    "metrics_from_returns",
    "portfolio_metrics",
    "provenance",
    "run",
    "simulate",
    "weights_turnover",
]
