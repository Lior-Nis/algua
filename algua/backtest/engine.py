from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

from algua.backtest.metrics import avg_gross_exposure, weights_turnover
from algua.backtest.result import BacktestResult
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_ANN = 252  # trading days/year, for annualization


class BacktestError(RuntimeError):
    pass


def _config_hash(strategy: LoadedStrategy) -> str:
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": {
                "rebalance_frequency": strategy.execution.rebalance_frequency,
                "decision_lag_bars": strategy.execution.decision_lag_bars,
                "max_gross_exposure": strategy.execution.max_gross_exposure,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
) -> BacktestResult:
    timeframe = "1d"
    try:
        bars = provider.get_bars(strategy.universe, start, end, timeframe)
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    # Per-bar decision loop: strategy only ever sees rows <= t (parity guarantee).
    weights = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
    for t in adj.index:
        view = bars.loc[:t]  # label slice on tz-aware DatetimeIndex -> rows with timestamp <= t
        w = strategy.target_weights(view)
        if len(w) > 0:
            row = w.reindex(weights.columns).fillna(0.0)
            weights.loc[t] = row.values

    # Enforce t -> t+1: decisions at t take effect no earlier than t + lag.
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

    metrics = _metrics(pf, weights_eff)
    return BacktestResult(
        strategy=strategy.name,
        metrics=metrics,
        config_hash=_config_hash(strategy),
        data_source=type(provider).__name__,
        timeframe=timeframe,
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        seed=getattr(provider, "seed", seed),
    )


def _metrics(pf: vbt.Portfolio, weights_eff: pd.DataFrame) -> dict[str, float]:
    total_return = float(pf.total_return())
    returns = pf.returns()
    ann_vol = float(returns.std() * np.sqrt(_ANN))
    mean_ann = float(returns.mean() * _ANN)
    sharpe = float(mean_ann / ann_vol) if ann_vol > 0 else 0.0
    n_periods = len(returns)
    cagr = float((1.0 + total_return) ** (_ANN / n_periods) - 1.0) if n_periods > 0 else 0.0
    max_dd = float(pf.max_drawdown())
    n_rebalances = int((weights_eff.diff().abs().sum(axis=1) > 1e-12).sum())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": weights_turnover(weights_eff),
        "avg_gross_exposure": avg_gross_exposure(weights_eff),
        "n_rebalances": n_rebalances,
    }
