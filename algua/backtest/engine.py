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
_SUPPORTED_CADENCES = {"1d"}  # this slice rebalances on every daily bar only


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


def _build_portfolio(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Fetch bars, run the per-bar decision loop (enforcing gross exposure), apply the
    t->t+1 shift, and simulate. Returns (portfolio, effective-weights). Shared by run()
    and walk_forward()."""
    timeframe = "1d"
    cadence = strategy.execution.rebalance_frequency.lower()
    if cadence not in _SUPPORTED_CADENCES:
        raise BacktestError(
            f"rebalance_frequency {strategy.execution.rebalance_frequency!r} not supported; "
            f"this slice rebalances daily only ({sorted(_SUPPORTED_CADENCES)})"
        )
    try:
        bars = provider.get_bars(strategy.universe, start, end, timeframe)
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    weights = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
    for t in adj.index:
        view = bars.loc[:t]
        w = strategy.target_weights(view)
        if len(w) > 0:
            row = w.reindex(weights.columns).fillna(0.0)
            gross = float(row.abs().sum())
            max_gross = strategy.execution.max_gross_exposure
            if gross > max_gross + 1e-9:
                raise BacktestError(
                    f"strategy '{strategy.name}' targeted gross exposure {gross:.4f} at {t} "
                    f"exceeding max_gross_exposure={max_gross}"
                )
            weights.loc[t] = row.values

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


def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
) -> BacktestResult:
    pf, weights_eff = _build_portfolio(strategy, provider, start, end)
    metrics = _metrics(pf, weights_eff)
    return BacktestResult(
        strategy=strategy.name,
        metrics=metrics,
        config_hash=_config_hash(strategy),
        data_source=type(provider).__name__,
        timeframe="1d",
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        seed=getattr(provider, "seed", seed),
        snapshot_id=getattr(provider, "snapshot_id", None),
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
