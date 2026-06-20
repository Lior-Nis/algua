from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import BacktestError, adj_grid, build_portfolio, fetch_symbols
from algua.backtest.metrics import metrics_from_returns
from algua.backtest.result import config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_MIN_WINDOW_BARS = 5


def _reject_pit_sidecar(strategy: LoadedStrategy, where: str) -> None:
    """Fail closed (clearly) when a PIT-sidecar strategy reaches walk_forward/sweep: their provider
    threading is deferred (#132), so without this they'd hit a confusing deep BacktestError. Covers
    BOTH lanes — fundamentals (pre-existing rough edge) and news."""
    if strategy.config.needs_fundamentals or strategy.config.needs_news:
        kind = "needs_fundamentals" if strategy.config.needs_fundamentals else "needs_news"
        raise BacktestError(
            f"{kind} strategies are not supported in {where} yet (#132 follow-up): "
            f"provider threading through {where} is deferred"
        )


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
    # SENSITIVE: any operator-facing emission of a WalkForwardResult MUST withhold this field
    # EXCEPT `research promote`, which is the sole command that reveals and burns the holdout.
    holdout_metrics: dict[str, Any]
    stability: dict[str, float]
    code_hash: str | None = None
    dependency_hash: str | None = None
    # Point-in-time universe provenance — separate from the bars `snapshot_id` (see BacktestResult).
    universe_name: str | None = None
    universe_snapshots: list[dict[str, str]] | None = None
    # SENSITIVE — stronger than holdout_metrics: the raw OOS return vector lets a researcher
    # identify which days their strategy failed and tune a later strategy to exploit the same
    # holdout window. NEVER serialized: to_dict() excludes it; only research promote persists it
    # (in promotion.run_gate), and only the sibling-only store read may surface it. (#221 Slice 1)
    holdout_returns: tuple[list[float], list[str]] | None = None
    # FULL-PERIOD equal-weighted cross-sectional daily return of the universe (#221 Slice 4) — the
    # market/benchmark series for vol-tertile regime labeling. NOT sensitive, but bulky → excluded
    # from to_dict (an internal gate input, like holdout_returns). None if the benchmark read fails.
    market_returns: tuple[list[float], list[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        # holdout_returns is SENSITIVE — never serialized (#221 Slice 1).
        # market_returns is bulky (full-period aggregate) — internal gate input (#221 Slice 4).
        # Both are excluded so the raw vectors never appear in operator-facing output.
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if f.name not in ("holdout_returns", "market_returns")}


def _market_return_series(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    universe_by_date: Mapping[date, Collection[str]] | None,
) -> tuple[list[float], list[str]] | None:
    """Equal-weighted cross-sectional daily return of the universe (PIT: same provider/snapshot as
    the backtest). Returns (returns, ISO-dates) over the FULL period, or None on any failure.

    Uses the SAME provider and universe_by_date the backtest used (PIT-identical — a second read
    of the same immutable snapshot). The series spans the FULL period (not just the holdout) so a
    later 21-bar trailing vol has lookback for every holdout date (#221 Slice 4).
    """
    try:
        symbols = fetch_symbols(strategy, universe_by_date)
        bars = provider.get_bars(symbols, start, end, "1d")
        adj = adj_grid(bars)                          # (dates x symbols) adj_close panel
        xs = adj.pct_change().mean(axis=1).dropna()   # equal-weighted cross-sectional return
        if xs.empty:
            return None
        return ([float(x) for x in xs.to_numpy()],
                [str(idx.date()) for idx in xs.index])
    except Exception:
        return None


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
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
    on_peek: Callable[[str], None] | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    assume_terminal_last_close: bool = False,
) -> WalkForwardResult:
    """Run the strategy once, then segment its return series into K windows + a final holdout.

    The returned ``holdout_metrics`` are SENSITIVE: callers that emit this result to operators
    (CLI output, MLflow artifacts, API responses, etc.) MUST withhold the ``holdout_metrics``
    field. Only ``research promote`` may reveal it — and doing so burns the holdout (single-use).

    ``on_peek`` (if given) is called exactly once, with the strategy ``config_hash``, immediately
    BEFORE the holdout window is evaluated. It is the burn point for a single-use holdout: a caller
    that commits a durable "burn" here can rely on nothing fallible-and-releasing running after it.
    """
    _reject_pit_sidecar(strategy, "walk-forward")
    pf, _weights, _forced = build_portfolio(
        strategy, provider, start, end,
        universe_by_date=universe_by_date,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
    # Compute the full-period benchmark series immediately after build_portfolio (same provider +
    # universe_by_date — PIT-identical, a second read of the same immutable snapshot). Never raises.
    market_returns = _market_return_series(strategy, provider, start, end, universe_by_date)
    returns = pf.returns()
    bounds, holdout = _segment_bounds(len(returns), windows, holdout_frac)

    window_metrics = [
        {"index": i, **_segment_record(returns, s, e)} for i, (s, e) in enumerate(bounds)
    ]

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
    cfg_hash = config_hash(strategy)

    # Burn-on-peek boundary: the holdout metric is evaluated on the NEXT line, so any single-use
    # burn the caller commits in on_peek is durable before the peek. Nothing fallible-and-releasing
    # may be added between on_peek and the holdout evaluation.
    if on_peek is not None:
        on_peek(cfg_hash)
    holdout_metrics = _segment_record(returns, holdout[0], holdout[1])
    holdout_seg = returns.iloc[holdout[0]:holdout[1]]
    holdout_returns = (
        [float(x) for x in holdout_seg.to_numpy()],
        [str(idx.date()) for idx in holdout_seg.index],
    )

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=cfg_hash,
        timeframe="1d",
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
        universe_name=universe_name,
        universe_snapshots=universe_snapshots,
        holdout_returns=holdout_returns,
        market_returns=market_returns,
        **prov,
    )
