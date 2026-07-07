from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import (
    BacktestError,
    adj_grid,
    build_portfolio,
    fetch_symbols,
    members_as_of,
)
from algua.backtest.metrics import metrics_from_returns
from algua.backtest.result import config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.types import DataProvider, FundamentalsProvider, NewsProvider
from algua.strategies.base import LoadedStrategy

_MIN_WINDOW_BARS = 5


def _segment_bounds(
    n: int, windows: int, holdout_frac: float, embargo: int = 0
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Partition n bars (by index) into K equal windows + a final holdout, as half-open ranges,
    with an `embargo`-bar PURGE GAP between the last in-sample bar and the holdout (issue #345).

    Holdout = the last int(n*holdout_frac) bars `[train_n, n)` (UNCHANGED by the embargo, so the
    #192 single-use holdout-burn interval stays identical). The embargo carves the last `embargo`
    training bars `[train_n - embargo, train_n)` out of the in-sample region: the windows split
    `[0, train_n - embargo)` into `windows` equal pieces (remainder to the LAST window). The carved
    gap guarantees `max(train_idx) = train_n - embargo - 1 < train_n - embargo = holdout_start -
    embargo`, so no in-sample sample index lies within `embargo` bars of the holdout — the
    Lopez de Prado purge/embargo assertion. The holdout still reads earlier bars as feature
    HISTORY (a test set reading its own past inputs is not leakage); the gap only decorrelates the
    in-sample *selection* statistics from the holdout.
    """
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1)")
    if embargo < 0:
        # A negative embargo would set usable_train > train_n, overlapping the windows INTO the
        # holdout (re-opening the leak) while the carved-gap postcondition still held numerically.
        raise BacktestError(f"embargo must be >= 0, got {embargo}")
    holdout_n = int(n * holdout_frac)
    if holdout_n < 1:
        raise BacktestError(
            f"holdout_frac {holdout_frac} of {n} bars rounds to a 0-bar holdout; "
            f"increase --holdout-frac or widen the period so the holdout is non-empty"
        )
    train_n = n - holdout_n
    usable_train = train_n - embargo  # in-sample region after the embargo purge
    base = usable_train // windows if usable_train > 0 else 0
    if base < _MIN_WINDOW_BARS:
        raise BacktestError(
            f"not enough bars: {train_n} train bars minus a {embargo}-bar embargo leaves "
            f"{usable_train} usable / {windows} windows is < {_MIN_WINDOW_BARS} bars/window; "
            f"widen the period, lower --windows, or lower the embargo (feature_lookback)"
        )
    bounds: list[tuple[int, int]] = []
    s = 0
    for i in range(windows):
        e = usable_train if i == windows - 1 else s + base
        bounds.append((s, e))
        s = e
    # Postcondition: the carved gap is exactly `embargo` bars (and >= 0, since embargo >= 0).
    assert bounds[-1][1] == usable_train
    assert train_n - bounds[-1][1] == embargo
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
    # Purge/embargo gap (bars) carved between the in-sample windows and the holdout (#345).
    # = max(feature_lookback, decision_lag_bars) when the strategy declares a lookback, else 0.
    # Defaulted only so hand-built test fixtures need not set it; `walk_forward` always passes it.
    embargo: int = 0
    code_hash: str | None = None
    dependency_hash: str | None = None
    # Point-in-time universe provenance — separate from the bars `snapshot_id` (see BacktestResult).
    universe_name: str | None = None
    universe_snapshots: list[dict[str, str]] | None = None
    # PIT sidecar snapshot provenance (issue #132); None unless the strategy is needs_*.
    fundamentals_snapshot: str | None = None
    news_snapshot: str | None = None
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
        daily = adj.pct_change()
        if universe_by_date is not None:
            # PIT masking: apply the SAME as-of membership the engine uses in _decision_weights
            # so the benchmark never includes symbols before their effective-date join.
            # Non-members at date t are set to NaN so pandas .mean(axis=1) skips them,
            # correctly averaging only the as-of members at each bar. Reuses members_as_of
            # (the engine's public as-of helper) — no reinvention of the masking logic.
            masked = daily.copy()
            for ts in daily.index:
                members = members_as_of(universe_by_date, ts)
                non_members = [c for c in daily.columns if c not in members]
                if non_members:
                    masked.loc[ts, non_members] = float("nan")
            xs = masked.mean(axis=1).dropna()
        else:
            # Static universe: no masking needed — current behavior is correct.
            xs = daily.mean(axis=1).dropna()
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


def _resolve_embargo(strategy: LoadedStrategy, embargo: int | None) -> int:
    """The walk-forward purge gap (#345). An explicit ``embargo`` wins (the exploratory CLI
    override). Otherwise derive from the strategy: a DECLARED ``feature_lookback`` gives
    ``max(feature_lookback, decision_lag_bars)`` — the feature-window span OR the t->t+1 decision
    lag, whichever is larger; an UNDECLARED lookback (``None``) gives ``0`` (legacy zero-gap; the
    agent promote path refuses an undeclared lookback upstream)."""
    if embargo is not None:
        return embargo
    lookback = strategy.config.feature_lookback
    if lookback is None:
        return 0
    return max(lookback, strategy.execution.decision_lag_bars)


def walk_forward(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    windows: int = 4,
    holdout_frac: float = 0.2,
    embargo: int | None = None,
    seed: int | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
    on_peek: Callable[[str], None] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    assume_terminal_last_close: bool = False,
    compute_holdout: bool = True,
) -> WalkForwardResult:
    """Run the strategy once, then segment its return series into K windows + a final holdout,
    separated by an ``embargo``-bar purge gap (issue #345).

    ``embargo`` (bars between the last in-sample window and the holdout): when ``None`` it is
    derived from the strategy — ``max(feature_lookback, decision_lag_bars)`` if the strategy
    DECLARES ``feature_lookback`` (``config.feature_lookback is not None``), else ``0`` (an
    undeclared strategy keeps the legacy zero-gap behavior; the agent ``research promote`` path
    refuses an undeclared lookback upstream). An explicit value overrides the derivation.

    The returned ``holdout_metrics`` are SENSITIVE: callers that emit this result to operators
    (CLI output, MLflow artifacts, API responses, etc.) MUST withhold the ``holdout_metrics``
    field. Only ``research promote`` may reveal it — and doing so burns the holdout (single-use).

    ``on_peek`` (if given) is called exactly once, with the strategy ``config_hash``, immediately
    BEFORE the holdout window is evaluated. It is the burn point for a single-use holdout: a caller
    that commits a durable "burn" here can rely on nothing fallible-and-releasing running after it.

    ``compute_holdout`` (default ``True`` preserves today's behavior). When ``False`` — the PBO/CSCV
    window-only path (#467) — the holdout STATISTIC is never computed: the IDENTICAL
    ``_segment_bounds`` are carved (same K in-sample windows, same purge/embargo gap, same holdout
    interval reserved-but-unscored, so ``window_metrics`` is BIT-IDENTICAL to a normal run), but the
    holdout slice is never scored (``holdout_metrics = {}``, ``holdout_returns = None``),
    ``on_peek`` is NEVER called (no single-use burn), and ``_market_return_series`` is SKIPPED
    (``market_returns = None`` — the benchmark series is a regime-gate input the PBO path never
    consumes). ``build_portfolio(start, end)`` STILL runs over the full period, so the underlying
    holdout bar RANGE is read and strategy code executes over it (the accepted residual, #467 R2-1);
    only the holdout STATISTIC and burn are elided.
    """
    # Model lane (#376) is NOT supported in walk-forward/sweep this slice: a single fixed-per-run
    # model is not point-in-time safe across rolling OOS windows (each window would need the model
    # that predates ITS train end). Fail closed until per-window model binding is built (follow-up).
    if strategy.config.needs_model:
        raise BacktestError(
            f"strategy {strategy.name!r} declares needs_model; the model lane is only wired into "
            f"`backtest run` (single fixed period). Walk-forward needs per-window model binding — "
            f"not built yet (#376 follow-up); refusing to run (fail closed)"
        )
    embargo = _resolve_embargo(strategy, embargo)
    # PIT sidecar providers (#132) are threaded straight into build_portfolio (= simulate, which
    # consumes them); the `_reject_pit_sidecar` guard is removed here — unblocking needs_* in
    # walk-forward is the point of #132 slice 4 (the engine still fails closed if a needs_*
    # strategy is run without its provider).
    pf, _weights, _forced = build_portfolio(
        strategy, provider, start, end,
        universe_by_date=universe_by_date,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
    # Compute the full-period benchmark series immediately after build_portfolio (same provider +
    # universe_by_date — PIT-identical, a second read of the same immutable snapshot). Never raises.
    # SKIPPED on the window-only path (#467): the benchmark is a regime-gate input the PBO path
    # never consumes, so skipping it avoids a redundant holdout-spanning read.
    market_returns = (
        _market_return_series(strategy, provider, start, end, universe_by_date)
        if compute_holdout else None
    )
    returns = pf.returns()
    bounds, holdout = _segment_bounds(len(returns), windows, holdout_frac, embargo)

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

    # Holdout STATISTIC + single-use burn (#467 window-only path skips BOTH): on the
    # compute_holdout=False path the holdout is carved (bounds above) but never scored, on_peek is
    # never called, and no holdout return vector is materialized. window_metrics — carved from the
    # IDENTICAL _segment_bounds — is unaffected, so it stays BIT-IDENTICAL to a normal run.
    holdout_metrics: dict[str, Any]
    holdout_returns: tuple[list[float], list[str]] | None
    if compute_holdout:
        # Burn-on-peek boundary: the holdout metric is evaluated on the NEXT line, so any single-use
        # burn the caller commits in on_peek is durable before the peek. Nothing
        # fallible-and-releasing may be added between on_peek and the holdout evaluation.
        if on_peek is not None:
            on_peek(cfg_hash)
        holdout_metrics = _segment_record(returns, holdout[0], holdout[1])
        holdout_seg = returns.iloc[holdout[0]:holdout[1]]
        holdout_returns = (
            [float(x) for x in holdout_seg.to_numpy()],
            [str(idx.date()) for idx in holdout_seg.index],
        )
    else:
        holdout_metrics = {}
        holdout_returns = None

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=cfg_hash,
        timeframe="1d",
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        embargo=embargo,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
        universe_name=universe_name,
        universe_snapshots=universe_snapshots,
        fundamentals_snapshot=(
            getattr(fundamentals_provider, "snapshot_id", None)
            if strategy.config.needs_fundamentals else None),
        news_snapshot=(
            getattr(news_provider, "snapshot_id", None)
            if strategy.config.needs_news else None),
        holdout_returns=holdout_returns,
        market_returns=market_returns,
        **prov,
    )
