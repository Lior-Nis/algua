from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date, datetime

import numpy as np
import pandas as pd
import vectorbt as vbt

from algua.backtest.decision_path import _decision_weights_fast_or_loop
from algua.backtest.delisting import DelistingExitError, DelistingRecord, apply_delisting_exits
from algua.backtest.errors import BacktestError
from algua.backtest.grid import adj_grid
from algua.backtest.metrics import portfolio_metrics
from algua.backtest.pit_view import (
    _assert_fundamentals_shape,
    _assert_news_shape,
    _static_operating_view,
)
from algua.backtest.result import BacktestResult, config_hash, provenance
from algua.backtest.stamps import runtime_stamps
from algua.contracts.model_types import normalize_as_of
from algua.contracts.types import (
    DataProvider,
    FundamentalsProvider,
    NewsProvider,
)
from algua.strategies.base import LoadedStrategy

_SUPPORTED_CADENCES = {"1d"}  # this slice rebalances on every daily bar only

# Residual-position tolerance for the post-delisting-exit guarantee.
POSITION_EPS = 1e-9


def fetch_symbols(
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


def adj_open_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """The ADJUSTED-open execution grid (issue #383): raw `open` scaled by the SAME per-bar
    adjustment ratio that maps raw close -> adj_close, so it is to adj_close exactly what raw open
    is to raw close.

        adj_open[t] = open[t] * (adj_close[t] / close[t])

    Splits and dividends scale a whole bar uniformly, so this preserves `adj_open/adj_close ==
    open/close` and is the correct next-bar-open reference in the backtest's adjusted frame. Shares
    `adj_grid`'s (timestamp x symbol) index/columns EXACTLY, so it is a drop-in fill-price grid that
    leaves holdout_window / the #192 single-use holdout identity / PIT masking / the returns index
    untouched — only the price VALUES change.

    Fail-safe cells: any bar where `close <= 0`, `open <= 0`, either is NaN, or the ratio is
    non-finite (inf/-inf/NaN) yields a NaN adj_open — an untradeable bar that `from_orders` treats
    as a no-fill, identical to a missing bar. So a corrupt OHLC row can never inject a bogus fill
    price (it can only decline to fill)."""
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="open").sort_index()
    close = bars.reset_index().pivot(
        index="timestamp", columns="symbol", values="close"
    ).sort_index()
    adj_close = adj_grid(bars)
    open_ = wide.to_numpy(dtype="float64")
    close_ = close.to_numpy(dtype="float64")
    adjc = adj_close.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((close_ > 0) & np.isfinite(close_), adjc / close_, np.nan)
        adj_open = np.where((open_ > 0) & np.isfinite(open_), open_ * ratio, np.nan)
    adj_open[~np.isfinite(adj_open)] = np.nan  # inf/-inf ratios collapse to a no-fill cell
    return pd.DataFrame(adj_open, index=adj_close.index, columns=adj_close.columns)


def simulate(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    assume_terminal_last_close: bool = False,
) -> tuple[vbt.Portfolio, pd.DataFrame, list[dict]]:
    """Fetch bars, compute pre-lag decision weights (per-bar loop, or the vectorized fast path when
    the strategy exposes a parity-guarded `signal_panel_fn` — see `_decision_weights_fast_or_loop`),
    enforcing the shared long-only + gross-exposure risk checks; then apply the t->t+1 shift, the
    delisting-aware exit overlay, and simulate. Returns (portfolio, executed-weights, forced-exits).
    The shift lives ONLY here — the panel fn (like the loop) returns DECISION-time weights, never
    executable ones.

    This is the public simulation step: bars -> (portfolio, executed weights, forced exits).
    Metrics are computed separately (see algua.backtest.metrics). Shared by run()/walk_forward().

    Delisting-aware exit (`delisting_records`): a position held past a symbol's last bar is realized
    at the record's terminal price and removed (see `apply_delisting_exits`); a held-into-gap symbol
    with no record fails closed unless `assume_terminal_last_close` (human-only).

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
        bars = provider.get_bars(fetch_symbols(strategy, universe_by_date), start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the universe/period")

    # Model lane PIT guard (#376): a fixed-per-run model is point-in-time safe ONLY if it predates
    # the whole evaluated period. Enforced HERE — the mandatory simulate() boundary shared by run()
    # and walk_forward() — so a direct programmatic call can never run a model that saw data after
    # the first decision bar. training_as_of and the first bar are normalized to comparable UTC
    # instants; an unparseable training_as_of fails closed.
    if strategy.config.needs_model:
        assert strategy.config.model_ref is not None
        first_bar = pd.Timestamp(bars.index.min())
        first_bar_utc = (
            first_bar.tz_localize("UTC") if first_bar.tzinfo is None
            else first_bar.tz_convert("UTC")
        )
        try:
            training_as_of = normalize_as_of(strategy.config.model_ref.training_as_of)
        except (ValueError, TypeError) as exc:
            raise BacktestError(
                f"strategy {strategy.name!r}: model "
                f"{strategy.config.model_ref.name!r} v{strategy.config.model_ref.version} has an "
                f"unparseable training_as_of {strategy.config.model_ref.training_as_of!r} "
                f"(fail closed): {exc}"
            ) from exc
        if training_as_of > first_bar_utc:
            raise BacktestError(
                f"strategy {strategy.name!r}: model {strategy.config.model_ref.name!r} "
                f"v{strategy.config.model_ref.version} has training_as_of "
                f"{strategy.config.model_ref.training_as_of} AFTER the first decision bar "
                f"{first_bar_utc.date().isoformat()} — that model saw future data (look-ahead); "
                f"refusing to backtest (fail closed)"
            )

    adj = adj_grid(bars)

    if universe_by_date is None:
        # Static mode: project the strategy-visible view + grid to the operating universe so an
        # undeclared symbol a misbehaving provider returned cannot influence in-universe decisions
        # (observation parity, #208). PIT keeps its per-bar as-of mask instead.
        bars, adj = _static_operating_view(strategy, bars, adj)

    fundamentals: pd.DataFrame | None = None
    if strategy.config.needs_fundamentals:
        if fundamentals_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_fundamentals but no "
                f"fundamentals_provider was supplied (fail closed)"
            )
        fundamentals = fundamentals_provider.get_fundamentals(
            fetch_symbols(strategy, universe_by_date), end
        )
        _assert_fundamentals_shape(fundamentals)

    news: pd.DataFrame | None = None
    if strategy.config.needs_news:
        if news_provider is None:
            raise BacktestError(
                f"strategy {strategy.name!r} declares needs_news but no news_provider was "
                f"supplied (fail closed)"
            )
        news = news_provider.get_news(fetch_symbols(strategy, universe_by_date), end)
        _assert_news_shape(news)

    weights = _decision_weights_fast_or_loop(
        strategy, bars, adj,
        universe_by_date=universe_by_date, fundamentals=fundamentals, news=news,
    )

    lag = strategy.execution.decision_lag_bars
    weights_eff = weights.shift(lag).fillna(0.0)

    # Fill-price basis (issue #383): pick the intra-bar execution-price grid the contract pins, so
    # the backtest fills on the SAME semantic reference the paper/live loop uses. Default "open"
    # fills at the adjusted next-bar open (adj_open_grid); "close" is the legacy adj_close basis.
    # `adj` (the adj_close grid) stays the sim DATE-INDEX / holdout identity source of truth — only
    # the fill PRICE grid handed to from_orders changes.
    exec_grid = adj_open_grid(bars) if strategy.execution.fill_price == "open" else adj

    try:
        # terminal_price_grid=adj pins the delisting `assume_terminal_last_close` fallback to the
        # last adj_CLOSE (a delisting is a close-of-book realization, not an open fill) regardless
        # of the ordinary fill basis; the ordinary (non-terminal) fill cells come from exec_grid.
        adj_exec, weights_exec, forced_exits = apply_delisting_exits(
            exec_grid, weights_eff, delisting_records,
            assume_terminal_last_close=assume_terminal_last_close,
            terminal_price_grid=adj,
        )
    except DelistingExitError as exc:
        raise BacktestError(str(exc)) from exc

    # call_seq="auto" (sells before buys under cash_sharing) only when a forced exit needs the
    # same-bar liquidation cash — keeps non-delisting backtests bit-identical to today.
    extra = {"call_seq": "auto"} if forced_exits else {}
    # Transaction costs (issue #325). Charged on `weights_exec` — the ALREADY t->t+1-shifted,
    # delisting-adjusted execution weights — so cost is applied at execution time on the bar the
    # trade fills, introducing NO look-ahead. vectorbt applies `fees` as a proportional charge on
    # |trade notional| and `slippage` as an adverse per-side fill-price move (buys/covers fill
    # higher, sells/shorts fill lower), so the model is symmetric across sides. The forced
    # delisting liquidation is a real (modeled) trade and is charged the same conservative cost;
    # over-charging a rare terminal exit only makes returns LOWER, never inflates edge — the safe
    # direction for a gate that must not overstate edge. DEFAULT-ON via the ExecutionContract.
    pf = vbt.Portfolio.from_orders(
        close=adj_exec,
        size=weights_exec,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        freq="1D",
        fees=strategy.execution.fees,
        slippage=strategy.execution.slippage,
        **extra,
    )

    if forced_exits:
        positions = pf.assets()
        for fe in forced_exits:
            sym = fe["symbol"]
            bar = pd.Timestamp(fe["bar"])
            after = positions[sym].loc[positions.index >= bar]
            if bool((after.abs() > POSITION_EPS).any()):
                raise BacktestError(
                    f"delisting exit for {sym} left a residual position after {fe['bar']}"
                )
        returns = pf.returns()
        if not bool(np.isfinite(returns.fillna(0.0)).all()):
            raise BacktestError("non-finite returns after delisting exits")

    return pf, weights_exec, forced_exits


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
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    delisting_snapshot: str | None = None,  # surfaced in BacktestResult provenance (#212)
    assume_terminal_last_close: bool = False,
) -> BacktestResult:
    pf, weights_eff, forced_exits = simulate(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
    metrics = portfolio_metrics(
        pf, weights_eff,
        fees=strategy.execution.fees, slippage=strategy.execution.slippage,
    )
    stamps = runtime_stamps()
    prov = provenance(provider, seed)
    # Surface daily returns for downstream correlation analysis (#222 Task 7).
    returns = pf.returns()
    if not bool(np.isfinite(returns.fillna(0.0)).all()):
        returns_series: pd.Series | None = None  # fail-closed: non-finite returns not surfaced
    else:
        returns_series = returns
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
        news_snapshot=(
            getattr(news_provider, "snapshot_id", None)
            if strategy.config.needs_news else None
        ),
        # Model provenance (#376): the pinned model_ref (name/version/digest/training_as_of/
        # provenance_digest) is stamped HERE, at the same mandatory boundary as the run, so a
        # programmatic caller cannot get an unstamped result. None for non-model strategies.
        model_ref=(
            strategy.config.model_ref.as_dict()
            if strategy.config.needs_model and strategy.config.model_ref is not None
            else None
        ),
        delisting_snapshot=delisting_snapshot,
        forced_exits=forced_exits,
        returns=returns_series,
        **prov,
    )
