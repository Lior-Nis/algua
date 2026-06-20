"""Tests for Finding 2: PIT membership masking in _market_return_series.

Symbol B joins only on a later date; the market return on early dates must equal symbol A's
return alone (B excluded), NOT the mean of A and B.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from algua.backtest.walkforward import _market_return_series
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

# ---------------------------------------------------------------------------
# Minimal deterministic provider: fixed price paths for A and B
# ---------------------------------------------------------------------------

_BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]

START = datetime(2022, 1, 3, tzinfo=UTC)  # a Monday to land on a market session
END = datetime(2022, 1, 14, tzinfo=UTC)   # ~10 trading days


def _make_bars_for_symbols(
    symbols: list[str],
    start: datetime,
    end: datetime,
    seed: int = 0,
) -> pd.DataFrame:
    """Build deterministic bar data using calendar sessions, one row per (symbol, date)."""
    from algua.calendar.market_calendar import MarketCalendar

    session_dates = MarketCalendar("XNYS").sessions_in_range(start.date(), end.date())
    sessions = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in session_dates], name="timestamp"
    )
    frames = []
    for i, sym in enumerate(sorted(symbols)):
        rng = np.random.default_rng(seed + i + 1)
        rets = rng.normal(loc=0.0005 * (i + 1), scale=0.005, size=len(sessions))
        close = 100.0 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "timestamp": sessions,
            "symbol": sym,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
        }))
    out = pd.concat(frames).set_index("timestamp").sort_values(["timestamp", "symbol"])
    return out[_BAR_COLUMNS]


class _FixedProvider:
    """DataProvider that returns pre-built bars (subset to the requested symbols)."""

    reproducible = True

    def __init__(self, bars: pd.DataFrame, seed: int = 0) -> None:
        self._bars = bars
        self.seed = seed

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        return self._bars[self._bars["symbol"].isin(symbols)]


def _strategy_with_universe(symbols: list[str]) -> LoadedStrategy:
    from algua.portfolio.construction import get_construction_policy

    cfg = StrategyConfig(
        name="pit_test",
        universe=symbols,
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


# ---------------------------------------------------------------------------
# Finding 2 test: PIT masking in _market_return_series
# ---------------------------------------------------------------------------

def test_pit_market_returns_excludes_nonmember_symbol_on_early_dates():
    """Symbol B joins only from session index 5 onward (universe_by_date).
    On EARLY dates (before B's join), the market return must equal A's return alone
    (B excluded via as-of masking), NOT the mean of A and B.

    This tests the FIX for Finding 2: _market_return_series must mask adj_close per
    date to as-of universe membership when universe_by_date is not None.
    """
    from algua.calendar.market_calendar import MarketCalendar

    all_bars = _make_bars_for_symbols(["AAA", "BBB"], START, END, seed=10)
    provider = _FixedProvider(all_bars, seed=10)
    strategy = _strategy_with_universe(["AAA", "BBB"])

    # Build session list to determine the cutover date
    session_dates = MarketCalendar("XNYS").sessions_in_range(START.date(), END.date())
    assert len(session_dates) >= 6, "need at least 6 sessions for this test"
    # B joins from session index 5 onward
    join_date = session_dates[5]

    # universe_by_date: before join_date only AAA; from join_date AAA+BBB
    universe_by_date: dict[date, set[str]] = {
        session_dates[0]: {"AAA"},
        join_date: {"AAA", "BBB"},
    }

    result = _market_return_series(strategy, provider, START, END, universe_by_date)
    assert result is not None, "_market_return_series must return a result"
    rets, dates = result

    # Build expected market returns: on early dates, only AAA contributes
    adj = (
        all_bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )
    daily = adj.pct_change()

    for ret, ds in zip(rets, dates, strict=False):
        d = date.fromisoformat(ds)
        ts = pd.Timestamp(d, tz="UTC")
        if ts not in daily.index:
            continue
        # Determine as-of membership on this date
        eligible_keys = [k for k in universe_by_date if k <= d]
        if not eligible_keys:
            continue
        members = universe_by_date[max(eligible_keys)]
        # Expected: mean of adj_close pct_change for MEMBERS ONLY on this date
        row = daily.loc[ts, list(members)]
        expected = float(row.mean())
        if not np.isfinite(expected):
            continue  # first bar has NaN pct_change; skip
        assert abs(ret - expected) < 1e-10, (
            f"On {ds}: expected {expected:.6f} (members={members}), got {ret:.6f}. "
            f"This means BBB contributed BEFORE its join date — PIT mask not applied."
        )

    # Also verify that on early dates (before join_date), BBB's return did NOT contribute:
    # A's return != mean(A, B) in general (different price paths) — so if masking is absent
    # the two differ. Find one early date where they differ.
    for ret, ds in zip(rets, dates, strict=False):
        d = date.fromisoformat(ds)
        ts = pd.Timestamp(d, tz="UTC")
        if d >= join_date:
            continue  # only check early dates
        if ts not in daily.index:
            continue
        a_ret = float(daily.loc[ts, "AAA"])
        b_ret = float(daily.loc[ts, "BBB"])
        if not (np.isfinite(a_ret) and np.isfinite(b_ret)):
            continue
        ab_mean = (a_ret + b_ret) / 2.0
        if abs(a_ret - ab_mean) > 1e-10:
            # On this date A's return != mean(A, B) — so if the market return equals A's
            # return, the masking is correct; if it equals the mean, it's wrong.
            assert abs(ret - a_ret) < 1e-10, (
                f"On early date {ds}: expected A-only return {a_ret:.6f}, "
                f"got {ret:.6f} (mean(A,B)={ab_mean:.6f}). "
                f"BBB is leaking into the benchmark before its join date."
            )
            break  # one assertion is enough


def test_static_universe_market_returns_unchanged():
    """When universe_by_date is None (static mode), _market_return_series behaves as before:
    all symbols contribute to every bar's return (no masking applied)."""
    all_bars = _make_bars_for_symbols(["AAA", "BBB"], START, END, seed=20)
    provider = _FixedProvider(all_bars, seed=20)
    strategy = _strategy_with_universe(["AAA", "BBB"])

    result = _market_return_series(strategy, provider, START, END, None)
    assert result is not None

    # Verify against expected equal-weighted mean of all symbols at each date
    adj = (
        all_bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )
    daily = adj.pct_change().mean(axis=1).dropna()
    rets, dates = result
    assert len(rets) == len(daily), "static mode: all dates should be present"
    for r, d, exp in zip(rets, dates, daily, strict=False):
        assert abs(r - exp) < 1e-10, f"Static mode mismatch on {d}: {r} vs {exp}"
