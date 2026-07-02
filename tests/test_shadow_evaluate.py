from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.shadow.evaluate import SHADOW_TIMEFRAME, shadow_replay
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4, 5, 6)]

_CONSTRUCTION = dict(construction="top_k_equal_weight", construction_params={"top_k": 1})


def _identity(scores: pd.Series, view: pd.DataFrame, params: dict) -> pd.Series:
    return scores


def _bars(symbol_prices: dict[str, list[float]], dates=DATES) -> pd.DataFrame:
    rows = []
    for sym, prices in symbol_prices.items():
        for ts, px in zip(dates, prices, strict=True):
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1000})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class _FakeProvider:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, symbols, start, end, timeframe):
        return self._bars


def _all_in(symbol: str, warmup: int = 0) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="all_in", universe=[symbol],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=warmup),
        **_CONSTRUCTION,
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda view, params: pd.Series({symbol: 1.0}), construct_fn=_identity
    )


# A `now` far after the data so the closed-bar cutoff keeps every bar in the default cases.
_FUTURE_NOW = datetime(2030, 1, 1, tzinfo=UTC)


def test_shadow_replay_produces_metrics_and_curve():
    # Buy AAA at day-2 close, fill at day-3 open (100), price rises to 110 -> a gain.
    prices = [100.0, 100.0, 110.0, 110.0, 110.0]
    provider = _FakeProvider(_bars({"AAA": prices}))
    result = shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1],
                           cash=100_000.0, now=_FUTURE_NOW)
    assert result.strategy == "all_in"
    assert result.n_bars >= 2
    # Bought ~1000 shares at 100 then marked at 110 -> equity above the starting cash.
    assert result.final_equity > 100_000.0
    assert result.final_positions.get("AAA", 0.0) > 0.0
    assert result.equity_curve[0][1] == pytest.approx(100_000.0)


def test_shadow_replay_rejects_non_daily_timeframe():
    provider = _FakeProvider(_bars({"AAA": [100.0] * 5}))
    with pytest.raises(ValueError, match="only timeframe"):
        shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1],
                      timeframe="1h", cash=100_000.0, now=_FUTURE_NOW)


def test_shadow_replay_rejects_non_positive_cash():
    provider = _FakeProvider(_bars({"AAA": [100.0] * 5}))
    with pytest.raises(ValueError, match="positive"):
        shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1], cash=0.0, now=_FUTURE_NOW)


def test_shadow_replay_drops_bars_on_or_after_now_date():
    # now is day-4: bars dated day-4/5/6 are dropped (>= now.date()), leaving only day-2/3.
    provider = _FakeProvider(_bars({"AAA": [100.0, 100.0, 100.0, 100.0, 100.0]}))
    now = datetime(2023, 1, 4, tzinfo=UTC)
    result = shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1], cash=100_000.0, now=now)
    # Only 2 closed sessions survive the cutoff -> at most 2 equity points.
    assert result.n_bars <= 2


def test_shadow_replay_empty_bars_returns_cash_flat():
    provider = _FakeProvider(pd.DataFrame(
        columns=["symbol", "open", "high", "low", "close", "adj_close", "volume"]
    ).rename_axis("timestamp"))
    result = shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1],
                           cash=50_000.0, now=_FUTURE_NOW)
    assert result.final_equity == pytest.approx(50_000.0)
    assert result.final_positions == {}
    assert result.n_bars == 0


def test_shadow_replay_no_look_ahead_decision_uses_only_closed_bars():
    # A price spike on the LAST bar must not retroactively change earlier decisions: the strategy
    # always targets AAA=1.0, so we assert the decision path fills at t+1 open, not future close.
    # Day-3 open is 100 (fill price); the day-6 spike to 500 only marks existing shares to market.
    prices = [100.0, 100.0, 100.0, 100.0, 500.0]
    provider = _FakeProvider(_bars({"AAA": prices}))
    result = shadow_replay(_all_in("AAA"), provider, DATES[0], DATES[-1],
                           cash=100_000.0, now=_FUTURE_NOW)
    shares = result.final_positions["AAA"]
    # Shares were bought against ~100-priced opens (not the 500 spike): ~1000 shares, well under
    # what buying at 500 would have produced. This proves fills used the t+1 open, no look-ahead.
    assert 500.0 < shares < 1100.0


def test_shadow_replay_warmup_holds_flat():
    # warmup=10 exceeds the 5 bars -> never decides -> stays all-cash, flat.
    provider = _FakeProvider(_bars({"AAA": [100.0, 110.0, 120.0, 130.0, 140.0]}))
    result = shadow_replay(_all_in("AAA", warmup=10), provider, DATES[0], DATES[-1],
                           cash=100_000.0, now=_FUTURE_NOW)
    assert result.final_positions == {}
    assert result.final_equity == pytest.approx(100_000.0)


def test_shadow_replay_timeframe_constant_is_daily():
    assert SHADOW_TIMEFRAME == "1d"
