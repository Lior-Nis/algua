from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.live.live_loop import run_tick
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4)]


def _bars(symbol_prices):
    rows = []
    for sym, prices in symbol_prices.items():
        for ts, px in zip(DATES, prices, strict=True):
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1000})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class _FakeProvider:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, symbols, start, end, timeframe):
        return self._bars


class _FakeBroker:
    def __init__(self, positions=None):
        self._positions = pd.Series(positions or {}, dtype="float64")
        self.submitted = []
        self.cancels = 0

    def get_positions(self):
        return self._positions

    def cancel_open_orders(self):
        self.cancels += 1

    def submit(self, intent):
        self.submitted.append(intent)
        return f"order-{len(self.submitted)}"


def _strategy(weights, warmup_bars=0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series(weights))


def test_run_tick_submits_target_and_cancels_first():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert broker.cancels == 1
    assert len(result.submitted) == 1 and result.submitted[0]["symbol"] == "AAA"
    assert result.submitted[0]["order_id"] == "order-1"
    assert result.decision_ts == DATES[-1]


def test_run_tick_exits_dropped_symbol():
    broker = _FakeBroker(positions={"BBB": 10.0})  # held but not in target
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    syms = {o["symbol"]: o["target_weight"] for o in result.submitted}
    assert syms["BBB"] == 0.0  # exit order for the dropped name


def test_run_tick_warmup_not_met_submits_nothing():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})  # 3 sessions
    result = run_tick(_strategy({"AAA": 1.0}, warmup_bars=5), broker, _FakeProvider(bars),
                      DATES[0], DATES[-1])
    assert result.submitted == [] and broker.submitted == []


def test_run_tick_gross_breach_raises():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0, "BBB": 1.0}), broker, _FakeProvider(bars),
                 DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure"
