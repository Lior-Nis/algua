from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import build_intents, run_paper
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4, 5)]


def _bars(symbol_prices: dict[str, list[float]]) -> pd.DataFrame:
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


def _all_in(symbol: str) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="all_in", universe=[symbol],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series({symbol: 1.0}))


def test_build_intents_emits_on_weight_change():
    weights = pd.Series({"AAA": 1.0})
    positions = pd.Series(dtype="float64")
    closes = pd.Series({"AAA": 100.0})
    intents = build_intents(weights, positions, closes, equity=10_000.0,
                            decision_ts=DATES[0])
    assert len(intents) == 1 and intents[0].symbol == "AAA"


def test_build_intents_noop_when_already_at_target():
    intents = build_intents(pd.Series(dtype="float64"), pd.Series(dtype="float64"),
                            pd.Series(dtype="float64"), equity=10_000.0, decision_ts=DATES[0])
    assert intents == []


def test_build_intents_asserts_positive_equity():
    with pytest.raises(AssertionError, match="positive equity"):
        build_intents(pd.Series({"AAA": 1.0}), pd.Series(dtype="float64"),
                      pd.Series({"AAA": 100.0}), equity=0.0, decision_ts=DATES[0])


def test_run_paper_buys_and_reconciles():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    result = run_paper(_all_in("AAA"), SimBroker(cash=10_000.0), _FakeProvider(bars),
                       DATES[0], DATES[-1])
    assert result.reconcile_ok is True
    assert result.final_positions.get("AAA", 0.0) == 100.0
    assert len(result.fills) >= 1


def test_fills_never_share_timestamp_with_their_decision_bar():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    result = run_paper(_all_in("AAA"), SimBroker(cash=10_000.0), _FakeProvider(bars),
                       DATES[0], DATES[-1])
    for f in result.fills:
        assert f.fill_ts > f.decision_ts


def test_run_paper_rejects_negative_weights_long_only():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    cfg = StrategyConfig(name="shorty", universe=["AAA"],
                         execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1))
    short = LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series({"AAA": -1.0}))
    with pytest.raises(ValueError, match="long-only"):
        run_paper(short, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])


def _falling(symbol="AAA"):
    return _bars({symbol: [100.0, 90.0, 80.0, 70.0]})


def _strategy(weights: dict, warmup_bars: int = 0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
    )
    return LoadedStrategy(config=cfg, fn=lambda view, params: pd.Series(weights))


def test_gross_exposure_breach_raises():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0, 100.0]})
    strat = _strategy({"AAA": 1.0, "BBB": 1.0})  # gross 2.0 > 1.0
    with pytest.raises(RiskBreach) as ei:
        run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure"


def test_drawdown_breach_raises():
    strat = _strategy({"AAA": 1.0})  # all-in a falling stock
    with pytest.raises(RiskBreach) as ei:
        run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(_falling()),
                  DATES[0], DATES[-1], max_drawdown=0.05)
    assert ei.value.kind == "drawdown"


def test_warmup_gate_delays_first_order():
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    strat = _strategy({"AAA": 1.0}, warmup_bars=2)
    result = run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])
    assert all(o.intent.decision_ts >= DATES[1] for o in result.orders)
    assert len(result.orders) >= 1


def test_final_bar_drawdown_breach_raises():
    # Price craters only on the LAST bar; in-loop checks (at earlier closes) never see it,
    # so the post-loop final-equity drawdown check must catch it.
    strat = _strategy({"AAA": 1.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 50.0]})
    with pytest.raises(RiskBreach) as ei:
        run_paper(strat, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1],
                  max_drawdown=0.1)
    assert ei.value.kind == "drawdown"
