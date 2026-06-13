from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import build_intents, run_paper
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4, 5)]

_CONSTRUCTION = dict(construction="top_k_equal_weight", construction_params={"top_k": 1})


def _identity(scores: pd.Series, view: pd.DataFrame, params: dict) -> pd.Series:
    """Test-local construction: the injected scores ARE the target weights, so the precise vector
    under test reaches the risk rails unchanged."""
    return scores


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
        **_CONSTRUCTION,
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda view, params: pd.Series({symbol: 1.0}), construct_fn=_identity
    )


def test_build_intents_emits_on_weight_change():
    weights = pd.Series({"AAA": 1.0})
    intents = build_intents(weights, current_weights={}, decision_ts=DATES[0])
    assert len(intents) == 1 and intents[0].symbol == "AAA"


def test_build_intents_noop_when_already_at_target():
    intents = build_intents(pd.Series({"AAA": 1.0}), current_weights={"AAA": 1.0},
                            decision_ts=DATES[0])
    assert intents == []


class _NonPositiveEquityBroker:
    """SimBroker stand-in whose equity is non-positive — a held book that has gone past zero. Lets
    the run_paper sizing guard be exercised directly without having to drive a real SimBroker below
    zero (#162). Records submits so the test can assert nothing was sent."""

    def __init__(self, equity_value: float) -> None:
        self.cash = 0.0
        self._equity = equity_value
        self.submitted: list = []

    def equity(self, closes) -> float:
        return self._equity

    def get_positions(self):
        return pd.Series({"AAA": 10.0}, dtype="float64")

    def submit(self, intent):
        self.submitted.append(intent)
        return "should-not-be-submitted"

    def fill_pending(self, opens, fill_ts):
        return []


@pytest.mark.parametrize("equity_value", [0.0, -500.0, float("nan")])
def test_run_paper_non_positive_equity_breaches_before_orders(equity_value):
    # #162: replace the (python -O strippable) `assert equity > 0` with a real fail-closed breach,
    # so a non-usable sizing denominator (zero, negative, or NaN) never reaches the mv/equity
    # division and order phase. NaN slips a bare `<= 0` guard, so it's covered explicitly.
    broker = _NonPositiveEquityBroker(equity_value)
    bars = _bars({"AAA": [100.0, 100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_paper(_all_in("AAA"), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "non_positive_equity"
    assert broker.submitted == []  # nothing sent


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
                         execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
                         **_CONSTRUCTION)
    short = LoadedStrategy(
        config=cfg, signal_fn=lambda view, params: pd.Series({"AAA": -1.0}), construct_fn=_identity
    )
    with pytest.raises(ValueError, match="long-only"):
        run_paper(short, SimBroker(cash=10_000.0), _FakeProvider(bars), DATES[0], DATES[-1])


def _falling(symbol="AAA"):
    return _bars({symbol: [100.0, 90.0, 80.0, 70.0]})


def _strategy(weights: dict, warmup_bars: int = 0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
        **_CONSTRUCTION,
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda view, params: pd.Series(weights), construct_fn=_identity
    )


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
