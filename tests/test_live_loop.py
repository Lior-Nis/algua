from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.alpaca_broker import BrokerError, TickSnapshot
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
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
    def __init__(self, positions=None, equity=100_000.0):
        self._positions = pd.Series(positions or {}, dtype="float64")
        self._equity = equity
        self.submitted = []
        self.client_order_ids = []
        self.cancels = 0
        self.snapshots = 0

    def get_positions(self):
        return self._positions

    def cancel_open_orders(self):
        self.cancels += 1

    def snapshot(self, universe):
        self.snapshots += 1
        syms = set(universe) | set(self._positions.index)
        qtys = {s: float(self._positions.get(s, 0.0)) for s in syms}
        # market value priced at $1/share for simplicity (qty == market value)
        return TickSnapshot(equity=self._equity, market_values=dict(qtys), qtys=qtys)

    def submit_sized(self, intent, snap, client_order_id=None):
        if intent.symbol not in snap.qtys:
            raise BrokerError(f"{intent.symbol} not in universe")
        self.submitted.append(intent)
        self.client_order_ids.append(client_order_id)
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
    assert broker.snapshots == 1  # #20: ONE snapshot per tick, not per symbol
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


def test_run_tick_drops_partial_current_session_bar():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})  # sessions Jan 2,3,4 2023
    # now = Jan 4 -> that session is "today" (possibly partial); decide on Jan 3 instead.
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      now=datetime(2023, 1, 4, 12, 0, tzinfo=UTC))
    assert result.decision_ts == DATES[1]


def test_run_tick_persists_each_order_immediately_with_client_order_id():
    # #18: on_submitted fires per accepted order (before the next submit), carrying a deterministic
    # client_order_id passed through to the broker.
    broker = _FakeBroker(positions={"BBB": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]})
    persisted = []
    hooks = TickHooks(
        client_order_id_for=lambda strat, ts, sym: f"{strat}-{sym}",
        on_submitted=persisted.append,
    )
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      hooks=hooks)
    assert {r.symbol for r in persisted} == {o["symbol"] for o in result.submitted}
    assert all(isinstance(r, SubmittedOrder) for r in persisted)
    assert all(c is not None for c in broker.client_order_ids)  # coid threaded to the broker


def test_run_tick_halts_before_submit_when_switch_trips():
    # #21: should_halt() true => abort before any cancel/submit.
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(should_halt=lambda: True)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert broker.cancels == 0 and broker.submitted == []  # nothing sent


def test_run_tick_drawdown_breach_halts_before_trading():
    # #27: equity below the persisted peak by more than max_drawdown trips before any order.
    broker = _FakeBroker(equity=80_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(peak_equity=100_000.0)
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks, max_drawdown=0.1)  # 20% drop > 10%
    assert ei.value.kind == "drawdown"
    assert broker.cancels == 0 and broker.submitted == []


def test_run_tick_reconcile_mismatch_raises():
    # #18: DB-derived positions disagreeing with the broker's pre-submit book halts the tick.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(derived_positions={"AAA": 999.0})  # DB thinks we hold far more
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert ei.value.kind == "reconcile"


def test_run_tick_reconcile_empty_db_vs_held_broker_raises():
    # #18 drift: the DB lost its record (empty derived) while the broker still holds positions.
    # Supplying the hook (even empty) must reconcile and halt, not skip on empty.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(derived_positions={})  # DB says flat, broker holds 10 -> drift
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert ei.value.kind == "reconcile"
    assert broker.cancels == 0 and broker.submitted == []  # halted before any order


def test_run_tick_no_reconcile_hook_does_not_compare():
    # No derived_positions hook supplied: the loop must not attempt reconcile (back-compat for the
    # pure decide+submit path) and trades normally.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert result.reconcile_ok is True
    assert broker.cancels == 1


def test_run_tick_halts_after_cancel_before_submit_when_switch_trips():
    # #21: the kill-switch trips after cancel is already in flight. The second should_halt() check
    # (after cancel, before the submit loop) must abort before any order is sent.
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    calls = {"n": 0}

    def _halt():
        calls["n"] += 1
        return calls["n"] >= 2  # first check (pre-cancel) passes; second (post-cancel) trips

    hooks = TickHooks(should_halt=_halt)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert broker.cancels == 1  # cancel ran
    assert broker.submitted == []  # but nothing was submitted


def test_run_tick_realized_gross_breach_trips_before_submit():
    # #27: the broker book is already over the realized gross limit BEFORE this tick. The realized
    # check must trip/flatten before any new order is sent, not after.
    broker = _FakeBroker(positions={"AAA": 100_000.0, "BBB": 100_000.0}, equity=100_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure_realized"
    assert broker.cancels == 0 and broker.submitted == []  # tripped before cancel + submit


def test_run_tick_ratchets_peak_equity():
    broker = _FakeBroker(equity=120_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      hooks=TickHooks(peak_equity=100_000.0))
    assert result.peak_equity == 120_000.0  # new high
