from datetime import UTC, datetime

import pandas as pd

from algua.contracts.types import OrderIntent, Side
from algua.execution.sim_broker import SimBroker

T0 = datetime(2023, 1, 2, tzinfo=UTC)
T1 = datetime(2023, 1, 3, tzinfo=UTC)


def _intent(symbol, weight, side):
    return OrderIntent(symbol=symbol, side=side, target_weight=weight, decision_ts=T0)


def test_fill_buys_at_next_open_and_spends_cash():
    b = SimBroker(cash=10_000.0)
    b.submit(_intent("AAA", 0.5, Side.BUY))
    fills = b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    assert len(fills) == 1 and fills[0].qty == 50.0 and fills[0].price == 100.0
    assert b.cash == 10_000.0 - 50 * 100.0
    assert b.get_positions()["AAA"] == 50.0


def test_sells_processed_before_buys_to_free_cash():
    b = SimBroker(cash=0.0)
    b.positions["AAA"] = 100.0
    b.submit(_intent("AAA", 0.0, Side.SELL))
    b.submit(_intent("BBB", 1.0, Side.BUY))
    opens = pd.Series({"AAA": 100.0, "BBB": 50.0})
    fills = b.fill_pending(opens, fill_ts=T1)
    syms = {f.symbol: f.qty for f in fills}
    assert syms["AAA"] == -100.0
    assert syms["BBB"] > 0
    assert b.cash >= 0.0


def test_buy_clamped_to_available_cash():
    b = SimBroker(cash=150.0)
    b.submit(_intent("AAA", 1.0, Side.BUY))
    b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    assert b.get_positions().get("AAA", 0.0) == 1.0
    assert b.cash >= 0.0


def test_pending_cleared_after_fill():
    b = SimBroker(cash=1000.0)
    b.submit(_intent("AAA", 0.5, Side.BUY))
    b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    assert b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1) == []


def test_sim_broker_conforms_to_broker_protocol():
    from algua.contracts.types import Broker
    assert isinstance(SimBroker(cash=1.0), Broker)


def test_every_pending_order_yields_a_fill_record():
    # #26: an unaffordable buy must NOT silently vanish — it returns a rejected zero-qty Fill.
    b = SimBroker(cash=50.0)  # can't afford even one share at 100
    oid = b.submit(_intent("AAA", 1.0, Side.BUY))
    fills = b.fill_pending(pd.Series({"AAA": 100.0}), fill_ts=T1)
    assert len(fills) == 1
    assert fills[0].broker_order_id == oid
    assert fills[0].qty == 0.0 and fills[0].status == "rejected"
    assert b.get_positions().get("AAA", 0.0) == 0.0


def test_untradeable_price_yields_rejected_fill():
    b = SimBroker(cash=10_000.0)
    oid = b.submit(_intent("AAA", 1.0, Side.BUY))
    fills = b.fill_pending(pd.Series({"AAA": float("nan")}), fill_ts=T1)  # no price
    assert len(fills) == 1 and fills[0].broker_order_id == oid
    assert fills[0].status == "rejected" and fills[0].qty == 0.0


def test_partial_fill_when_cash_short():
    # Hold BBB so equity (2000) exceeds cash (100): sizing wants 20 shares of AAA at 100, but only
    # $100 cash is on hand -> 1 share fills, status "partial".
    b = SimBroker(cash=100.0)
    b.positions["BBB"] = 19.0  # priced at 100 -> $1900 of equity
    b.submit(_intent("AAA", 1.0, Side.BUY))
    fills = b.fill_pending(pd.Series({"AAA": 100.0, "BBB": 100.0}), fill_ts=T1)
    aaa = next(f for f in fills if f.symbol == "AAA")
    assert aaa.qty == 1.0 and aaa.status == "partial"
