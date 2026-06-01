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
