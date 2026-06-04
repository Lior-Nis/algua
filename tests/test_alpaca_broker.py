from datetime import UTC, datetime

import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution import alpaca_broker as ab
from algua.execution.alpaca_broker import AlpacaPaperBroker, BrokerError

T0 = datetime(2023, 1, 2, tzinfo=UTC)


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests:
    """Routes GET by URL suffix; records POST bodies."""

    def __init__(self, routes, post_resp=None):
        self.routes = routes
        self.post_resp = post_resp
        self.posted = []

    def get(self, url, headers=None, timeout=None):
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, text="not found")

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted.append(json)
        return self.post_resp


def _broker():
    return AlpacaPaperBroker(api_key="k", api_secret="s")


def test_alpaca_broker_conforms_to_broker_protocol():
    from algua.contracts.types import Broker
    assert isinstance(_broker(), Broker)


def test_account_parses_floats(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "50000",
                                        "buying_power": "150000"})}))
    acct = _broker().account()
    assert (acct.equity, acct.cash, acct.buying_power) == (100000.0, 50000.0, 150000.0)


def test_get_positions_parses_and_empty(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "10"},
                                          {"symbol": "BBB", "qty": "5"}])}))
    pos = _broker().get_positions()
    assert pos["AAA"] == 10.0 and pos["BBB"] == 5.0

    monkeypatch.setattr(ab, "requests", _FakeRequests({"/v2/positions": _FakeResp(200, [])}))
    assert _broker().get_positions().empty


def test_submit_buy_delta_posts_notional(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},  # flat
        post_resp=_FakeResp(201, {"id": "order-1"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    oid = _broker().submit(OrderIntent("AAA", Side.BUY, 0.5, T0))
    assert oid == "order-1"
    assert fake.posted[0] == {"symbol": "AAA", "notional": "50000.00", "side": "buy",
                              "type": "market", "time_in_force": "day"}


def test_submit_sell_delta(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "600",
                                           "market_value": "60000"}])},
        post_resp=_FakeResp(201, {"id": "order-2"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    _broker().submit(OrderIntent("AAA", Side.SELL, 0.5, T0))
    assert fake.posted[0]["side"] == "sell" and fake.posted[0]["notional"] == "10000.00"


def test_submit_noop_below_threshold(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "500",
                                           "market_value": "50000"}])},
        post_resp=_FakeResp(201, {"id": "unused"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    assert _broker().submit(OrderIntent("AAA", Side.BUY, 0.5, T0)) == "noop"
    assert fake.posted == []


def test_snapshot_is_one_account_and_one_positions_get(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "10",
                                           "market_value": "1000"}])})
    monkeypatch.setattr(ab, "requests", fake)
    snap = _broker().snapshot(["AAA", "BBB"])
    assert snap.equity == 100000.0
    assert snap.market_values == {"AAA": 1000.0, "BBB": 0.0}  # BBB in universe but flat
    assert snap.qtys == {"AAA": 10.0, "BBB": 0.0}


def test_snapshot_includes_held_symbol_outside_universe(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "ZZZ", "qty": "5",
                                           "market_value": "500"}])})
    monkeypatch.setattr(ab, "requests", fake)
    snap = _broker().snapshot(["AAA"])  # ZZZ held but not in universe -> still folded in
    assert snap.qtys == {"AAA": 0.0, "ZZZ": 5.0}


def test_submit_sized_uses_fixed_equity_denominator(monkeypatch):
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "order-x"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0, "BBB": 0.0},
                           qtys={"AAA": 0.0, "BBB": 0.0})
    _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap)
    _broker().submit_sized(OrderIntent("BBB", Side.BUY, 0.3, T0), snap)
    # both sized off the SAME snapshot equity (100k), not a re-read that drifted as AAA filled
    assert fake.posted[0]["notional"] == "50000.00"
    assert fake.posted[1]["notional"] == "30000.00"


def test_submit_sized_rejects_symbol_outside_universe(monkeypatch):
    # #29: a typo'd symbol absent from the snapshot universe must raise, not size a phantom buy
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "x"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    with pytest.raises(BrokerError, match="not in the strategy universe"):
        _broker().submit_sized(OrderIntent("TYPOO", Side.BUY, 0.5, T0), snap)
    assert fake.posted == []


def test_non_2xx_raises_broker_error(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(403, text="forbidden")}))
    with pytest.raises(BrokerError):
        _broker().account()


class _RaisingRequests:
    def get(self, url, headers=None, timeout=None):
        raise ab.RequestException("connection reset")


class _BadJSONResp:
    status_code = 200

    def json(self):
        raise ValueError("not json")


def test_network_error_wrapped_in_broker_error(monkeypatch):
    monkeypatch.setattr(ab, "requests", _RaisingRequests())
    with pytest.raises(BrokerError, match="failed"):
        _broker().account()


def test_malformed_json_wrapped_in_broker_error(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests({"/v2/account": _BadJSONResp()}))
    with pytest.raises(BrokerError, match="malformed JSON"):
        _broker().account()


class _FakeRequestsWithDelete(_FakeRequests):
    def __init__(self, delete_resp):
        super().__init__({})
        self._delete_resp = delete_resp
        self.deleted = []

    def delete(self, url, headers=None, timeout=None):
        self.deleted.append(url)
        return self._delete_resp


def test_cancel_open_orders_ok(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"id": "a", "status": 200}]))
    monkeypatch.setattr(ab, "requests", fake)
    _broker().cancel_open_orders()
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/orders"]


def test_cancel_open_orders_non_2xx_raises(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(500, text="boom")))
    with pytest.raises(BrokerError):
        _broker().cancel_open_orders()


def test_cancel_open_orders_partial_failure_raises(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"id": "a", "status": 200},
                                                   {"id": "b", "status": 500}]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().cancel_open_orders()


def test_cancel_open_orders_non_dict_item_raises(monkeypatch):
    # a malformed (non-dict) 207 item must not pass silently
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"id": "a", "status": 200}, "oops"]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().cancel_open_orders()


class _FakeDeleteRouter(_FakeRequests):
    """DELETE returns a per-symbol status keyed off the URL's trailing path segment."""

    def __init__(self, status_by_symbol):
        super().__init__({})
        self.status_by_symbol = status_by_symbol
        self.deleted = []

    def delete(self, url, headers=None, timeout=None):
        self.deleted.append(url)
        symbol = url.rsplit("/", 1)[-1]
        status = self.status_by_symbol.get(symbol, 200)
        return _FakeResp(status, {"id": "o", "symbol": symbol}, text="err")


def test_close_positions_ok_is_scoped_per_symbol(monkeypatch):
    fake = _FakeDeleteRouter({"AAA": 200, "BBB": 200})
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_positions(["AAA", "BBB"])
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/positions/AAA",
                            "https://paper-api.alpaca.markets/v2/positions/BBB"]


def test_close_positions_empty_is_noop(monkeypatch):
    fake = _FakeDeleteRouter({})
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_positions([])  # no symbols -> no DELETE calls, no error
    assert fake.deleted == []


def test_close_positions_skips_404_no_position(monkeypatch):
    fake = _FakeDeleteRouter({"AAA": 404})  # no open position -> skip, no raise
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_positions(["AAA"])


def test_close_positions_failure_raises(monkeypatch):
    fake = _FakeDeleteRouter({"AAA": 200, "BBB": 500})
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().close_positions(["AAA", "BBB"])
