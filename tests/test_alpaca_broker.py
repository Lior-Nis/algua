import copy
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
    """Routes GET and DELETE by URL suffix; records POST bodies."""

    def __init__(self, routes, post_resp=None):
        self.routes = routes
        self.post_resp = post_resp
        self.posted = []
        self.deleted = []

    def get(self, url, headers=None, timeout=None):
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, text="not found")

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted.append(json)
        return self.post_resp

    def delete(self, url, headers=None, timeout=None):
        self.deleted.append(url)
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, text="not found")


def _broker():
    return AlpacaPaperBroker(api_key="k", api_secret="s")


def test_rejects_non_paper_base_url():
    # #28: the platform invariant is paper-only, never live — constructing against the live host
    # (or any non-paper host) must be impossible, not a warning.
    with pytest.raises(BrokerError, match="refusing"):
        AlpacaPaperBroker(api_key="k", api_secret="s", base_url="https://api.alpaca.markets")


def test_accepts_paper_base_url():
    b = AlpacaPaperBroker(api_key="k", api_secret="s",
                          base_url="https://paper-api.alpaca.markets/")
    assert b.base_url == "https://paper-api.alpaca.markets"


def test_alpaca_broker_conforms_to_broker_protocol():
    from algua.contracts.types import Broker
    assert isinstance(_broker(), Broker)


def test_account_parses_floats(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "acct-1", "equity": "100000", "cash": "50000",
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
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
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
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "600",
                                           "market_value": "60000"}])},
        post_resp=_FakeResp(201, {"id": "order-2"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    _broker().submit(OrderIntent("AAA", Side.SELL, 0.5, T0))
    assert fake.posted[0]["side"] == "sell" and fake.posted[0]["notional"] == "10000.00"


def test_submit_noop_below_threshold(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "500",
                                           "market_value": "50000"}])},
        post_resp=_FakeResp(201, {"id": "unused"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    assert _broker().submit(OrderIntent("AAA", Side.BUY, 0.5, T0)) == "noop"
    assert fake.posted == []


def test_snapshot_is_one_account_and_one_positions_get(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "10",
                                           "market_value": "1000"}])})
    monkeypatch.setattr(ab, "requests", fake)
    snap = _broker().snapshot(["AAA", "BBB"])
    assert snap.equity == 100000.0
    assert snap.market_values == {"AAA": 1000.0, "BBB": 0.0}  # BBB in universe but flat
    assert snap.qtys == {"AAA": 10.0, "BBB": 0.0}


def test_snapshot_includes_held_symbol_outside_universe(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
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


def test_cancel_open_orders_207_non_list_body_raises(monkeypatch):
    # #22: a 207 whose body is not a per-order list must raise, not be read as "all cancelled".
    fake = _FakeRequestsWithDelete(_FakeResp(207, {"message": "service degraded"}))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError, match="non-list body"):
        _broker().cancel_open_orders()


def test_cancel_open_orders_non_int_status_is_failure(monkeypatch):
    # #22: a present-but-non-int status must count as a failure, not raise an uncaught ValueError.
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"id": "a", "status": "bad"}]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError, match="failed to cancel"):
        _broker().cancel_open_orders()


class _FlakyRequests(_FakeRequests):
    """First N GET calls return a retryable status, then a 200. Records attempt count."""

    def __init__(self, fail_times, final_payload):
        super().__init__({})
        self.fail_times = fail_times
        self.final_payload = final_payload
        self.attempts = 0

    def get(self, url, headers=None, timeout=None):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            return _FakeResp(503, text="unavailable")
        return _FakeResp(200, self.final_payload)


def test_retries_transient_status_then_succeeds(monkeypatch):
    monkeypatch.setattr(ab, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
    fake = _FlakyRequests(fail_times=2,
                          final_payload={"id": "a", "equity": "1",
                                         "cash": "1", "buying_power": "1"})
    monkeypatch.setattr(ab, "requests", fake)
    acct = _broker().account()
    assert acct.equity == 1.0 and fake.attempts == 3  # 2 failures + 1 success


def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(ab, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
    fake = _FlakyRequests(fail_times=99, final_payload={})  # always 503
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError, match="503"):
        _broker().account()


class _RetryableThenRaise(_FakeRequests):
    def __init__(self):
        super().__init__({})
        self.attempts = 0

    def get(self, url, headers=None, timeout=None):
        self.attempts += 1
        raise ab.RequestException("timeout")


def test_retries_transport_error_then_raises(monkeypatch):
    monkeypatch.setattr(ab, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
    fake = _RetryableThenRaise()
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError, match="after 3 attempts"):
        _broker().account()
    assert fake.attempts == 3


class _TimeoutThenSucceedPost(_FakeRequests):
    """POST raises a transport error (timeout) on the first attempt, then returns 201 on the
    retry. Records EVERY attempt's body so we can assert the retried POST is byte-identical."""

    def __init__(self, ok_payload):
        super().__init__({})
        self._ok = _FakeResp(201, ok_payload)
        self.attempts = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.attempts += 1
        # Deep-copy so we capture the body AS SENT on each attempt — not a shared reference that
        # would make the two-attempts-equal assertion tautological if production mutated it.
        self.posted.append(copy.deepcopy(json))
        if self.attempts == 1:
            raise ab.RequestException("submit timed out")
        return self._ok


def test_submit_timeout_then_retry_reuses_client_order_id(monkeypatch):
    # #166 gap 2: a submit POST that times out is retried by _request (transport errors are always
    # retried). The retried POST must re-send the SAME deterministic client_order_id so Alpaca
    # de-duplicates the order rather than double-filling. This unit test proves the CLIENT-SIDE
    # guarantee (identical body incl. client_order_id on the retry) + a single returned order id;
    # the actual no-double-order is Alpaca's server-side dedup on client_order_id (out of scope).
    monkeypatch.setattr(ab, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
    fake = _TimeoutThenSucceedPost({"id": "order-1"})
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})

    oid = _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                 client_order_id="cfg-20230102-AAA")

    assert oid == "order-1"                    # one logical order, not two
    assert fake.attempts == 2                  # first timed out, retry succeeded
    assert len(fake.posted) == 2 and fake.posted[0] == fake.posted[1]   # byte-identical resubmit
    assert fake.posted[0]["client_order_id"] == "cfg-20230102-AAA"


def test_submit_sized_passes_client_order_id(monkeypatch):
    # #18/#24: the deterministic client_order_id is sent so a retried submit is idempotent.
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "order-x"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                           client_order_id="cfg-2023-06-01-AAA")
    assert fake.posted[0]["client_order_id"] == "cfg-2023-06-01-AAA"


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


def test_close_all_positions_ok(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200}]))
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_all_positions()
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/positions?cancel_orders=true"]


def test_close_all_positions_empty_is_noop(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(207, [])))
    _broker().close_all_positions()  # no positions -> no error


def test_close_all_positions_partial_failure_raises(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200},
                                                   {"symbol": "BBB", "status": 500}]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().close_all_positions()


def test_close_all_positions_non_list_body_raises(monkeypatch):
    # a panic flatten must not treat an unexpected (non-list) body as success
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(207, {"oops": 1})))
    with pytest.raises(BrokerError):
        _broker().close_all_positions()


def test_base_broker_alone_reaches_no_venue():
    # the base class has an EMPTY allowed-host set, so it can't be constructed against any host
    from algua.execution.alpaca_broker import _AlpacaBroker
    with pytest.raises(BrokerError):
        _AlpacaBroker("k", "s", "https://paper-api.alpaca.markets")


def _live_auth():
    from algua.contracts.types import LiveAuthorization
    return LiveAuthorization(strategy_id=1, code_hash="c", config_hash="cf",
                             dependency_hash="d", principal="lior", authorized_at="2026-06-05")


def test_live_broker_requires_authorization():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    with pytest.raises(BrokerError, match="LiveAuthorization"):
        AlpacaLiveBroker(None, "k", "s")  # no token
    with pytest.raises(BrokerError):
        AlpacaLiveBroker({"principal": "lior"}, "k", "s")  # a look-alike dict is not the token


def test_live_broker_constructs_with_authorization_and_live_host():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    b = AlpacaLiveBroker(_live_auth(), "k", "s")
    assert b.base_url == "https://api.alpaca.markets"
    assert b.authorization.principal == "lior"


def test_live_broker_rejects_non_live_host():
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    with pytest.raises(BrokerError, match="refusing"):
        AlpacaLiveBroker(_live_auth(), "k", "s", base_url="https://paper-api.alpaca.markets")


def test_live_broker_account_uses_inherited_rest(monkeypatch):
    from algua.execution import alpaca_broker as ab
    from algua.execution.alpaca_broker import AlpacaLiveBroker
    fake = _FakeRequests({"/v2/account": _FakeResp(200, {"id": "live-1", "equity": "5",
                                                         "cash": "5", "buying_power": "5"})})
    monkeypatch.setattr(ab, "requests", fake)
    acct = AlpacaLiveBroker(_live_auth(), "k", "s").account()
    assert acct.equity == 5.0


def test_brokers_reject_http_plaintext():
    # API keys must never travel over plaintext (codex review)
    from algua.execution.alpaca_broker import AlpacaLiveBroker, AlpacaPaperBroker
    with pytest.raises(BrokerError, match="https"):
        AlpacaPaperBroker("k", "s", "http://paper-api.alpaca.markets")
    with pytest.raises(BrokerError, match="https"):
        AlpacaLiveBroker(_live_auth(), "k", "s", "http://api.alpaca.markets")


def test_account_activities_reads_list(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account/activities": _FakeResp(200, [{"id": "a1", "activity_type": "FILL"}])}))
    acts = _broker().account_activities()
    assert acts == [{"id": "a1", "activity_type": "FILL"}]


def test_list_open_orders(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/orders?status=open": _FakeResp(200, [{"id": "o1", "client_order_id": "c1"}])}))
    assert _broker().list_open_orders() == [{"id": "o1", "client_order_id": "c1"}]


def test_cancel_order_by_id(monkeypatch):
    fake = _FakeRequests({})
    monkeypatch.setattr(ab, "requests", fake)
    # a 204 (or 200) is success; a 404 (already gone) is a no-op, not an error
    monkeypatch.setattr(fake, "delete", lambda url, headers=None, timeout=None: _FakeResp(204))
    _broker().cancel_order("o1")  # must not raise


def test_submit_offset_posts_qty_order(monkeypatch):
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "off-1"}))
    monkeypatch.setattr(ab, "requests", fake)
    oid = _broker().submit_offset("AAA", 7.0, "coid-flat")      # long 7 -> SELL 7
    assert oid == "off-1"
    assert fake.posted[0] == {"symbol": "AAA", "qty": "7", "side": "sell", "type": "market",
                              "time_in_force": "day", "client_order_id": "coid-flat"}
    _broker().submit_offset("BBB", -3.0, "coid-cover")          # short 3 -> BUY 3
    assert fake.posted[1]["side"] == "buy" and fake.posted[1]["qty"] == "3"


def test_submit_sized_reserve_trims_buy(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    # reserve grants only 20k of the intended 50k -> posted notional is trimmed to 20000.00
    _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                           reserve=lambda sym, n: 20_000.0)
    assert fake.posted[0]["notional"] == "20000.00"


def test_submit_sized_reserve_zero_skips_buy(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    assert _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                  reserve=lambda sym, n: 0.0) == "skipped"
    assert fake.posted == []   # nothing posted


def test_submit_sized_reserve_ignores_sells(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "100000",
                                        "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "600",
                                           "market_value": "60000"}])},
        post_resp=_FakeResp(201, {"id": "o2"}))
    monkeypatch.setattr(ab, "requests", fake)
    # a SELL toward 0.5 must not consult reserve (a reserve returning 0 would wrongly skip a sell)
    _broker().submit_sized(OrderIntent("AAA", Side.SELL, 0.5, T0),
                           _broker().snapshot(["AAA"]), reserve=lambda sym, n: 0.0)
    assert fake.posted[0]["side"] == "sell"


def test_submit_sized_reserve_below_min_notional_skips(monkeypatch):
    # a buy trimmed below Alpaca's $1 minimum must skip, not post a sub-$1 order that gets rejected
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    assert _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                  reserve=lambda sym, n: 0.50) == "skipped"
    assert fake.posted == []


# ---------------------------------------------------------------------------
# #124 forward-test gate: clock, account id, windowed activities
# ---------------------------------------------------------------------------

class _PaginatingRequests:
    """Serves GET requests from a list of responses per URL prefix, recording requested URLs."""

    def __init__(self, pages_by_prefix: dict[str, list[_FakeResp]]):
        self._pages = {k: list(v) for k, v in pages_by_prefix.items()}
        self.requested_urls: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requested_urls.append(url)
        for prefix, pages in self._pages.items():
            if prefix in url:
                if pages:
                    return pages.pop(0)
                return _FakeResp(500, text="no more pages configured")
        return _FakeResp(404, text="not found")

    def post(self, url, headers=None, json=None, timeout=None):  # pragma: no cover
        return _FakeResp(405, text="unexpected POST")

    def delete(self, url, headers=None, timeout=None):  # pragma: no cover
        return _FakeResp(405, text="unexpected DELETE")


def test_clock_returns_broker_timestamp(monkeypatch):
    # Happy path: GET /v2/clock returns ISO timestamp string.
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/clock": _FakeResp(200, {"timestamp": "2026-06-11T14:00:00-04:00",
                                      "is_open": True})}))
    ts = _broker().clock()
    assert ts == "2026-06-11T14:00:00-04:00"


def test_clock_malformed_body_raises(monkeypatch):
    # Body missing "timestamp" must raise BrokerError — skewed local clock cannot be silently used.
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/clock": _FakeResp(200, {"is_open": True})}))
    with pytest.raises(BrokerError, match="missing timestamp"):
        _broker().clock()


def test_account_carries_id(monkeypatch):
    # account() must surface account_id from the "id" field.
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "abc-123", "equity": "100000",
                                        "cash": "50000", "buying_power": "150000"})}))
    acct = _broker().account()
    assert acct.account_id == "abc-123"


def test_account_missing_id_raises(monkeypatch):
    # Missing "id" must raise — an account without an identity cannot anchor hygiene checks.
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "50000",
                                        "buying_power": "150000"})}))
    with pytest.raises(BrokerError, match="missing account id"):
        _broker().account()


def test_account_empty_id_raises(monkeypatch):
    # Explicitly empty "id" must also raise.
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "", "equity": "100000",
                                        "cash": "50000", "buying_power": "150000"})}))
    with pytest.raises(BrokerError, match="missing account id"):
        _broker().account()


def _make_page(n: int, start_idx: int = 0) -> list[dict]:
    """Return n activity dicts with sequential id fields."""
    return [{"id": f"act-{start_idx + i}", "activity_type": "FILL"} for i in range(n)]


def test_activities_window_paginates(monkeypatch):
    """Three pages: 100 + 100 + 3 = 203 items; page_token forwarded on calls 2 and 3."""
    page1 = _make_page(100, start_idx=0)
    page2 = _make_page(100, start_idx=100)
    page3 = _make_page(3, start_idx=200)
    fake = _PaginatingRequests({"/v2/account/activities": [
        _FakeResp(200, page1),
        _FakeResp(200, page2),
        _FakeResp(200, page3),
    ]})
    monkeypatch.setattr(ab, "requests", fake)
    result = _broker().account_activities_window(after="2026-01-01", until="2026-06-11")
    # All 203 items concatenated in order.
    assert len(result) == 203
    assert result[0]["id"] == "act-0"
    assert result[100]["id"] == "act-100"
    assert result[200]["id"] == "act-200"
    # page_token from page1's last item ("act-99") on call 2, from page2's last ("act-199") on 3.
    assert "page_token=act-99" in fake.requested_urls[1]
    assert "page_token=act-199" in fake.requested_urls[2]


def test_activities_window_fails_closed(monkeypatch):
    # (a) non-list body raises BrokerError
    monkeypatch.setattr(ab, "requests", _PaginatingRequests(
        {"/v2/account/activities": [_FakeResp(200, {"message": "error"})]}))
    with pytest.raises(BrokerError, match="expected list"):
        _broker().account_activities_window(after="2026-01-01", until="2026-06-11")

    # (b) a FULL page (100 items) whose last item lacks "id" raises BrokerError
    full_page_no_id = _make_page(100)
    full_page_no_id[-1] = {"activity_type": "FILL"}  # last item missing "id"
    monkeypatch.setattr(ab, "requests", _PaginatingRequests(
        {"/v2/account/activities": [_FakeResp(200, full_page_no_id)]}))
    with pytest.raises(BrokerError, match="cannot paginate exhaustively"):
        _broker().account_activities_window(after="2026-01-01", until="2026-06-11")

    # (c) a non-dict item in the list raises BrokerError
    mixed_page = _make_page(3) + ["not-a-dict"]
    monkeypatch.setattr(ab, "requests", _PaginatingRequests(
        {"/v2/account/activities": [_FakeResp(200, mixed_page)]}))
    with pytest.raises(BrokerError, match="malformed item"):
        _broker().account_activities_window(after="2026-01-01", until="2026-06-11")


def test_activities_window_encodes_tz_offset(monkeypatch):
    # The forward gate passes full ISO-8601 datetimes with '+00:00' offsets.  A raw '+' in a
    # query string is decoded as a SPACE server-side, silently corrupting the window bound.
    # Verify that after/until are percent-encoded so the URL never contains a literal '+'.
    fake = _PaginatingRequests({"/v2/account/activities": [_FakeResp(200, _make_page(1))]})
    monkeypatch.setattr(ab, "requests", fake)
    _broker().account_activities_window(
        after="2026-06-01T00:00:00+00:00",
        until="2026-06-11T14:00:00+00:00",
    )
    url = fake.requested_urls[0]
    # The timezone offset '+' must be encoded; no raw '+' may appear in the query string.
    assert "%2B" in url, f"expected %2B in URL, got: {url}"
    assert "+" not in url.split("?", 1)[1], f"raw '+' found in query string: {url}"


def test_live_readonly_broker_requires_live_https_host():
    from algua.execution.alpaca_broker import AlpacaLiveReadOnlyBroker, BrokerError

    # accepts the live host with no LiveAuthorization
    b = AlpacaLiveReadOnlyBroker("k", "s", base_url="https://api.alpaca.markets")
    assert b.base_url == "https://api.alpaca.markets"
    # refuses a non-live / non-https host (the platform invariant)
    import pytest
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="https://paper-api.alpaca.markets")
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="http://api.alpaca.markets")
