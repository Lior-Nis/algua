import copy
from datetime import UTC, datetime

import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution import alpaca_broker as ab
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.execution.alpaca_rejections import DEAD_ORDER_SKIP
from algua.execution.errors import BrokerError

T0 = datetime(2023, 1, 2, tzinfo=UTC)


def _no_sleep(monkeypatch):
    """Stub out the retry primitive's backoff sleep so retry tests run instantly. `_request` now
    delegates to algua.primitives.retry.call_with_backoff (Task 11), which no longer reads a
    module-level `time` off `ab` — the previous `monkeypatch.setattr(ab, "time", ...)` trick no
    longer has anything to patch, so we inject a no-op `sleep` into the retry call instead."""
    real = ab.call_with_backoff
    monkeypatch.setattr(
        ab, "call_with_backoff",
        lambda *a, **kw: real(*a, **{**kw, "sleep": lambda _s: None}),
    )


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
        self.redirects_allowed = []  # every call's allow_redirects flag, for the #394 assertion

    def get(self, url, headers=None, timeout=None, allow_redirects=None):
        self.redirects_allowed.append(allow_redirects)
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, text="not found")

    def post(self, url, headers=None, json=None, timeout=None, allow_redirects=None):
        self.redirects_allowed.append(allow_redirects)
        self.posted.append(json)
        return self.post_resp

    def delete(self, url, headers=None, timeout=None, allow_redirects=None):
        self.redirects_allowed.append(allow_redirects)
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


def test_request_disables_redirects(monkeypatch):
    # #394: every credential-bearing verb must pass allow_redirects=False so a 3xx cannot chase
    # the APCA headers to a foreign host.
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "a", "equity": "1", "cash": "0",
                                        "buying_power": "0"})})
    monkeypatch.setattr(ab, "requests", fake)
    _broker().account()
    assert fake.redirects_allowed == [False]


def test_request_does_not_follow_redirect(monkeypatch):
    # A 3xx is non-2xx, so it surfaces as a BrokerError instead of the creds being re-sent (#394).
    fake = _FakeRequests({"/v2/account": _FakeResp(302, text="redirect")})
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().account()


def test_account_parses_floats(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "acct-1", "equity": "100000", "cash": "50000",
                                        "buying_power": "150000"})}))
    acct = _broker().account()
    assert (acct.equity, acct.cash, acct.buying_power) == (100000.0, 50000.0, 150000.0)


def test_account_parses_last_equity(monkeypatch):
    # last_equity (prior trading-session close) flows into AccountState for the book breaker (#390).
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "acct-1", "equity": "90000", "cash": "0",
                                        "buying_power": "0", "last_equity": "100000"})}))
    assert _broker().account().last_equity == 100000.0


def test_account_missing_last_equity_defaults_zero(monkeypatch):
    # absent last_equity -> 0.0 so the daily-loss breaker fails closed, not crash account().
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"id": "acct-1", "equity": "90000", "cash": "0",
                                        "buying_power": "0"})}))
    assert _broker().account().last_equity == 0.0


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


def test_posted_notional_matches_submit_sized(monkeypatch):
    """`posted_notional(grant)` returns EXACTLY what a reserved BUY of `grant` posts through
    submit_sized: floor to cents (ROUND_FLOOR), and 0.0 (skip, no POST) below MIN_NOTIONAL. The
    paper run-all buying-power pool debits by this value, so the two MUST agree."""
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})

    # (a) a fractional grant posts the floored cents; posted_notional agrees
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "order-x"}))
    monkeypatch.setattr(ab, "requests", fake)
    oid = _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                 reserve=lambda _s, _a: 100.999)
    assert oid == "order-x"
    assert fake.posted[0]["notional"] == "100.99"
    assert ab.posted_notional(100.999) == 100.99

    # (b) a sub-MIN_NOTIONAL grant is skipped (no POST); posted_notional is 0.0
    fake2 = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "nope"}))
    monkeypatch.setattr(ab, "requests", fake2)
    assert _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                                  reserve=lambda _s, _a: 0.50) == "skipped"
    assert fake2.posted == []
    assert ab.posted_notional(0.50) == 0.0


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
    def get(self, url, headers=None, timeout=None, allow_redirects=None):
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

    def delete(self, url, headers=None, timeout=None, allow_redirects=None):
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

    def get(self, url, headers=None, timeout=None, allow_redirects=None):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            return _FakeResp(503, text="unavailable")
        return _FakeResp(200, self.final_payload)


def test_retries_transient_status_then_succeeds(monkeypatch):
    _no_sleep(monkeypatch)
    fake = _FlakyRequests(fail_times=2,
                          final_payload={"id": "a", "equity": "1",
                                         "cash": "1", "buying_power": "1"})
    monkeypatch.setattr(ab, "requests", fake)
    acct = _broker().account()
    assert acct.equity == 1.0 and fake.attempts == 3  # 2 failures + 1 success


def test_retries_exhausted_raises(monkeypatch):
    _no_sleep(monkeypatch)
    fake = _FlakyRequests(fail_times=99, final_payload={})  # always 503
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError, match="503"):
        _broker().account()


class _RetryableThenRaise(_FakeRequests):
    def __init__(self):
        super().__init__({})
        self.attempts = 0

    def get(self, url, headers=None, timeout=None, allow_redirects=None):
        self.attempts += 1
        raise ab.RequestException("timeout")


def test_retries_transport_error_then_raises(monkeypatch):
    _no_sleep(monkeypatch)
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

    def post(self, url, headers=None, json=None, timeout=None, allow_redirects=None):
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
    _no_sleep(monkeypatch)
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

    def delete(self, url, headers=None, timeout=None, allow_redirects=None):
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
    monkeypatch.setattr(
        fake, "delete",
        lambda url, headers=None, timeout=None, allow_redirects=None: _FakeResp(204),
    )
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


def test_submit_offset_qty_precise_no_scientific_notation(monkeypatch):
    # #269: the offset qty must be exact fixed-point, never `format(qty,'g')` (6 sig figs /
    # scientific notation). A fractional believed qty must flatten in full, and a large qty must
    # not render as '2.5e+06'.
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "off"}))
    monkeypatch.setattr(ab, "requests", fake)
    b = _broker()
    b.submit_offset("AAA", 12.345678901, "c1")     # fractional (9 dp) -> kept in full
    assert fake.posted[-1]["qty"] == "12.345678901"
    b.submit_offset("BBB", 2_500_000.0, "c2")      # large -> NOT '2.5e+06'
    assert fake.posted[-1]["qty"] == "2500000"
    b.submit_offset("CCC", 0.000123456, "c3")      # small -> full precision, no sci notation
    assert fake.posted[-1]["qty"] == "0.000123456"
    b.submit_offset("DDD", 0.1 + 0.2, "c4")        # float-sum noise -> flattens to 0.3 (9 dp)
    assert fake.posted[-1]["qty"] == "0.3"


def test_submit_offset_subnano_residual_is_noop(monkeypatch):
    # #269: a believed residual below Alpaca's 9-dp precision is effectively flat — return noop,
    # never POST a malformed qty='0' order. No HTTP call is made.
    fake = _FakeRequests({}, post_resp=_FakeResp(201, {"id": "off"}))
    monkeypatch.setattr(ab, "requests", fake)
    assert _broker().submit_offset("AAA", 1e-12, "c") == "noop"
    assert fake.posted == []


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


def test_submit_sized_floors_reserved_notional_never_rounds_up(monkeypatch):
    # #389 GATE-2: a risk-reserved BUY amount is granted against book-level headroom; the submitted
    # notional must be floored to cents, never rounded UP, or the actual gross could edge a fraction
    # of a cent PAST a book cap the accumulator believes is exactly met. 12345.678 -> "12345.67".
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions": _FakeResp(200, [])},
        post_resp=_FakeResp(201, {"id": "o1"}))
    monkeypatch.setattr(ab, "requests", fake)
    snap = ab.TickSnapshot(equity=100000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    _broker().submit_sized(OrderIntent("AAA", Side.BUY, 0.5, T0), snap,
                           reserve=lambda sym, n: 12_345.678)
    assert fake.posted[0]["notional"] == "12345.67"  # floored down, NOT "12345.68"


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

    def get(self, url, headers=None, timeout=None, allow_redirects=None):
        self.requested_urls.append(url)
        for prefix, pages in self._pages.items():
            if prefix in url:
                if pages:
                    return pages.pop(0)
                return _FakeResp(500, text="no more pages configured")
        return _FakeResp(404, text="not found")

    def post(  # pragma: no cover
        self, url, headers=None, json=None, timeout=None, allow_redirects=None
    ):
        return _FakeResp(405, text="unexpected POST")

    def delete(  # pragma: no cover
        self, url, headers=None, timeout=None, allow_redirects=None
    ):
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
    from algua.execution.alpaca_broker import AlpacaLiveReadOnlyBroker
    from algua.execution.errors import BrokerError

    # accepts the live host with no LiveAuthorization
    b = AlpacaLiveReadOnlyBroker("k", "s", base_url="https://api.alpaca.markets")
    assert b.base_url == "https://api.alpaca.markets"
    # refuses a non-live / non-https host (the platform invariant)
    import pytest
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="https://paper-api.alpaca.markets")
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="http://api.alpaca.markets")


def test_get_order_by_client_order_id_returns_dict(monkeypatch):
    # #312 recovery primitive: a matching order is returned as a dict (any status).
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/orders:by_client_order_id?client_order_id=coid-1":
            _FakeResp(200, {"id": "b-9", "client_order_id": "coid-1", "symbol": "AAA",
                            "status": "filled"})}))
    order = _broker().get_order_by_client_order_id("coid-1")
    assert order is not None and order["id"] == "b-9" and order["symbol"] == "AAA"


def test_get_order_by_client_order_id_404_returns_none(monkeypatch):
    # A coid never accepted at the venue -> 404 -> None (recovery preserves the local row).
    monkeypatch.setattr(ab, "requests", _FakeRequests({}))  # every GET -> 404
    assert _broker().get_order_by_client_order_id("missing") is None


def test_get_order_by_client_order_id_non_dict_raises(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/orders:by_client_order_id?client_order_id=coid-1":
            _FakeResp(200, ["not", "a", "dict"])}))
    with pytest.raises(BrokerError, match="expected an order object"):
        _broker().get_order_by_client_order_id("coid-1")


def test_get_order_by_client_order_id_encodes_coid(monkeypatch):
    # The coid is url-encoded (a sanitized coid can carry reserved chars); no raw char leaks.
    fake = _FakeRequests({"client_order_id=a%2Fb": _FakeResp(200, {"id": "b1"})})
    monkeypatch.setattr(ab, "requests", fake)
    assert _broker().get_order_by_client_order_id("a/b") == {"id": "b1"}


# --- #560: a duplicate client_order_id must not abort the whole cycle -------------------------
#
# Measured on the live box before this change: 18 aborts in 6 hours, every one of them this
# rejection, with the paper lane producing nothing for days. `paper run-all` ticks every tenant
# inside ONE cycle, so a single 422 took down all of them.


class _DuplicatingRequests(_FakeRequests):
    """Alpaca's real behaviour: the first POST of a client_order_id lands, every later POST of the
    same id is rejected 422. Keeps the ids it has seen so a test can submit twice and get the
    genuine sequence rather than a hardcoded response."""

    def __init__(self, routes, first_resp, duplicate_resp):
        super().__init__(routes, post_resp=first_resp)
        self.duplicate_resp = duplicate_resp
        self._seen: set[str] = set()

    def post(self, url, headers=None, json=None, timeout=None, allow_redirects=None):
        self.posted.append(json)
        self.redirects_allowed.append(allow_redirects)
        coid = (json or {}).get("client_order_id")
        if coid is not None and coid in self._seen:
            return self.duplicate_resp
        if coid is not None:
            self._seen.add(coid)
        return self.post_resp


_DUPLICATE_422 = _FakeResp(
    422, text='{"code":42210000,"message":"client_order_id must be unique"}')


def _venue_order(**over):
    """An Alpaca order object as the by-client-order-id lookup returns it."""
    return {"id": "order-abc", "client_order_id": "coid-1", "symbol": "AAPL",
            "side": "buy", "qty": "1", **over}


def _snap_routes():
    return {"/v2/account": _FakeResp(200, {"id": "acct-1", "equity": "10000", "cash": "10000",
                                           "buying_power": "10000", "last_equity": "10000"}),
            "/v2/positions": _FakeResp(200, [])}


def _intent():
    return OrderIntent("AAPL", Side.BUY, 0.5, T0)


def _recovering(returned_order=None, first=None):
    routes = _snap_routes()
    routes["/v2/orders:by_client_order_id?client_order_id=coid-1"] = _FakeResp(
        200, _venue_order() if returned_order is None else returned_order)
    return _DuplicatingRequests(routes, first or _FakeResp(200, _venue_order()), _DUPLICATE_422)


def test_a_duplicate_resolves_to_the_order_that_already_landed(monkeypatch):
    """The regression. Before this, the second submit raised and took the cycle with it."""
    fake = _recovering()
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    first = broker.submit_sized(_intent(), snap, "coid-1")
    second = broker.submit_sized(_intent(), snap, "coid-1")
    assert first == "order-abc"
    assert second == "order-abc", "a retried submit must resolve to the order already on the venue"


def test_recovery_reads_and_never_re_posts(monkeypatch):
    """Minting a fresh id would turn one order into two in the live lane. The duplicate id IS the
    safety mechanism."""
    fake = _recovering()
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(_intent(), snap, "coid-1")
    broker.submit_sized(_intent(), snap, "coid-1")
    assert [p.get("client_order_id") for p in fake.posted] == ["coid-1", "coid-1"]


def test_a_422_that_is_not_a_duplicate_still_raises(monkeypatch):
    """Insufficient buying power is also a 422. Swallowing it would report a phantom order id for
    an order that never existed — strictly worse than the abort this removes."""
    fake = _FakeRequests(_snap_routes(), post_resp=_FakeResp(
        422, text='{"code":40310000,"message":"insufficient buying power"}'))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    with pytest.raises(BrokerError, match="422"):
        broker.submit_sized(_intent(), snap, "coid-1")


def test_a_submit_without_a_coid_cannot_take_the_recovery_path(monkeypatch):
    """With no id there is nothing to look up; such a 422 must raise like any other failure."""
    fake = _FakeRequests(_snap_routes(), post_resp=_DUPLICATE_422)
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    with pytest.raises(BrokerError, match="422"):
        broker.submit_sized(_intent(), snap, None)


def test_a_duplicate_the_venue_cannot_produce_raises(monkeypatch):
    """The venue said the id was taken and then 404ed on it. Inventing an id here would record a
    fill against an order nobody can reconcile."""
    routes = _snap_routes()
    routes["/v2/orders:by_client_order_id?client_order_id=coid-1"] = _FakeResp(404, text="nope")
    fake = _DuplicatingRequests(routes, _FakeResp(200, _venue_order()), _DUPLICATE_422)
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(_intent(), snap, "coid-1")
    with pytest.raises(BrokerError, match="no order carries it"):
        broker.submit_sized(_intent(), snap, "coid-1")


@pytest.mark.parametrize("bad, why", [
    ({"symbol": "MSFT"}, "a mis-addressed lookup must never attribute another symbol's order"),
    ({"side": "sell"}, "accepting a buy as a sell would misreport a position as closed"),
    ({"client_order_id": "someone-else"}, "the venue echoed a different id"),
    ({"id": ""}, "an empty broker id would be recorded as a real one"),
    ({"id": 12345}, "a truthy non-str would coerce to a bogus broker id"),
])
def test_a_recovered_order_that_is_not_ours_is_refused(monkeypatch, bad, why):
    """The recovered payload is a SAFETY BOUNDARY, as in #312 stranded-order recovery."""
    fake = _recovering(_venue_order(**bad))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(_intent(), snap, "coid-1")
    with pytest.raises(BrokerError, match="does not match the submitted order"):
        broker.submit_sized(_intent(), snap, "coid-1")


def test_the_liquidation_path_recovers_too_and_checks_side(monkeypatch):
    """submit_offset is the dangerous one: flatten counts what comes back as the liquidation."""
    ok = _recovering(_venue_order(side="sell", qty="5"), first=_FakeResp(200, _venue_order()))
    monkeypatch.setattr(ab, "requests", ok)
    assert _broker().submit_offset("AAPL", 5.0, "coid-1") == "order-abc"
    assert _broker().submit_offset("AAPL", 5.0, "coid-1") == "order-abc"

    wrong = _recovering(_venue_order(side="buy", qty="5"))
    monkeypatch.setattr(ab, "requests", wrong)
    broker = _broker()
    broker.submit_offset("AAPL", 5.0, "coid-1")
    with pytest.raises(BrokerError, match="does not match the submitted order"):
        broker.submit_offset("AAPL", 5.0, "coid-1")


def test_a_differently_sized_resubmit_still_recovers(monkeypatch):
    """Size is deliberately NOT compared: the same decision re-run under a changed allocation posts
    a different notional for the same coid, and the landed order is still the right answer.

    The SUBMITTED notional must actually differ between the two calls, or this tests nothing -- an
    earlier version varied only the fake lookup payload and passed for the wrong reason.
    """
    fake = _recovering(_venue_order(qty="999"))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(OrderIntent("AAPL", Side.BUY, 0.25, T0), snap, "coid-1")
    assert broker.submit_sized(
        OrderIntent("AAPL", Side.BUY, 0.75, T0), snap, "coid-1") == "order-abc"
    posted = [p.get("notional") for p in fake.posted]
    assert posted[0] != posted[1], "the two submits must genuinely differ in size"


@pytest.mark.parametrize("status", ["canceled", "cancelled", "expired", "rejected", "replaced"])
def test_a_dead_order_is_skipped_not_reported_as_a_submission(monkeypatch, status):
    """THE cancel-then-recover hole. Every tick cancels open orders before its submit loop, so a
    same-decision re-run finds the id held by the order it just cancelled.

    Two wrong answers, both tried: reporting it as this tick's submission corrupts the ledger with a
    dead order, and RAISING re-creates the cycle abort this module exists to remove — observed live
    every 20 minutes on `cross_sectional_momentum-20260921T000000Z-AAPL`. The id cannot be reused,
    so the leg simply gets no order for this decision.
    """
    fake = _recovering(_venue_order(status=status))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(_intent(), snap, "coid-1")
    assert broker.submit_sized(_intent(), snap, "coid-1") == DEAD_ORDER_SKIP


@pytest.mark.parametrize("status", ["filled", "partially_filled", "new", "accepted"])
def test_recovery_accepts_an_order_that_really_reached_the_book(monkeypatch, status):
    """Filled and partially-filled orders DID land -- that is what recovery is for. An unknown
    status is accepted too: refusing it would re-create the cycle abort, and the per-tick reconcile
    is the backstop."""
    fake = _recovering(_venue_order(status=status))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    broker.submit_sized(_intent(), snap, "coid-1")
    assert broker.submit_sized(_intent(), snap, "coid-1") == "order-abc"


def test_a_recovered_order_refunds_its_buying_power_reservation(monkeypatch):
    """The pool is debited BEFORE the POST. A recovery placed no NEW order -- the account's buying
    power already reflects the earlier one -- so keeping the debit double-counts it and trims or
    skips later tenants in the same cycle for buying power nothing consumed."""
    fake = _recovering()
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    pool = {"available": 10_000.0}
    released: list = []

    def _reserve(symbol, notional):
        grant = min(notional, pool["available"])
        pool["available"] -= ab.posted_notional(grant)
        return grant

    def _release(symbol, notional):
        released.append((symbol, notional))
        pool["available"] += notional

    broker.submit_sized(_intent(), snap, "coid-1", reserve=_reserve, release=_release)
    after_first = pool["available"]
    assert released == [], "a genuinely new order must NOT be refunded"

    broker.submit_sized(_intent(), snap, "coid-1", reserve=_reserve, release=_release)
    assert len(released) == 1, "the re-identified submit must refund exactly once"
    assert pool["available"] == pytest.approx(after_first), "the pool must be left unchanged"


def test_only_a_genuinely_new_order_reports_posted_new(monkeypatch):
    """The flag must distinguish the two events, or the refund fires on every submit — or never."""
    fake = _recovering()
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    body = {"symbol": "AAPL", "side": "buy", "client_order_id": "coid-1"}
    _id, posted_new = broker._post_order(body, "/v2/orders", "coid-1")
    assert posted_new is True, "the first submit really does reach the venue"
    _id2, posted_new2 = broker._post_order(body, "/v2/orders", "coid-1")
    assert posted_new2 is False, "a re-identified order placed nothing new this cycle"


@pytest.mark.parametrize("status_code, text, why", [
    (403, '{"code":42210000,"message":"client_order_id must be unique"}',
     "only a 422 is this rejection; a 403 is authorization"),
    (422, '{"code":40310000,"message":"quantity must be unique per something"}',
     "no mention of client_order_id"),
    (422, '{"code":42210000,"message":"client_order_id is malformed"}',
     "mentions the field but does not mean duplicate"),
])
def test_each_half_of_the_classifier_is_load_bearing(status_code, text, why):
    """Every conjunct must matter on its own. A single negative case that mismatches ALL THREE at
    once leaves the individual conjuncts unkilled by mutation -- which is what an earlier version
    of this file did."""
    assert not ab.is_duplicate_client_order_id(status_code, text), why


def test_the_classifier_accepts_the_real_rejection():
    assert ab.is_duplicate_client_order_id(
        422, '{"code":42210000,"message":"client_order_id must be unique"}')


def test_a_skipped_dead_order_also_refunds_its_reservation(monkeypatch):
    """No order exists for this leg, so the cycle consumed nothing — keeping the debit would starve
    a later tenant of buying power nothing used."""
    fake = _recovering(_venue_order(status="canceled"))
    monkeypatch.setattr(ab, "requests", fake)
    broker = _broker()
    snap = broker.snapshot(["AAPL"])
    pool = {"available": 10_000.0}

    def _reserve(symbol, notional):
        grant = min(notional, pool["available"])
        pool["available"] -= ab.posted_notional(grant)
        return grant

    def _release(symbol, notional):
        pool["available"] += notional

    broker.submit_sized(_intent(), snap, "coid-1", reserve=_reserve, release=_release)
    before = pool["available"]
    assert broker.submit_sized(
        _intent(), snap, "coid-1", reserve=_reserve, release=_release) == DEAD_ORDER_SKIP
    assert pool["available"] == pytest.approx(before)


def test_the_skip_sentinel_is_one_the_tick_loop_already_retracts():
    """`live_loop` retracts the phantom intent row for exactly these sentinels. A novel value would
    leave a NULL-broker_order_id row that stranded-order recovery re-queries forever."""
    assert DEAD_ORDER_SKIP in ("noop", "skipped")
