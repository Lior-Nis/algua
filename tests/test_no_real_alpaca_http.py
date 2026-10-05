"""T20 (Story 2.2 §7/§8): the suite-wide autouse guard refuses real Alpaca HTTP.

``tests/conftest.py::_no_real_alpaca_http`` wraps ``requests.Session.request``. Each refused
call raises ``AssertionError``, is recorded, and sends nothing; teardown fails any test that
recorded one. These tests request the fixture by name to read the record, and clear it before
teardown so the deliberate refusals do not fail them.
"""

from __future__ import annotations

import pytest
import requests
from requests.adapters import HTTPAdapter


class _Sent:
    """Stands in for the transport below ``Session.request``: records what would have been sent."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def send(self, adapter, request, *args, **kwargs):
        self.urls.append(request.url)
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"{}"
        resp.url = request.url
        resp.request = request
        return resp


@pytest.fixture
def transport(monkeypatch) -> _Sent:
    sent = _Sent()
    monkeypatch.setattr(HTTPAdapter, "send", lambda adapter, request, *a, **k: sent.send(
        adapter, request, *a, **k))
    return sent


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: requests.get("https://paper-api.alpaca.markets/v2/clock"),
         "GET https://paper-api.alpaca.markets/v2/clock"),
        (lambda: requests.delete("https://api.alpaca.markets/v2/orders/x"),
         "DELETE https://api.alpaca.markets/v2/orders/x"),
        (lambda: requests.Session().request("GET", "https://data.alpaca.markets/v2/stocks/bars"),
         "GET https://data.alpaca.markets/v2/stocks/bars"),
    ],
)
def test_alpaca_host_is_refused_recorded_and_not_sent(_no_real_alpaca_http, transport, call,
                                                      expected):
    with pytest.raises(AssertionError, match="real Alpaca HTTP in a test"):
        call()
    assert _no_real_alpaca_http == [expected]
    assert transport.urls == []  # nothing reached the transport
    _no_real_alpaca_http.clear()  # the deliberate refusal must not fail this test at teardown


@pytest.mark.parametrize(
    "url", ["https://alpaca.markets/x", "https://PAPER-API.Alpaca.Markets/v2/clock"])
def test_host_predicate_refuses_apex_and_any_case(_no_real_alpaca_http, transport, url):
    with pytest.raises(AssertionError):
        requests.get(url)
    assert len(_no_real_alpaca_http) == 1
    assert transport.urls == []
    _no_real_alpaca_http.clear()


@pytest.mark.parametrize(
    "url", ["https://alpaca.markets.example.test/v2/clock", "https://notalpaca.markets/v2/clock"])
def test_non_alpaca_host_passes_through(_no_real_alpaca_http, transport, url):
    resp = requests.get(url)
    assert resp.status_code == 200
    assert transport.urls == [url]  # the original request path ran, down to the transport
    assert _no_real_alpaca_http == []


def test_guard_is_autouse():
    # Requests NO fixture: the wrapper must be installed for every test, not only those that ask.
    assert getattr(requests.Session.request, "_algua_alpaca_guard", False) is True
