import pandas as pd
import pytest
import requests

from algua.config.settings import Settings
from algua.data.contracts import BarRequest
from algua.data.providers import get_provider, register_provider
from algua.data.providers.alpaca import AlpacaBarProvider, _normalize_alpaca
from algua.data.providers.errors import ProviderError
from algua.data.providers.yfinance import YFinanceBarProvider, _normalize_yfinance


def test_yfinance_normalizer_handles_single_symbol_frame():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Adj Close": [100.25],
            "Volume": [1000],
        },
        index=pd.Index([pd.Timestamp("2026-01-02")], name="Date"),
    )

    frame = _normalize_yfinance(raw, ("AAPL",))

    assert frame.to_dict("records") == [
        {
            "ts": pd.Timestamp("2026-01-02"),
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.25,
            "volume": 1000,
        }
    ]


def test_yfinance_provider_rejects_auto_adjustment():
    provider = YFinanceBarProvider()
    request = BarRequest(("AAPL",), "2026-01-02", "2026-01-03", adjustment="auto")

    with pytest.raises(ValueError, match="adj_close"):
        provider.get_bars(request)


def test_alpaca_normalizer_flattens_symbol_keyed_payload():
    frame = _normalize_alpaca(
        {
            "bars": {
                "AAPL": [
                    {
                        "t": "2026-01-02T14:30:00Z",
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1000,
                    }
                ]
            }
        }
    )

    assert frame.to_dict("records") == [
        {
            "ts": "2026-01-02T14:30:00Z",
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    ]


def test_alpaca_provider_merges_raw_close_with_adjusted_close(monkeypatch):
    responses = {
        "raw": {
            "bars": {
                "AAPL": [
                    {
                        "t": "2026-01-02T14:30:00Z",
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.0,
                        "c": 100.5,
                        "v": 1000,
                    }
                ]
            }
        },
        "all": {
            "bars": {
                "AAPL": [
                    {
                        "t": "2026-01-02T14:30:00Z",
                        "o": 99.5,
                        "h": 100.5,
                        "l": 98.5,
                        "c": 100.0,
                        "v": 1000,
                    }
                ]
            }
        },
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(*_args, **kwargs):
        return FakeResponse(responses[kwargs["params"]["adjustment"]])

    monkeypatch.setattr("algua.data.providers.alpaca.requests.get", fake_get)
    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    bars = provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))

    assert bars.frame.to_dict("records") == [
        {
            "ts": "2026-01-02T14:30:00Z",
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "adj_close": 100.0,
        }
    ]
    assert bars.source_metadata["adjustment"] == "raw+all"


def test_alpaca_timeframe_maps_1d_to_alpaca_format():
    from algua.data.providers.alpaca import _alpaca_timeframe

    assert _alpaca_timeframe("1d") == "1Day"
    assert _alpaca_timeframe("1Day") == "1Day"
    assert _alpaca_timeframe("1Min") == "1Min"  # unknown-to-us passes through


def test_get_provider_builds_yfinance():
    provider = get_provider("yfinance", Settings())
    assert isinstance(provider, YFinanceBarProvider)


def test_get_provider_builds_alpaca_from_settings():
    settings = Settings(alpaca_api_key="key", alpaca_api_secret="secret")
    provider = get_provider("alpaca", settings)
    assert isinstance(provider, AlpacaBarProvider)
    assert provider.api_key == "key"
    assert provider.api_secret == "secret"
    assert provider.base_url == settings.alpaca_data_url.rstrip("/")


def test_get_provider_alpaca_requires_credentials():
    # Explicit None overrides any ambient ALGUA_ALPACA_* env so the test is hermetic.
    settings = Settings(alpaca_api_key=None, alpaca_api_secret=None)
    with pytest.raises(ValueError, match="ALGUA_ALPACA_API_KEY"):
        get_provider("alpaca", settings)


def test_get_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="unsupported bar provider: nope"):
        get_provider("nope", Settings())


def test_register_provider_is_open_for_extension():
    """A new provider plugs in via the registry without touching the CLI or get_provider."""
    sentinel = object()
    register_provider("dummy", lambda _settings: sentinel)
    try:
        assert get_provider("dummy", Settings()) is sentinel
    finally:
        from algua.data.providers import _REGISTRY

        del _REGISTRY["dummy"]


# --- #58: yfinance normalizer must not silently drop columns/symbols ---


def test_yfinance_normalizer_raises_naming_missing_columns():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            # no "Adj Close", no "Volume"
        },
        index=pd.Index([pd.Timestamp("2026-01-02")], name="Date"),
    )

    with pytest.raises(ProviderError) as exc:
        _normalize_yfinance(raw, ("AAPL",))

    message = str(exc.value)
    assert "adj_close" in message
    assert "volume" in message


def test_yfinance_normalizer_raises_on_requested_but_missing_symbols():
    # A request for AAPL/MSFT/GOOG that returns only AAPL must fail rather than
    # produce a frame whose coverage is narrower than the manifest will claim.
    raw = pd.DataFrame(
        {
            ("AAPL", "Open"): [100.0],
            ("AAPL", "High"): [101.0],
            ("AAPL", "Low"): [99.0],
            ("AAPL", "Close"): [100.5],
            ("AAPL", "Adj Close"): [100.25],
            ("AAPL", "Volume"): [1000],
        },
        index=pd.Index([pd.Timestamp("2026-01-02")], name="Date"),
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    with pytest.raises(ProviderError) as exc:
        _normalize_yfinance(raw, ("AAPL", "MSFT", "GOOG"))

    message = str(exc.value)
    assert "GOOG" in message
    assert "MSFT" in message


def test_yfinance_provider_raises_on_partial_coverage(monkeypatch):
    raw = pd.DataFrame(
        {
            ("AAPL", "Open"): [100.0],
            ("AAPL", "High"): [101.0],
            ("AAPL", "Low"): [99.0],
            ("AAPL", "Close"): [100.5],
            ("AAPL", "Adj Close"): [100.25],
            ("AAPL", "Volume"): [1000],
        },
        index=pd.Index([pd.Timestamp("2026-01-02")], name="Date"),
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    import sys
    import types

    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **k: raw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    provider = YFinanceBarProvider()
    with pytest.raises(ProviderError, match="MSFT"):
        provider.get_bars(BarRequest(("AAPL", "MSFT"), "2026-01-02", "2026-01-03"))


def test_yfinance_download_failure_wrapped_as_provider_error(monkeypatch):
    import sys
    import types

    def boom(*_a, **_k):
        raise RuntimeError("yfinance: HTTPSConnectionPool read timed out")

    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    provider = YFinanceBarProvider()
    with pytest.raises(ProviderError, match="yfinance download failed"):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))


# --- #59: provider transport errors must stay inside the JSON contract ---


def test_yfinance_import_failure_wrapped_as_provider_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("No module named 'yfinance'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = YFinanceBarProvider()

    with pytest.raises(ProviderError, match="yfinance"):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))


def test_alpaca_http_error_wrapped_as_provider_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise requests.HTTPError("401 Unauthorized")

        def json(self):
            return {}

    monkeypatch.setattr(
        "algua.data.providers.alpaca.requests.get", lambda *a, **k: FakeResponse()
    )
    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    with pytest.raises(ProviderError):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))


def test_alpaca_connection_error_wrapped_as_provider_error(monkeypatch):
    def boom(*_a, **_k):
        raise requests.ConnectionError("name resolution failed")

    monkeypatch.setattr("algua.data.providers.alpaca.requests.get", boom)
    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    with pytest.raises(ProviderError):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))


def test_alpaca_retries_on_429_then_succeeds(monkeypatch):
    payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2026-01-02T14:30:00Z",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1000,
                }
            ]
        }
    }

    class FlakyResponse:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

        def json(self):
            return self._body

    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        # First call (raw, attempt 1) returns 429, then everything succeeds.
        if calls["n"] == 1:
            return FlakyResponse(429, {})
        return FlakyResponse(200, payload)

    monkeypatch.setattr("algua.data.providers.alpaca.requests.get", fake_get)
    monkeypatch.setattr("algua.data.providers.alpaca.time.sleep", lambda _s: None)

    provider = AlpacaBarProvider(api_key="key", api_secret="secret")
    bars = provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))

    assert calls["n"] >= 3  # retried the 429 at least once, plus the second fetch
    assert set(bars.frame["symbol"]) == {"AAPL"}


def test_alpaca_raises_when_raw_and_adjusted_key_sets_differ(monkeypatch):
    # The adjusted view is missing the MSFT bar; the inner merge would silently
    # drop it, producing a partial snapshot that still looks valid.
    responses = {
        "raw": {
            "bars": {
                "AAPL": [
                    {"t": "2026-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 100.5, "v": 1}
                ],
                "MSFT": [
                    {"t": "2026-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 200.5, "v": 1}
                ],
            }
        },
        "all": {
            "bars": {
                "AAPL": [
                    {"t": "2026-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 100.0, "v": 1}
                ],
            }
        },
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(*_a, **kwargs):
        return FakeResponse(responses[kwargs["params"]["adjustment"]])

    monkeypatch.setattr("algua.data.providers.alpaca.requests.get", fake_get)
    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    with pytest.raises(ProviderError, match="MSFT"):
        provider.get_bars(BarRequest(("AAPL", "MSFT"), "2026-01-02", "2026-01-03"))


def test_alpaca_malformed_json_body_wrapped_as_provider_error(monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        "algua.data.providers.alpaca.requests.get", lambda *a, **k: FakeResponse()
    )
    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    with pytest.raises(ProviderError, match="malformed JSON"):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))


def test_alpaca_gives_up_after_retries_on_persistent_5xx(monkeypatch):
    class ServerError:
        status_code = 503

        def raise_for_status(self):
            raise requests.HTTPError("503 Service Unavailable")

        def json(self):
            return {}

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        return ServerError()

    monkeypatch.setattr("algua.data.providers.alpaca.requests.get", fake_get)
    monkeypatch.setattr("algua.data.providers.alpaca.time.sleep", lambda _s: None)

    provider = AlpacaBarProvider(api_key="key", api_secret="secret")

    with pytest.raises(ProviderError):
        provider.get_bars(BarRequest(("AAPL",), "2026-01-02", "2026-01-03"))
    assert attempts["n"] > 1  # actually retried before giving up
