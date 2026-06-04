import pandas as pd
import pytest

from algua.config.settings import Settings
from algua.data.contracts import BarRequest
from algua.data.providers import get_provider, register_provider
from algua.data.providers.alpaca import AlpacaBarProvider, _normalize_alpaca
from algua.data.providers.yfinance import YFinanceBarProvider, _normalize_yfinance
from algua.data.schema import to_bar_schema


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
            # yfinance daily Date index is naive; the normalizer treats it as a UTC session date
            # so the frame already satisfies the tz-aware UTC bar schema.
            "ts": pd.Timestamp("2026-01-02", tz="UTC"),
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.25,
            "volume": 1000,
        }
    ]


def test_yfinance_daily_output_satisfies_bar_schema():
    # End-to-end guard: the naive daily Date index must be normalized to tz-aware UTC by the
    # provider so to_bar_schema (which rejects naive timestamps) accepts the output unchanged.
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
    out = to_bar_schema(frame)

    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2026-01-02", tz="UTC")


def test_yfinance_intraday_output_is_converted_to_utc():
    # Intraday yfinance returns a tz-aware (exchange-local) index; the normalizer must convert the
    # same instant to UTC, not shift it.
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Adj Close": [100.25],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-01-02T09:30:00", tz="America/New_York")], name="Datetime"
        ),
    )

    frame = _normalize_yfinance(raw, ("AAPL",))
    out = to_bar_schema(frame)

    assert out.index[0] == pd.Timestamp("2026-01-02T14:30:00", tz="UTC")


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
    with pytest.raises(ValueError, match="ALGUA_ALPACA_API_KEY"):
        get_provider("alpaca", Settings())


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
