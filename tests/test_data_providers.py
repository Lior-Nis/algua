import pandas as pd
import pytest

from algua.data.contracts import BarRequest
from algua.data.providers.alpaca import AlpacaBarProvider, _normalize_alpaca
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
