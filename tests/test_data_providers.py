import pandas as pd

from algua.data.providers.alpaca import _normalize_alpaca
from algua.data.providers.yfinance import _normalize_yfinance


def test_yfinance_normalizer_handles_single_symbol_frame():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
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
            "volume": 1000,
        }
    ]


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
