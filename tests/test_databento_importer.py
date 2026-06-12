# tests/test_databento_importer.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algua.data.importers.databento import parse_databento_raw


def _write_raw(path: Path, ts, closes, *, opens=None, highs=None, lows=None, vols=None) -> None:
    n = len(closes)
    opens = opens or [float(c) for c in closes]
    highs = highs or [float(c) + 1 for c in closes]
    lows = lows or [float(c) - 1 for c in closes]
    vols = vols or [100.0] * n
    pd.DataFrame(
        {"ts": ts, "open": opens, "high": highs, "low": lows, "close": [float(c) for c in closes],
         "volume": vols}
    ).to_parquet(path)


def test_parse_raw_utc_and_naive(tmp_path):
    # tz-aware UTC midnight passes; tz-naive midnight is localized to UTC.
    p1 = tmp_path / "AAPL.parquet"
    _write_raw(p1, pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"), [100, 110, 120])
    out = parse_databento_raw(p1)
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dt.tz) == "UTC"

    p2 = tmp_path / "MSFT.parquet"
    _write_raw(p2, pd.date_range("2024-01-01", periods=3, freq="D"), [100, 110, 120])  # naive
    assert str(parse_databento_raw(p2)["ts"].dt.tz) == "UTC"


def test_parse_raw_rejects_non_utc_tz(tmp_path):
    p = tmp_path / "AAPL.parquet"
    _write_raw(p, pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern"), [100, 110])
    with pytest.raises(ValueError, match="non-UTC"):
        parse_databento_raw(p)


def test_parse_raw_rejects_non_midnight(tmp_path):
    p = tmp_path / "AAPL.parquet"
    ts = pd.to_datetime(["2024-01-01 16:00", "2024-01-02 16:00"])  # naive, non-midnight
    _write_raw(p, ts, [100, 110])
    with pytest.raises(ValueError, match="midnight"):
        parse_databento_raw(p)


def test_parse_raw_rejects_bad_ohlcv(tmp_path):
    p = tmp_path / "AAPL.parquet"
    _write_raw(p, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 110],
               highs=[float("inf"), 111])
    with pytest.raises(ValueError, match="finite"):
        parse_databento_raw(p)
    p2 = tmp_path / "MSFT.parquet"
    _write_raw(p2, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 0])  # close 0
    with pytest.raises(ValueError, match="> 0"):
        parse_databento_raw(p2)
    p3 = tmp_path / "TSLA.parquet"
    _write_raw(p3, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 110],
               vols=[100.0, -1.0])
    with pytest.raises(ValueError, match="volume"):
        parse_databento_raw(p3)


def test_parse_raw_missing_column(tmp_path):
    p = tmp_path / "AAPL.parquet"
    pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=1, tz="UTC"), "close": [100.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing columns"):
        parse_databento_raw(p)
