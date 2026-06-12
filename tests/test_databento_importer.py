# tests/test_databento_importer.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algua.data.corpactions import Dividend, Split
from algua.data.importers.databento import parse_databento_corp_actions, parse_databento_raw


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


def _write_ca(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path)


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_ca_parse_split_and_dividend(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
        {"symbol": "msft", "ex_date": "2024-01-10", "kind": "Dividend ", "value": 1.0},  # case/space
    ])
    ev = parse_databento_corp_actions(p)
    assert ev["AAPL"] == [Split(ex_date=_utc("2024-01-03"), ratio=2.0),
                          Dividend(ex_date=_utc("2024-02-01"), cash=0.5)] or \
           sorted([type(e).__name__ for e in ev["AAPL"]]) == ["Dividend", "Split"]
    assert ev["MSFT"] == [Dividend(ex_date=_utc("2024-01-10"), cash=1.0)]


def test_ca_unknown_kind_and_bad_value(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "merger", "value": 1.0}])
    with pytest.raises(ValueError, match="unknown kind"):
        parse_databento_corp_actions(p)
    p2 = tmp_path / "ca2.parquet"
    _write_ca(p2, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "dividend", "value": 0.0}])
    with pytest.raises(ValueError, match="value"):
        parse_databento_corp_actions(p2)


def test_ca_non_midnight_ex_date_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [{"symbol": "AAPL", "ex_date": "2024-01-03 12:00", "kind": "split", "value": 2.0}])
    with pytest.raises(ValueError, match="midnight|UTC"):
        parse_databento_corp_actions(p)


def test_ca_same_date_two_dividends_both_kept(tmp_path):
    # regular + special dividend on one ex-date with distinct event_id -> both kept (engine sums).
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "r1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "s1"},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 2


def test_ca_event_id_dedup_and_conflict(tmp_path):
    # duplicate (symbol, event_id) identical economics -> one; differing economics -> raise.
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 1

    p2 = tmp_path / "ca2.parquet"
    _write_ca(p2, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.9, "event_id": "d1"},
    ])
    with pytest.raises(ValueError, match="differing economics"):
        parse_databento_corp_actions(p2)


def test_ca_event_id_blank_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "d1"},
        {"symbol": "AAPL", "ex_date": "2024-03-01", "kind": "dividend", "value": 0.5, "event_id": None},
    ])
    with pytest.raises(ValueError, match="event_id"):
        parse_databento_corp_actions(p)


def test_ca_same_event_id_across_symbols_kept(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "x"},
        {"symbol": "MSFT", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5, "event_id": "x"},
    ])
    ev = parse_databento_corp_actions(p)
    assert len(ev["AAPL"]) == 1 and len(ev["MSFT"]) == 1


def test_ca_no_event_id_exact_dup_dropped(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
        {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
    ])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 1
