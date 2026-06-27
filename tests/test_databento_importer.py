# tests/test_databento_importer.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algua.data.contracts import DatabentoImportRequest, FirstRateImportRequest
from algua.data.corpactions import Dividend, Split
from algua.data.importers import get_importer
from algua.data.importers.databento import (
    DatabentoImporter,
    parse_databento_corp_actions,
    parse_databento_raw,
)


def _write_raw(path: Path, ts, closes, *, opens=None, highs=None, lows=None, vols=None) -> None:
    n = len(closes)
    opens = opens or [float(c) for c in closes]
    highs = highs or [float(c) + 1 for c in closes]
    lows = lows or [float(c) - 1 for c in closes]
    vols = vols or [100.0] * n
    pd.DataFrame(
        {
            "ts": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": [float(c) for c in closes],
            "volume": vols,
        }
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
    _write_raw(
        p,
        pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        [100, 110],
        highs=[float("inf"), 111],
    )
    with pytest.raises(ValueError, match="finite"):
        parse_databento_raw(p)
    p2 = tmp_path / "MSFT.parquet"
    _write_raw(p2, pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"), [100, 0])  # close 0
    with pytest.raises(ValueError, match="> 0"):
        parse_databento_raw(p2)
    p3 = tmp_path / "TSLA.parquet"
    _write_raw(
        p3,
        pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        [100, 110],
        vols=[100.0, -1.0],
    )
    with pytest.raises(ValueError, match="volume"):
        parse_databento_raw(p3)


def test_parse_raw_missing_column(tmp_path):
    p = tmp_path / "AAPL.parquet"
    pd.DataFrame(
        {"ts": pd.date_range("2024-01-01", periods=1, tz="UTC"), "close": [100.0]}
    ).to_parquet(p)
    with pytest.raises(ValueError, match="missing columns"):
        parse_databento_raw(p)


def _write_ca(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path)


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_ca_parse_split_and_dividend(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(
        p,
        [
            {"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0},
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
            {"symbol": "msft", "ex_date": "2024-01-10", "kind": "Dividend ", "value": 1.0},
        ],
    )
    ev = parse_databento_corp_actions(p)
    # first-seen row order is preserved (no event_id path), so the exact list is deterministic
    assert ev["AAPL"] == [
        Split(ex_date=_utc("2024-01-03"), ratio=2.0),
        Dividend(ex_date=_utc("2024-02-01"), cash=0.5),
    ]
    assert ev["MSFT"] == [Dividend(ex_date=_utc("2024-01-10"), cash=1.0)]


def test_ca_unknown_kind_and_bad_value(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(
        p, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "merger", "value": 1.0}]
    )
    with pytest.raises(ValueError, match="unknown kind"):
        parse_databento_corp_actions(p)
    p2 = tmp_path / "ca2.parquet"
    _write_ca(
        p2, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "dividend", "value": 0.0}]
    )
    with pytest.raises(ValueError, match="value"):
        parse_databento_corp_actions(p2)


def test_ca_non_midnight_ex_date_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(
        p, [{"symbol": "AAPL", "ex_date": "2024-01-03 12:00", "kind": "split", "value": 2.0}]
    )
    with pytest.raises(ValueError, match="midnight|UTC"):
        parse_databento_corp_actions(p)


def _div(sym, ex, val, eid=None):
    """Shorthand for a dividend dict row."""
    row = {"symbol": sym, "ex_date": ex, "kind": "dividend", "value": val}
    if eid is not None:
        row["event_id"] = eid
    return row


def test_ca_same_date_two_dividends_both_kept(tmp_path):
    # regular + special dividend on one ex-date with distinct event_id -> both kept (engine sums).
    p = tmp_path / "ca.parquet"
    _write_ca(p, [_div("AAPL", "2024-02-01", 0.5, "r1"), _div("AAPL", "2024-02-01", 0.5, "s1")])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 2


def test_ca_event_id_dedup_and_conflict(tmp_path):
    # duplicate (symbol, event_id) identical economics -> one; differing economics -> raise.
    p = tmp_path / "ca.parquet"
    _write_ca(p, [_div("AAPL", "2024-02-01", 0.5, "d1"), _div("AAPL", "2024-02-01", 0.5, "d1")])
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 1

    p2 = tmp_path / "ca2.parquet"
    _write_ca(p2, [_div("AAPL", "2024-02-01", 0.5, "d1"), _div("AAPL", "2024-02-01", 0.9, "d1")])
    with pytest.raises(ValueError, match="differing economics"):
        parse_databento_corp_actions(p2)


def test_ca_event_id_blank_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [_div("AAPL", "2024-02-01", 0.5, "d1"), _div("AAPL", "2024-03-01", 0.5, None)])
    with pytest.raises(ValueError, match="event_id"):
        parse_databento_corp_actions(p)


def test_ca_same_event_id_across_symbols_kept(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [_div("AAPL", "2024-02-01", 0.5, "x"), _div("MSFT", "2024-02-01", 0.5, "x")])
    ev = parse_databento_corp_actions(p)
    assert len(ev["AAPL"]) == 1 and len(ev["MSFT"]) == 1


def test_ca_no_event_id_exact_dup_raises(tmp_path):
    # Without event_id, a same-(ex_date, value) duplicate is ambiguous (genuine repeat vs two
    # distinct same-date distributions) and would silently under-adjust adj_close, so fail
    # closed loudly and point the operator at event_id (#264).
    p = tmp_path / "ca.parquet"
    _write_ca(
        p,
        [
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
        ],
    )
    with pytest.raises(ValueError, match="event_id"):
        parse_databento_corp_actions(p)


def test_ca_no_event_id_distinct_rows_kept(tmp_path):
    # The raise is confined to EXACT-tuple duplicates: legitimate no-event_id rows that differ in
    # value or kind on the same ex_date are NOT duplicates and must still pass (#264).
    p = tmp_path / "ca.parquet"
    _write_ca(
        p,
        [
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.5},
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "dividend", "value": 0.7},
            {"symbol": "AAPL", "ex_date": "2024-02-01", "kind": "split", "value": 2.0},
        ],
    )
    assert len(parse_databento_corp_actions(p)["AAPL"]) == 3


def _run(raw_dir: Path, ca: Path, **kw):
    req = DatabentoImportRequest(raw_dir=raw_dir, corp_actions_path=ca, **kw)
    return list(DatabentoImporter().import_bars(req))


def test_importer_split_adj_close(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
        [100, 110, 50, 55],  # 2:1 split on bar index 2
    )
    _write_ca(ca, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}])
    [pb] = _run(raw, ca)
    assert list(pb.frame.columns) == [
        "ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"
    ]
    np.testing.assert_allclose(pb.frame["adj_close"].to_numpy(), [50, 55, 50, 55])
    assert pb.source_metadata["vendor"] == "databento"


def test_importer_same_date_regular_plus_special_dividend(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
        [100, 110, 120, 130],  # P_prev for ex 01-03 = close[1] = 110
    )
    _write_ca(ca, [_div("AAPL", "2024-01-03", 2.0, "r"), _div("AAPL", "2024-01-03", 3.0, "s")])
    [pb] = _run(raw, ca)
    m = (110 - 5) / 110  # summed cash 5, NOT cross-term
    np.testing.assert_allclose(pb.frame["adj_close"].to_numpy()[0], 100 * m)


def test_importer_multi_symbol_and_no_events(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        [100, 110, 120],
    )
    _write_raw(
        raw / "MSFT.parquet",
        pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        [10, 11, 12],
    )
    _write_ca(ca, [{"symbol": "AAPL", "ex_date": "2024-01-02", "kind": "split", "value": 2.0}])
    pbs = {pb.frame["symbol"].iloc[0]: pb for pb in _run(raw, ca)}
    assert set(pbs) == {"AAPL", "MSFT"}
    # no events → identity
    np.testing.assert_allclose(pbs["MSFT"].frame["adj_close"].to_numpy(), [10, 11, 12])
    # each symbol is exactly one chunk
    assert all(pb.frame["symbol"].nunique() == 1 for pb in pbs.values())


def test_importer_rejects_wrong_request_and_intraday(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
        [100],
    )
    _write_ca(ca, [])
    with pytest.raises(ValueError, match="DatabentoImportRequest"):
        list(DatabentoImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=raw)))
    with pytest.raises(ValueError, match="1d only"):
        _run(raw, ca, timeframe="1h")


def test_importer_dup_symbol_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(raw / "AAPL.parquet", pd.date_range("2024-01-01", periods=1, tz="UTC"), [100])
    _write_raw(raw / "aapl.parquet", pd.date_range("2024-01-01", periods=1, tz="UTC"), [100])
    _write_ca(ca, [])
    with pytest.raises(ValueError, match="duplicate symbol"):
        _run(raw, ca)


def test_registry_has_databento():
    assert isinstance(get_importer("databento"), DatabentoImporter)


def test_ca_null_ex_date_raises(tmp_path):
    p = tmp_path / "ca.parquet"
    _write_ca(p, [{"symbol": "AAPL", "ex_date": None, "kind": "split", "value": 2.0}])
    with pytest.raises(ValueError, match="ex_date"):
        parse_databento_corp_actions(p)


def test_importer_empty_raw_file_raises(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    # right columns, zero rows -> fail closed (don't silently drop a symbol from the snapshot)
    pd.DataFrame(
        {c: pd.Series([], dtype="float64") for c in ["open", "high", "low", "close", "volume"]}
        | {"ts": pd.Series([], dtype="datetime64[ns, UTC]")}
    ).to_parquet(raw / "AAPL.parquet")
    _write_ca(ca, [{"symbol": "AAPL", "ex_date": "2024-01-03", "kind": "split", "value": 2.0}])
    with pytest.raises(ValueError, match="no bars"):
        _run(raw, ca)


def test_importer_ca_symbol_without_raw_file_ignored(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    ca = tmp_path / "ca.parquet"
    _write_raw(
        raw / "AAPL.parquet",
        pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        [100, 110, 120],
    )
    # CA is a superset: it carries ZZZZ (no raw file) — that event is ignored; AAPL imports OK
    _write_ca(
        ca,
        [
            {"symbol": "AAPL", "ex_date": "2024-01-02", "kind": "split", "value": 2.0},
            {"symbol": "ZZZZ", "ex_date": "2024-01-02", "kind": "split", "value": 2.0},
        ],
    )
    [pb] = _run(raw, ca)
    assert pb.frame["symbol"].iloc[0] == "AAPL"
