import pandas as pd
import pytest

from algua.data.contracts import ImportRequest
from algua.data.importers.firstrate import (
    FirstRateImporter,
    parse_firstrate_file,
    symbol_from_filename,
)
from algua.data.schema import to_bar_schema, validate_bars


def _write(path, rows, header=False):
    text = ""
    if header:
        text += "datetime,open,high,low,close,volume\n"
    text += "".join(rows)
    path.write_text(text, encoding="utf-8")


def test_parse_headerless_daily(tmp_path):
    f = tmp_path / "AAPL_full_1day_UNADJUSTED.txt"
    _write(f, ["2024-07-01,10.0,11.0,9.5,10.5,1000\n", "2024-07-02,10.5,12.0,10.0,11.5,2000\n"])
    out = parse_firstrate_file(f)
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dt.tz) == "UTC"
    assert out["ts"].iloc[0] == pd.Timestamp("2024-07-01", tz="UTC")
    assert out["close"].iloc[1] == 11.5
    assert out["open"].dtype == "float64"


def test_parse_with_header(tmp_path):
    f = tmp_path / "MSFT_full_1day_adjsplitdiv.csv"
    _write(f, ["2024-07-01,1.0,1.0,1.0,1.0,5\n"], header=True)
    out = parse_firstrate_file(f)
    assert len(out) == 1
    assert out["volume"].iloc[0] == 5.0


def test_symbol_from_filename():
    assert symbol_from_filename("AAPL_full_1day_UNADJUSTED.txt") == "AAPL"
    assert symbol_from_filename("brk.b_full_1day_adjsplitdiv.csv") == "BRK.B"


def test_parse_missing_columns_raises(tmp_path):
    f = tmp_path / "AAPL_full_1day_UNADJUSTED.txt"
    # only 4 columns (no low/volume) -> required columns missing
    f.write_text("datetime,open,high,close\n2024-07-01,10.0,11.0,10.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        parse_firstrate_file(f)


def _firstrate_dirs(tmp_path):
    raw = tmp_path / "raw"
    adj = tmp_path / "adj"
    raw.mkdir()
    adj.mkdir()
    return raw, adj


def _write_pair(raw, adj, symbol, raw_rows, adj_rows):
    (raw / f"{symbol}_full_1day_UNADJUSTED.txt").write_text("".join(raw_rows), encoding="utf-8")
    (adj / f"{symbol}_full_1day_adjsplitdiv.txt").write_text("".join(adj_rows), encoding="utf-8")


def test_import_merges_raw_and_adjusted(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,100,110,95,105,1000\n", "2024-07-02,105,120,100,115,2000\n"],
        ["2024-07-01,50,55,47,52,1000\n", "2024-07-02,52,60,50,57,2000\n"],
    )
    chunks = list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))
    assert len(chunks) == 1
    frame = chunks[0].frame
    row = frame.set_index("ts").loc[pd.Timestamp("2024-07-01", tz="UTC")]
    assert row["close"] == 105.0
    assert row["adj_close"] == 52.0
    validate_bars(to_bar_schema(frame))


def test_import_yields_symbols_sorted(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["MSFT", "AAPL", "GOOG"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    chunks = list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))
    seen = [c.frame["symbol"].iloc[0] for c in chunks]
    assert seen == ["AAPL", "GOOG", "MSFT"]


def test_symbol_set_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "MSFT_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symbol sets differ"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_alias_collision_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (raw / "aapl_full_1day_OTHER.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate symbol"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_key_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,1,1,1,1,1\n", "2024-07-02,1,1,1,1,1\n"],
        ["2024-07-01,1,1,1,1,1\n"],
    )
    with pytest.raises(ValueError, match="key sets differ"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_nonpositive_price_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,0,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="nonpositive"):
        list(FirstRateImporter().import_bars(ImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_symbols_filter_subset(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["AAPL", "MSFT"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    req = ImportRequest(raw_dir=raw, adjusted_dir=adj, symbols=("AAPL",))
    chunks = list(FirstRateImporter().import_bars(req))
    assert [c.frame["symbol"].iloc[0] for c in chunks] == ["AAPL"]


def test_bad_timeframe_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="intraday import not yet supported"):
        list(FirstRateImporter().import_bars(
            ImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1h")))
