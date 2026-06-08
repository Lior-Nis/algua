import pandas as pd

from algua.data.importers.firstrate import parse_firstrate_file, symbol_from_filename


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
