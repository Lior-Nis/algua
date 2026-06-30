import pandas as pd
import pytest

from algua.data.contracts import FirstRateImportRequest
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


def test_parse_intraday_summer_edt_to_utc(tmp_path):
    # 09:30 ET in July is EDT (UTC-4) -> 13:30 UTC.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00,10.0,11.0,9.5,10.5,1000\n"])
    out = parse_firstrate_file(f, timeframe="1m")
    assert out["ts"].iloc[0] == pd.Timestamp("2024-07-01 13:30:00", tz="UTC")
    assert str(out["ts"].dt.tz) == "UTC"


def test_parse_intraday_winter_est_to_utc(tmp_path):
    # 09:30 ET in January is EST (UTC-5) -> 14:30 UTC.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-01-02 09:30:00,10.0,11.0,9.5,10.5,1000\n"])
    out = parse_firstrate_file(f, timeframe="1m")
    assert out["ts"].iloc[0] == pd.Timestamp("2024-01-02 14:30:00", tz="UTC")


def test_parse_intraday_dst_adjacent_valid(tmp_path):
    # 03:00 ET on spring-forward day (just after the gap) -> 07:00 UTC (EDT).
    # 00:30 ET on fall-back day (occurs once, unambiguous) -> 04:30 UTC (still EDT).
    f1 = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f1, ["2024-03-10 03:00:00,1,1,1,1,1\n"])
    assert parse_firstrate_file(f1, timeframe="1m")["ts"].iloc[0] == pd.Timestamp(
        "2024-03-10 07:00:00", tz="UTC"
    )
    f2 = tmp_path / "MSFT_full_1min_UNADJUSTED.txt"
    _write(f2, ["2024-11-03 00:30:00,1,1,1,1,1\n"])
    assert parse_firstrate_file(f2, timeframe="1m")["ts"].iloc[0] == pd.Timestamp(
        "2024-11-03 04:30:00", tz="UTC"
    )


def test_parse_intraday_nonexistent_dst_time_raises(tmp_path):
    # 02:30 ET on 2024-03-10 does not exist (spring-forward gap).
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-03-10 02:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="DST-ambiguous or nonexistent"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_ambiguous_dst_time_raises(tmp_path):
    # 01:30 ET on 2024-11-03 occurs twice (fall-back) -> ambiguous.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-11-03 01:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="DST-ambiguous or nonexistent"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_tz_aware_input_raises(tmp_path):
    # A wall-clock with an explicit offset is tz-aware -> rejected (wall-clock tz unknowable).
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00-04:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="must be naive"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_mixed_tz_offsets_raises_valueerror_not_attributeerror(tmp_path):
    # Rows with differing UTC offsets make pd.to_datetime return an object-dtype Series (no .dt).
    # Must surface as a clean ValueError (json_errors-catchable), never an AttributeError.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00-04:00,1,1,1,1,1\n", "2024-11-03 09:30:00-05:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="mixed timezone offsets"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_intraday_local_midnight_rejected(tmp_path):
    # A date-only / local-midnight value under an intraday timeframe = daily file misfed.
    f = tmp_path / "AAPL_full_1min_UNADJUSTED.txt"
    _write(f, ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="local-midnight"):
        parse_firstrate_file(f, timeframe="1m")


def test_parse_daily_nonmidnight_rejected(tmp_path):
    # An intraday-shaped value under timeframe=1d -> non-midnight 1d bar = contract violation.
    f = tmp_path / "AAPL_full_1day_UNADJUSTED.txt"
    _write(f, ["2024-07-01 09:30:00,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="non-midnight"):
        parse_firstrate_file(f, timeframe="1d")


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
        # Adjusted series ANCHORED at the last bar (adj close == raw close 115) — a back-adjusted
        # full series; the older bar carries the adjustment (52 vs raw 105). Only `close` is used as
        # adj_close, so the other adjusted columns are irrelevant. (#265)
        ["2024-07-01,50,55,47,52,1000\n", "2024-07-02,52,120,50,115,2000\n"],
    )
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    chunks = list(FirstRateImporter().import_bars(req))
    assert len(chunks) == 1
    frame = chunks[0].frame
    rows = frame.set_index("ts")
    assert rows.loc[pd.Timestamp("2024-07-01", tz="UTC"), "close"] == 105.0
    assert rows.loc[pd.Timestamp("2024-07-01", tz="UTC"), "adj_close"] == 52.0
    assert rows.loc[pd.Timestamp("2024-07-02", tz="UTC"), "adj_close"] == 115.0  # anchored
    validate_bars(to_bar_schema(frame, timeframe="1d"))


def test_import_rejects_mis_scaled_adjusted_series(tmp_path):
    # #265: a globally mis-scaled vendor adjusted series (e.g. cents vs dollars — adj ~= 100x raw,
    # not anchored at the last bar) is now rejected at import via check_adj_close_anchored, instead
    # of silently corrupting returns. Raw last close=115; adjusted last=5750 (cents) -> reject.
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,100,110,95,105,1000\n", "2024-07-02,105,120,100,115,2000\n"],
        ["2024-07-01,5200,5500,4700,5200,1000\n", "2024-07-02,5200,12000,5000,5750,2000\n"],
    )
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    with pytest.raises(ValueError, match="not anchored at the last bar"):
        list(FirstRateImporter().import_bars(req))


def test_import_yields_symbols_sorted(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["MSFT", "AAPL", "GOOG"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    chunks = list(FirstRateImporter().import_bars(req))
    seen = [c.frame["symbol"].iloc[0] for c in chunks]
    assert seen == ["AAPL", "GOOG", "MSFT"]


def test_symbol_set_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "MSFT_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="symbol sets differ"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_alias_collision_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (raw / "aapl_full_1day_OTHER.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate symbol"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_key_disagreement_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,1,1,1,1,1\n", "2024-07-02,1,1,1,1,1\n"],
        ["2024-07-01,1,1,1,1,1\n"],
    )
    with pytest.raises(ValueError, match="key sets differ"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_nonpositive_price_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,0,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="nonpositive"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_symbols_filter_subset(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    for sym in ["AAPL", "MSFT"]:
        _write_pair(raw, adj, sym, ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, symbols=("AAPL",))
    chunks = list(FirstRateImporter().import_bars(req))
    assert [c.frame["symbol"].iloc[0] for c in chunks] == ["AAPL"]


def test_unknown_timeframe_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(raw, adj, "AAPL", ["2024-07-01,1,1,1,1,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="unknown timeframe"):
        list(FirstRateImporter().import_bars(
            FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="5min")))


def _write_intraday_pair(raw, adj, symbol, raw_rows, adj_rows):
    (raw / f"{symbol}_full_1min_UNADJUSTED.txt").write_text("".join(raw_rows), encoding="utf-8")
    (adj / f"{symbol}_full_1min_adjsplitdiv.txt").write_text("".join(adj_rows), encoding="utf-8")


def test_intraday_import_merges_and_localizes(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_intraday_pair(
        raw, adj, "AAPL",
        ["2024-07-01 09:30:00,100,110,95,105,1000\n", "2024-07-01 09:31:00,105,120,100,115,2000\n"],
        # anchored at the last bar (adj close == raw close 115); older bar carries adjustment (#265)
        ["2024-07-01 09:30:00,50,55,47,52,1000\n", "2024-07-01 09:31:00,52,120,50,115,2000\n"],
    )
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1m")
    chunks = list(FirstRateImporter().import_bars(req))
    assert len(chunks) == 1
    frame = chunks[0].frame
    row = frame.set_index("ts").loc[pd.Timestamp("2024-07-01 13:30:00", tz="UTC")]
    assert row["close"] == 105.0
    assert row["adj_close"] == 52.0
    validate_bars(to_bar_schema(frame, timeframe="1m"))


def test_intraday_import_30m(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    (raw / "AAPL_full_30min_UNADJUSTED.txt").write_text(
        "2024-07-01 09:30:00,1,1,1,1,1\n2024-07-01 10:00:00,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_30min_adjsplitdiv.txt").write_text(
        "2024-07-01 09:30:00,1,1,1,1,1\n2024-07-01 10:00:00,1,1,1,1,1\n", encoding="utf-8")
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="30m")
    chunks = list(FirstRateImporter().import_bars(req))
    frame = chunks[0].frame
    assert frame["ts"].tolist() == [
        pd.Timestamp("2024-07-01 13:30:00", tz="UTC"),
        pd.Timestamp("2024-07-01 14:00:00", tz="UTC"),
    ]
    validate_bars(to_bar_schema(frame, timeframe="30m"))


def test_intraday_duplicate_message_shows_full_timestamp(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_intraday_pair(
        raw, adj, "AAPL",
        ["2024-07-01 09:30:00,1,1,1,1,1\n", "2024-07-01 09:30:00,2,2,2,2,2\n"],  # dup intraday ts
        ["2024-07-01 09:30:00,1,1,1,1,1\n"],
    )
    # 09:30 ET -> 13:30 UTC; the message renders the full UTC instant (time-of-day intact),
    # not a date-collapsed value.
    with pytest.raises(ValueError, match="duplicate timestamps in raw file.*13:30:00"):
        list(FirstRateImporter().import_bars(
            FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj, timeframe="1m")))


def test_duplicate_timestamp_in_raw_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    _write_pair(
        raw, adj, "AAPL",
        ["2024-07-01,1,1,1,1,1\n", "2024-07-01,2,2,2,2,2\n"],  # dup date in raw
        ["2024-07-01,1,1,1,1,1\n"],
    )
    with pytest.raises(ValueError, match="duplicate timestamps in raw file"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_nan_price_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    # blank close field in raw -> NaN
    _write_pair(raw, adj, "AAPL", ["2024-07-01,1,1,1,,1\n"], ["2024-07-01,1,1,1,1,1\n"])
    with pytest.raises(ValueError, match="NaN or nonpositive"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_raw_dir_with_adjusted_files_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    # dirs swapped: raw_dir holds an adjusted-named file, adj_dir holds the unadjusted one.
    (raw / "AAPL_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw dir holds files that don't look unadjusted"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_adjusted_dir_with_unadjusted_file_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    # operator pointed both dirs at the raw dir: adj_dir holds an unadjusted-named file.
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adjusted dir holds unadjusted-looking files"):
        list(FirstRateImporter().import_bars(FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))


def test_unadjusted_matched_as_token_not_substring(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    # An adjusted file whose name contains "unadjusted" only mid-word (no delimiter boundary) is
    # NOT a role marker, so it must not trip the adjusted-dir guard.
    (raw / "AAPL_full_1day_UNADJUSTED.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_preunadjustedx_adjsplitdiv.txt").write_text(
        "2024-07-01,1,1,1,1,1\n", encoding="utf-8"
    )
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    chunks = list(FirstRateImporter().import_bars(req))
    assert len(chunks) == 1


def test_case_insensitive_unadjusted_marker_ok(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    # raw uses lower-case `unadjusted`; adjusted uses `adjsplitdiv` -> import succeeds.
    (raw / "AAPL_full_1day_unadjusted.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    (adj / "AAPL_full_1day_adjsplitdiv.txt").write_text("2024-07-01,1,1,1,1,1\n", encoding="utf-8")
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    chunks = list(FirstRateImporter().import_bars(req))
    assert len(chunks) == 1
