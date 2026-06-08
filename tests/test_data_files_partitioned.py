import hashlib

import pandas as pd

from algua.data.files import logical_bars_hash


def _canon(rows):
    # rows: list of (ts_iso, symbol, o, h, l, c, adj, vol)
    df = pd.DataFrame(
        rows, columns=["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_logical_hash_is_order_invariant_and_deterministic():
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 2.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 3.0),
    ]
    h1 = logical_bars_hash(_canon(rows))
    h2 = logical_bars_hash(_canon(list(reversed(rows))))  # shuffled input
    assert h1 == h2  # same logical rows => same hash, regardless of row order
    assert len(h1) == len(hashlib.sha256().hexdigest())


def test_logical_hash_changes_on_value_change():
    base = _canon([("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0)])
    changed = _canon([("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 2.0)])
    assert logical_bars_hash(base) != logical_bars_hash(changed)


from datetime import datetime, UTC

from algua.data.files import read_partitioned_bars, write_partitioned_bars


def _canon_sorted(rows):
    return _canon(rows).sort_values(["symbol", "ts"])


def test_write_then_read_full_round_trips(tmp_path):
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 2.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 3.0),
    ]
    dest = tmp_path / "snap"
    file_count = write_partitioned_bars(_canon_sorted(rows), dest)
    assert file_count == 2  # one file per symbol (AAA, BBB)
    assert (dest / "symbol=AAA").is_dir() and (dest / "symbol=BBB").is_dir()

    out = read_partitioned_bars(dest)
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert len(out) == 3
    assert list(out.columns) == ["ts", "symbol", "open", "high", "low", "close",
                                 "adj_close", "volume"]
    assert all(isinstance(s, str) for s in out["symbol"])  # not categorical/dict


def test_read_pushes_down_symbol_and_half_open_window(tmp_path):
    rows = [
        ("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
        ("2024-07-02T00:00:00+00:00", "AAA", 11.0, 11.0, 11.0, 11.0, 11.0, 2.0),
        ("2024-07-03T00:00:00+00:00", "AAA", 12.0, 12.0, 12.0, 12.0, 12.0, 3.0),
        ("2024-07-01T00:00:00+00:00", "BBB", 20.0, 20.0, 20.0, 20.0, 20.0, 4.0),
    ]
    dest = tmp_path / "snap"
    write_partitioned_bars(_canon_sorted(rows), dest)

    out = read_partitioned_bars(
        dest, symbols=["AAA"],
        start=datetime(2024, 7, 1, tzinfo=UTC), end=datetime(2024, 7, 3, tzinfo=UTC),
    )
    assert set(out["symbol"]) == {"AAA"}                 # symbol pruning
    assert out["ts"].min() == pd.Timestamp("2024-07-01", tz="UTC")   # start inclusive
    assert out["ts"].max() == pd.Timestamp("2024-07-02", tz="UTC")   # end exclusive (07-03 dropped)


def test_read_empty_window_returns_empty_frame(tmp_path):
    rows = [("2024-07-01T00:00:00+00:00", "AAA", 10.0, 10.0, 10.0, 10.0, 10.0, 1.0)]
    dest = tmp_path / "snap"
    write_partitioned_bars(_canon_sorted(rows), dest)
    out = read_partitioned_bars(
        dest, symbols=["AAA"],
        start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert out.empty
