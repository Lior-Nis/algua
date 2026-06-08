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
