from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from algua.data.files import frame_to_parquet_bytes, write_bytes_atomic


def _frame():
    return pd.DataFrame(
        {"date": ["2023-01-01T00:00:00", "2023-01-02T00:00:00"], "ret": [0.01, -0.02]}
    )


def test_metadata_attached_and_sorted_key_order_independent():
    b1 = frame_to_parquet_bytes(_frame(), {"b": "2", "a": "1"})
    b2 = frame_to_parquet_bytes(_frame(), {"a": "1", "b": "2"})
    assert b1 == b2  # sorted keys -> insertion order irrelevant -> deterministic
    meta = pq.read_schema(pa.BufferReader(b1)).metadata
    assert meta == {b"a": b"1", b"b": b"2"}


def test_no_metadata_strips_schema_metadata():
    b = frame_to_parquet_bytes(_frame())
    assert pq.read_schema(pa.BufferReader(b)).metadata is None


def test_write_bytes_atomic_roundtrip_no_temp_left(tmp_path: Path):
    dest = tmp_path / "sub" / "series.parquet"
    write_bytes_atomic(b"hello-bytes", dest)
    assert dest.read_bytes() == b"hello-bytes"
    leftovers = [p.name for p in (tmp_path / "sub").iterdir() if p.name.startswith(".emit-")]
    assert leftovers == []
