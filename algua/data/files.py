from __future__ import annotations

import csv
import hashlib
import struct
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from algua.data.schema import BARS_FILE_HASH_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_tabular_rows(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            data_rows = sum(1 for _ in reader) - 1  # subtract header
        return max(data_rows, 0)
    if suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    return None


def copy_snapshot(source_path: Path, data_dir: Path, relative_path: Path) -> None:
    target_path = data_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def frame_to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize `frame` to canonical, reproducible parquet bytes.

    Determinism is what makes the content hash a stable provenance key (issue #55): the same logical
    dataset must produce byte-identical output across runs. We pin the arrow table directly (no
    pandas index sidecar), disable the file-level pandas metadata blob (which can vary), and fix the
    parquet writer/compression so two equal frames hash identically.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression="snappy", version="2.6")
    return buffer.getvalue().to_pybytes()


def write_bytes_snapshot(data: bytes, data_dir: Path, relative_path: Path) -> None:
    target_path = data_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)


def logical_bars_hash(canon: pd.DataFrame) -> str:
    """Content hash over the *logical* bar rows — independent of physical parquet layout/version.

    `canon` carries a tz-aware UTC `ts` column, a `symbol` column, and the six float columns. Rows
    are sorted by (ts, symbol); each column is serialized as fixed-width little-endian bytes (ts as
    int64 nanoseconds-since-epoch UTC, floats as IEEE-754 float64) with a NUL-joined symbol blob.
    Identical logical bars => identical digest regardless of write threading, file splitting, or
    pyarrow version. This is the snapshot identity for the partitioned bars layout (issue #130,
    GATE-1 HIGH #1/#2), replacing the single-file physical-bytes hash.
    """
    ordered = canon.sort_values(["ts", "symbol"], kind="stable")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(ordered)))
    ts_utc = ordered["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    ts_ns = ts_utc.to_numpy(dtype="datetime64[ns]").view("int64").astype("<i8")
    digest.update(ts_ns.tobytes())
    digest.update("\x00".join(ordered["symbol"].astype(str)).encode("utf-8"))
    digest.update(b"\x00")
    for col in BARS_FILE_HASH_COLUMNS:
        digest.update(ordered[col].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()
