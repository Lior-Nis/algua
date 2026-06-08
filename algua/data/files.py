from __future__ import annotations

import csv
import functools
import hashlib
import shutil
import struct
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
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


BARS_FILE_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adj_close", pa.float64()),
        ("volume", pa.float64()),
    ]
)
_BARS_PARTITIONING = pads.partitioning(pa.schema([("symbol", pa.string())]), flavor="hive")


def write_partitioned_bars(canon: pd.DataFrame, dest_dir: Path) -> int:
    """Write `canon` (a tz-aware-UTC `ts` column + `symbol` + OHLCV, pre-sorted by symbol then ts)
    as a hive-partitioned-by-symbol parquet dataset under `dest_dir`. Returns the parquet file
    count. The snapshot identity is the caller's `logical_bars_hash`, NOT these bytes, so write
    threading / file splitting are free to vary (issue #130)."""
    table = pa.Table.from_pandas(
        canon[["ts", "symbol", *BARS_FILE_HASH_COLUMNS]],
        schema=BARS_FILE_SCHEMA,
        preserve_index=False,
    )
    pads.write_dataset(
        table,
        dest_dir,
        format="parquet",
        partitioning=_BARS_PARTITIONING,
        basename_template="part-{i}.parquet",
    )
    return sum(1 for _ in dest_dir.rglob("*.parquet"))


def read_partitioned_bars(
    dest_dir: Path,
    *,
    symbols: list[str] | None = None,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    """Read a hive-partitioned bars dataset with predicate pushdown. `symbols` prunes partitions at
    the directory level (no other symbol's footer is read); `start`/`end` push a half-open
    `[start, end)` filter on `ts` down to the scanner. Any of the three may be None (unbounded).
    Returns a raw frame (`ts` column + `symbol` + OHLCV); the caller reshapes to bar-schema. Only
    matching fragments are scanned (issue #130)."""
    dataset = pads.dataset(dest_dir, format="parquet", partitioning=_BARS_PARTITIONING)
    conds = []
    if symbols is not None:
        conds.append(pads.field("symbol").isin(list(symbols)))
    if start is not None:
        conds.append(pads.field("ts") >= _ts_scalar(start))
    if end is not None:
        conds.append(pads.field("ts") < _ts_scalar(end))
    filt = functools.reduce(lambda a, b: a & b, conds) if conds else None
    table = dataset.to_table(columns=["ts", "symbol", *BARS_FILE_HASH_COLUMNS], filter=filt)
    return table.to_pandas()


def _ts_scalar(value: object) -> pa.Scalar:
    """Build a `timestamp[ns, tz=UTC]` pyarrow scalar from a datetime/Timestamp, normalizing naive
    inputs to UTC. Constructing the literal in the column's exact type avoids tz/precision-mismatch
    boundary bugs in the pushed-down filter (GATE-1 MEDIUM #4)."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return pa.scalar(ts.value, type=pa.timestamp("ns", tz="UTC"))


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
