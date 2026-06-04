from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
