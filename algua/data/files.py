from __future__ import annotations

import csv
import functools
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from urllib.parse import unquote

import numpy as np
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


def fsync_file(path: Path) -> None:
    """fsync a regular file's data to stable storage. Linux-only: a read-only fd still
    flushes the inode's dirty data pages. (Threat model is a single local Linux FS;
    macOS/NFS fsync semantics differ and are out of scope.)"""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise OSError(f"fsync_file({path}) failed: {exc}") from exc
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation entry within it becomes durable. O_DIRECTORY
    makes a non-directory path fail loudly (ENOTDIR) instead of silently fsyncing the wrong
    object. Linux-only (see `fsync_file`)."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise OSError(f"fsync_dir({path}) failed: {exc}") from exc
    finally:
        os.close(fd)


def fsync_parents(path: Path, *, stop_at: Path) -> None:
    """fsync every directory from `path.parent` up to and including `stop_at` (the durable
    store root). Covers ancestor directories newly created by `mkdir(parents=True)`: fsyncing
    only the leaf parent leaves a freshly-created intermediate dir's own name un-durable in
    *its* parent. `path` must be at or under `stop_at` (a `ValueError` is raised otherwise so a
    miswired call fails loudly instead of silently fsyncing up to the filesystem root)."""
    stop_at = stop_at.resolve()
    resolved = path.resolve()
    if resolved == stop_at:
        fsync_dir(stop_at)
        return
    if stop_at not in resolved.parents:
        raise ValueError(f"{path} is not under stop_at {stop_at}")
    current = resolved.parent
    while True:
        fsync_dir(current)
        if current == stop_at:
            break
        current = current.parent


def fsync_tree(root: Path) -> None:
    """Bottom-up fsync of every regular file, then every subdirectory, then `root` itself
    (`os.walk(topdown=False)`), so child durability precedes the parent's. For partitioned
    trees whose part-files pyarrow wrote without exposing a handle, we reopen+fsync each."""
    def _raise(exc: OSError) -> None:
        raise exc
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False, onerror=_raise):
        d = Path(dirpath)
        for name in filenames:
            fsync_file(d / name)
        fsync_dir(d)


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
    """Atomically publish `data` at `data_dir/relative_path` via a same-dir temp +
    `os.replace` (#158): a reader never observes a partially written file, and a same-id
    concurrent re-publish is benign (content-addressed => identical bytes; readers see the
    old or new inode, byte-identical). Power-loss durable (#184): the temp's bytes are
    fsynced before the rename, and the target's parent-dir chain (up to `data_dir`) after."""
    target_path = data_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".publish-")
    try:
        with os.fdopen(temp_fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, target_path)
        fsync_parents(target_path, stop_at=data_dir)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def validate_partitioned_bars_dir(
    target: Path, *, expected_row_count: int, expected_symbols: set[str]
) -> None:
    """Cheap metadata-only validation of a PRE-EXISTING bars dataset dir before adopting it
    as a committed snapshot (#158): EVERY regular file under `target` must match the expected
    hive layout `symbol=<SYM>/part-*.parquet` (exactly a `symbol=` head directory and a
    `part-*.parquet` basename) — any stray file (at the root or with a non-conforming name)
    is rejected immediately. For conforming files, each parquet footer must parse, the summed
    metadata row counts must equal `expected_row_count`, and the partition symbol set must equal
    `expected_symbols`. This is a partial-corruption and foreign-file detector for dirs left by
    the legacy direct-write ingest (not a cryptographic revalidation — content-addressing carries
    the rest). Mismatch => raise; the caller fails closed (never auto-deletes)."""
    import re as _re

    _PART_RE = _re.compile(r"^part-.*\.parquet$")
    total_rows = 0
    seen_symbols: set[str] = set()
    for p in target.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(target).parts
        if (
            len(rel_parts) != 2
            or not rel_parts[0].startswith("symbol=")
            or not _PART_RE.match(rel_parts[1])
        ):
            rel = p.relative_to(target)
            raise ValueError(
                f"adoption validation failed for {target}: unexpected file {rel!s}"
                " — only symbol=<SYM>/part-*.parquet files are allowed"
            )
        total_rows += pq.ParquetFile(p).metadata.num_rows  # raises on a torn footer
        seen_symbols.add(unquote(rel_parts[0][len("symbol="):]))
    if total_rows != expected_row_count or seen_symbols != expected_symbols:
        raise ValueError(
            f"adoption validation failed for {target}: rows {total_rows} (expected "
            f"{expected_row_count}), symbols {sorted(seen_symbols)} (expected "
            f"{sorted(expected_symbols)}); refusing to adopt a partial/foreign dir"
        )


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
    threading / file splitting are free to vary (issue #130).

    Idempotent-additive across calls into the same `dest_dir`: `existing_data_behavior=
    "overwrite_or_ignore"` writes only the partitions for the symbols in `canon` and leaves any
    sibling `symbol=<SYM>/` partitions from a prior call untouched (it never deletes the directory's
    other contents). The streamed importer relies on this to write one chunk's partition at a time
    into a shared staging dir; a single-shot `ingest_bars` write is unaffected."""
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
        existing_data_behavior="overwrite_or_ignore",
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


BARS_STREAMED_HASH_ALGO = "bars-symbol-merkle-v1"


def compose_bars_symbol_hash(leaves: list[tuple[str, int, str]]) -> str:
    """Order-independent (sorted-by-symbol), domain-separated composition of per-symbol logical
    leaf hashes into one streamed-bars content hash. Each leaf = (symbol, row_count, hex digest)."""
    digest = hashlib.sha256()
    digest.update(BARS_STREAMED_HASH_ALGO.encode())
    ordered = sorted(leaves, key=lambda leaf: leaf[0])
    digest.update(struct.pack("<Q", len(ordered)))
    for symbol, row_count, leaf_hex in ordered:
        sb = symbol.encode("utf-8")
        digest.update(struct.pack("<Q", len(sb)))
        digest.update(sb)
        digest.update(struct.pack("<Q", row_count))
        digest.update(bytes.fromhex(leaf_hex))
    return digest.hexdigest()


def logical_bars_hash(canon: pd.DataFrame) -> str:
    """Content hash over the *logical* bar rows — independent of physical parquet layout/version.

    `canon` carries a tz-aware UTC `ts` column, a `symbol` column, and the six float columns. Rows
    are sorted by (ts, symbol); each column is serialized as fixed-width little-endian bytes (ts as
    int64 nanoseconds-since-epoch UTC, floats as IEEE-754 float64). Symbols are encoded as a
    parallel little-endian uint64 length array followed by concatenated UTF-8 bytes
    (length-prefixed, collision-safe). Signed zeros in float columns are canonicalized to +0.0
    before hashing so that
    -0.0 and 0.0 compare as identical logical data.
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
    symbol_bytes = [s.encode("utf-8") for s in ordered["symbol"].astype(str)]
    lengths = np.array([len(b) for b in symbol_bytes], dtype="<u8")
    digest.update(lengths.tobytes())
    digest.update(b"".join(symbol_bytes))
    for col in BARS_FILE_HASH_COLUMNS:
        values = ordered[col].to_numpy(dtype="<f8") + 0.0  # +0.0 maps -0.0 -> +0.0
        digest.update(values.astype("<f8").tobytes())
    return digest.hexdigest()


def parquet_file_row_count(path: Path) -> int:
    """Full read-back of a single-file parquet: materialize the entire table (all columns,
    all row groups) so every data page is decompressed — power-loss truncation in the
    interior raises here — and return the row count. NOT a footer-only `metadata.num_rows`
    peek: that would report a count without touching the data pages (#184).

    Any pyarrow read failure (torn footer, unreadable page, etc.) is normalized to `ValueError`
    so the verify caller can treat every read-back failure uniformly as damage (a real
    programming bug still surfaces as its own exception type)."""
    try:
        return pq.read_table(path).num_rows
    except pa.ArrowException as exc:
        raise ValueError(f"unreadable parquet file {path}: {exc}") from exc


def parquet_dataset_row_count(dest_dir: Path) -> int:
    """Full read-back of a hive-partitioned bars dataset: read every partition's every column
    (`to_table(columns=None)`), forcing decompression of all data pages, and return the total
    row count. Must NOT use `count_rows()`/footer metadata/pruned-column reads — those skip the
    data pages this check exists to validate (#184).

    Any pyarrow read failure is normalized to `ValueError` (see `parquet_file_row_count`)."""
    try:
        return _parquet_dataset_row_count(dest_dir)
    except pa.ArrowException as exc:
        raise ValueError(f"unreadable parquet dataset {dest_dir}: {exc}") from exc


def _parquet_dataset_row_count(dest_dir: Path) -> int:
    dataset = pads.dataset(dest_dir, format="parquet", partitioning=_BARS_PARTITIONING)
    return dataset.to_table(columns=None).num_rows
