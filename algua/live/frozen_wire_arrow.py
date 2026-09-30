"""``bars.arrow``: the frozen planner's raw bar frame as an uncompressed Arrow IPC file.

Story 1.3c contract §4. Exactly the columns of ``BARS_SCHEMA`` in that order, no nulls, rows in
the frame's order, in at most one record batch that is never compressed — so the file's byte bound
is also a bound on the memory decoding it takes; both are checked before a value is read. Float
arrays are built directly from NumPy values with ``from_pandas=False`` — never through
``Table.from_pandas``, which turns NaN into null and attaches pandas metadata — so NaN and
infinities cross as values. Nothing is pickled and no object or extension type is
accepted: the decoder requires exact schema equality before it reads a single value, then rebuilds
the bar-schema frame (``docs/contracts/bar-schema.md``) from plain arrays. ``request.json`` refers
to the file as ``{file, rows, bars_digest}`` (the Story 1.3a logical digest), which the decoded
frame must reproduce.
"""

from __future__ import annotations

import struct
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from algua.live.frozen_wire_json import WireError, expect, expect_hex, expect_object
from algua.live.planner_binding import bars_digest

BARS_FILE: Final = "bars.arrow"
_REFERENCE_KEYS = frozenset({"file", "rows", "bars_digest"})
FLOAT_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
FRAME_COLUMNS = ("symbol", *FLOAT_COLUMNS)
BARS_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
        pa.field("symbol", pa.string()),
        *(pa.field(name, pa.float64()) for name in FLOAT_COLUMNS),
    ]
)
# Arrow's File.fbs / Message.fbs: the flatbuffer field indices read here, the `Block` struct size
# (offset: long, metaDataLength: int + 4 padding, bodyLength: long) and the file's tail after the
# footer (its int32 length, then the "ARROW1" magic).
_FOOTER_RECORD_BATCHES: Final = 3
_MESSAGE_HEADER: Final = 2
_RECORD_BATCH_COMPRESSION: Final = 3
_BLOCK_BYTES: Final = 24
_FILE_TAIL_BYTES: Final = 10


def encode_bars(frame: pd.DataFrame) -> bytes:
    """Deterministic Arrow IPC file bytes for a bar-schema frame."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise WireError("invalid_bars", "raw_bars must be a DataFrame with a DatetimeIndex")
    if frame.index.name != "timestamp" or str(frame.index.tz) != "UTC":
        raise WireError("invalid_bars", "raw_bars index must be a UTC index named 'timestamp'")
    if tuple(frame.columns) != FRAME_COLUMNS:
        raise WireError("invalid_bars", f"raw_bars columns must be exactly {list(FRAME_COLUMNS)}")
    try:
        arrays = [
            pa.array(frame.index.as_unit("ns").asi8, type=pa.timestamp("ns", tz="UTC")),
            pa.array(frame["symbol"].to_numpy(dtype=object), type=pa.string(), from_pandas=False),
            *(
                pa.array(frame[name].to_numpy(dtype=np.float64), type=pa.float64(),
                         from_pandas=False)
                for name in FLOAT_COLUMNS
            ),
        ]
        table = pa.Table.from_arrays(arrays, schema=BARS_SCHEMA)
    except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
        raise WireError("invalid_bars", str(exc)) from None
    if any(column.null_count for column in table.columns):
        raise WireError("invalid_bars", "raw_bars must not contain nulls")
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, BARS_SCHEMA, options=ipc.IpcWriteOptions(compression=None)) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _scalar(buf: memoryview, pos: int, fmt: str) -> int:
    if pos < 0 or pos + struct.calcsize(fmt) > len(buf):
        raise WireError("invalid_arrow", "IPC metadata points outside its buffer")
    return int(struct.unpack_from(fmt, buf, pos)[0])


def _target(buf: memoryview, pos: int) -> int:
    """Where the flatbuffer offset stored at ``pos`` points."""
    return pos + _scalar(buf, pos, "<I")


def _field(buf: memoryview, table: int, index: int) -> int | None:
    """Position of field ``index`` of the flatbuffer table at ``table``; None when it is absent."""
    vtable = table - _scalar(buf, table, "<i")
    slot = 4 + 2 * index
    if slot + 2 > _scalar(buf, vtable, "<H"):
        return None
    offset = _scalar(buf, vtable + slot, "<H")
    return table + offset if offset else None


def _batch_offsets(data: bytes) -> list[int]:
    """File offsets of the record batches the footer lists: exactly the ones ``read_all`` reads."""
    view = memoryview(data)
    end = len(view) - _FILE_TAIL_BYTES
    length = _scalar(view, end, "<i")
    if not 0 < length <= end:
        raise WireError("invalid_arrow", "IPC footer length is out of range")
    footer = view[end - length:end]
    batches = _field(footer, _target(footer, 0), _FOOTER_RECORD_BATCHES)
    if batches is None:
        return []
    vector = _target(footer, batches)
    count = _scalar(footer, vector, "<I")
    return [_scalar(footer, vector + 4 + _BLOCK_BYTES * i, "<q") for i in range(count)]


def _refuse_compressed_batches(data: bytes, batches: int) -> None:
    """Refuse a compressed record batch from its metadata alone, before decompressing anything.

    pyarrow exposes neither the footer's blocks nor a batch's ``compression`` field, decompressing
    to find out would defeat the byte bound as a memory bound, and a size heuristic cannot work
    (a batch's string data has no bound but the length its compressed buffer declares). So the
    few flatbuffer fields involved are read directly: pyarrow frames each footer-listed message,
    and the batch is refused when its ``RecordBatch.compression`` field is present.
    """
    offsets = _batch_offsets(data)
    if len(offsets) != batches:
        raise WireError("invalid_arrow", "IPC footer blocks disagree with the batch count")
    for offset in offsets:
        source = pa.BufferReader(data)
        try:
            source.seek(offset)
            message = ipc.read_message(source)
        except (pa.ArrowException, EOFError, OSError, ValueError) as exc:
            raise WireError("invalid_arrow", str(exc)) from None
        meta = memoryview(message.metadata)
        header = _field(meta, _target(meta, 0), _MESSAGE_HEADER)
        if message.type != "record batch" or header is None:
            raise WireError("invalid_arrow", "an IPC footer block is not a record batch")
        if _field(meta, _target(meta, header), _RECORD_BATCH_COMPRESSION) is not None:
            raise WireError("arrow_compressed", "bars record batches must be uncompressed")


def decode_bars(data: bytes) -> pd.DataFrame:
    """The bar-schema frame held by ``data``; refuses any schema drift, null, second record
    batch or compressed batch (the last two before reading any value)."""
    try:
        reader = ipc.open_file(pa.BufferReader(data))
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise WireError("invalid_arrow", str(exc)) from None
    if not reader.schema.equals(BARS_SCHEMA, check_metadata=True):
        raise WireError("arrow_schema", f"bars schema must be exactly {BARS_SCHEMA}")
    if reader.num_record_batches > 1:
        raise WireError("arrow_batches", "bars must be one record batch")
    _refuse_compressed_batches(data, reader.num_record_batches)
    try:
        table = reader.read_all()
        table.validate(full=True)
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise WireError("invalid_arrow", str(exc)) from None
    for name in BARS_SCHEMA.names:
        if table.column(name).null_count:
            raise WireError("arrow_null", f"bars column {name!r} contains nulls")
    nanos = np.asarray(table.column("timestamp").cast(pa.int64()).to_numpy(), dtype=np.int64)
    index = pd.DatetimeIndex(nanos.view("datetime64[ns]"), name="timestamp").tz_localize("UTC")
    columns: dict[str, np.ndarray] = {
        "symbol": np.array(table.column("symbol").to_pylist(), dtype=object)
    }
    for name in FLOAT_COLUMNS:
        columns[name] = np.array(table.column(name).to_numpy(), dtype=np.float64)
    return pd.DataFrame(columns, index=index)


def bars_reference(frame: pd.DataFrame) -> dict[str, Any]:
    """The ``early.bars`` object that binds ``request.json`` to this frame's file."""
    try:
        digest = bars_digest(frame)
    except ValueError as exc:
        raise WireError("invalid_bars", str(exc)) from None
    return {"file": BARS_FILE, "rows": len(frame), "bars_digest": digest}


def decode_referenced_bars(data: bytes, reference: Any) -> pd.DataFrame:
    """Decode ``data`` and require it to hold exactly the rows and digest ``reference`` names."""
    ref = expect_object(reference, _REFERENCE_KEYS, "early.bars")
    if type(ref["file"]) is not str or ref["file"] != BARS_FILE:
        raise WireError("bad_bars_file", f"bars file must be {BARS_FILE!r}")
    rows = expect(ref["rows"], int, "early.bars.rows")
    digest = expect_hex(ref["bars_digest"], 64, "early.bars.bars_digest")
    frame = decode_bars(data)
    if len(frame) != rows:
        raise WireError("bars_rows", f"bars hold {len(frame)} rows, request says {rows}")
    try:
        actual = bars_digest(frame)
    except ValueError as exc:
        raise WireError("invalid_bars", str(exc)) from None
    if actual != digest:
        raise WireError("bars_digest", "bars do not reproduce the request's bars_digest")
    return frame
