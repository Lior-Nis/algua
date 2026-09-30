"""Story 1.3c contract §4: `bars.arrow`, the frozen planner's bar frame on the wire.

An uncompressed Arrow IPC file with a fixed schema, built from NumPy values (never through
`Table.from_pandas`, which turns NaN into null), decoded back into the bar-schema frame with the
same logical digest and the same row order.
"""

from __future__ import annotations

import math
import struct
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from algua.live.frozen_wire import WireError
from algua.live.frozen_wire_arrow import BARS_SCHEMA, decode_bars, encode_bars
from algua.live.planner_binding import bars_digest
from tests.test_frozen_wire import arrow_bytes, make_bars

COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]


def _special_bars() -> pd.DataFrame:
    frame = make_bars()
    frame["open"] = [math.nan, math.inf, -math.inf, -0.0, 5e-324, 1.0]
    frame["volume"] = [0.0, -0.0, math.nan, 1e308, 2.5, 3.0]
    return frame


def test_schema_is_exactly_the_contract_columns():
    assert BARS_SCHEMA.names == ["timestamp", *COLUMNS]
    assert BARS_SCHEMA.field("timestamp").type == pa.timestamp("ns", tz="UTC")
    assert BARS_SCHEMA.field("symbol").type == pa.string()
    assert all(BARS_SCHEMA.field(name).type == pa.float64() for name in COLUMNS[1:])
    assert BARS_SCHEMA.metadata is None


def test_file_is_uncompressed_arrow_ipc_with_the_schema_and_no_nulls():
    data = encode_bars(_special_bars())
    assert data.startswith(b"ARROW1")
    table = ipc.open_file(pa.BufferReader(data)).read_all()
    assert table.schema.equals(BARS_SCHEMA, check_metadata=True)
    assert all(column.null_count == 0 for column in table.columns)


def test_round_trip_preserves_values_order_and_the_logical_digest():
    frame = _special_bars()
    decoded = decode_bars(encode_bars(frame))
    assert list(decoded.columns) == COLUMNS and decoded.index.name == "timestamp"
    assert str(decoded.index.tz) == "UTC"
    assert decoded["symbol"].dtype == object
    assert all(decoded[name].dtype == np.float64 for name in COLUMNS[1:])
    for name in COLUMNS[1:]:
        assert [v.hex() for v in decoded[name]] == [float(v).hex() for v in frame[name]]
    assert bars_digest(decoded) == bars_digest(frame)


def test_nan_is_a_value_not_a_null():
    decoded = decode_bars(encode_bars(_special_bars()))
    assert math.isnan(decoded["open"].iloc[0]) and math.isnan(decoded["volume"].iloc[2])


def test_row_order_is_the_frames_order():
    frame = make_bars(reverse=True)
    decoded = decode_bars(encode_bars(frame))
    assert list(zip(decoded.index, decoded["symbol"], strict=True)) == list(
        zip(frame.index, frame["symbol"], strict=True)
    )


def test_empty_frame_round_trips():
    empty = make_bars().iloc[0:0]
    decoded = decode_bars(encode_bars(empty))
    assert len(decoded) == 0 and list(decoded.columns) == COLUMNS
    assert bars_digest(decoded) == bars_digest(empty)


def test_non_float64_inputs_are_carried_as_float64_with_the_same_digest():
    frame = make_bars()
    frame["volume"] = frame["volume"].astype("int64")
    decoded = decode_bars(encode_bars(frame))
    assert decoded["volume"].dtype == np.float64
    assert bars_digest(decoded) == bars_digest(frame)


def test_encoding_is_deterministic():
    assert encode_bars(_special_bars()) == encode_bars(_special_bars())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.rename_axis("ts"),
        lambda f: f.tz_convert("America/New_York"),
        lambda f: f.tz_localize(None),
        lambda f: f[["open", "symbol", "high", "low", "close", "adj_close", "volume"]],
        lambda f: f.assign(symbol=[None, "A", "B", "C", "D", "E"]),
        lambda f: f.assign(close=["x"] * 6),
        lambda f: f.reset_index(),
    ],
    ids=["index_name", "non_utc", "naive", "column_order", "null_symbol", "text_close", "no_index"],
)
def test_encode_refuses_frames_outside_the_bar_schema(mutate):
    with pytest.raises(WireError) as caught:
        encode_bars(mutate(make_bars()))
    assert caught.value.reason == "invalid_bars"


def test_encode_refuses_a_non_frame():
    with pytest.raises(WireError) as caught:
        encode_bars([])  # type: ignore[arg-type]
    assert caught.value.reason == "invalid_bars"


def _table() -> pa.Table:
    return ipc.open_file(pa.BufferReader(encode_bars(make_bars()))).read_all()


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (lambda t: t.set_column(7, "volume", t.column("volume").cast(pa.int64())),
         "arrow_schema"),
        (lambda t: t.drop_columns(["volume"]), "arrow_schema"),
        (lambda t: t.append_column("extra", pa.array([1.0] * 6)), "arrow_schema"),
        (lambda t: t.set_column(1, "symbol", t.column("symbol").cast(pa.large_string())),
         "arrow_schema"),
        (lambda t: t.replace_schema_metadata({"k": "v"}), "arrow_schema"),
        (lambda t: t.set_column(0, "timestamp", t.column(0).cast(pa.timestamp("ms", tz="UTC"))),
         "arrow_schema"),
        (lambda t: t.set_column(2, "open", pa.array([None, 1.0, 1.0, 1.0, 1.0, 1.0])),
         "arrow_null"),
        (lambda t: t.set_column(1, "symbol", pa.array(["A", None, "B", "C", "D", "E"])),
         "arrow_null"),
        (lambda t: t.set_column(0, "timestamp", pa.array(
            [datetime(2023, 1, 2, tzinfo=UTC), None, *t.column(0).to_pylist()[2:]],
            type=pa.timestamp("ns", tz="UTC"))), "arrow_null"),
    ],
    ids=["int_volume", "missing_column", "extra_column", "large_string", "metadata",
         "millis", "null_open", "null_symbol", "null_timestamp"],
)
def test_decode_refuses_schema_drift_and_nulls(build, reason):
    with pytest.raises(WireError) as caught:
        decode_bars(arrow_bytes(build(_table())))
    assert caught.value.reason == reason


ZSTD_FRAME_MAGIC = b"\x28\xb5\x2f\xfd"


def _written(compression: str | None = None, max_chunksize: int | None = None) -> bytes:
    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(compression=compression)
    with ipc.new_file(sink, BARS_SCHEMA, options=options) as writer:
        writer.write_table(_table(), max_chunksize=max_chunksize)
    return sink.getvalue().to_pybytes()


def _undecompressable_zstd() -> bytes:
    """A zstd file whose compressed buffers no codec can decompress (their frame magic is gone)."""
    data = _written("zstd")
    assert ZSTD_FRAME_MAGIC in data
    broken = data.replace(ZSTD_FRAME_MAGIC, b"\0\0\0\0")
    with pytest.raises(OSError, match="ZSTD decompression failed"):
        ipc.open_file(pa.BufferReader(broken)).read_all()
    return broken


def test_a_plainly_written_single_batch_file_still_decodes():
    frame = decode_bars(_written())
    assert bars_digest(frame) == bars_digest(make_bars())


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (lambda: _written("zstd"), "arrow_compressed"),
        (lambda: _written("lz4"), "arrow_compressed"),
        # Refused from the batch metadata BEFORE any decompression: were the body decompressed
        # first, this file would fail as `invalid_arrow` instead.
        (_undecompressable_zstd, "arrow_compressed"),
        (lambda: _written(max_chunksize=3), "arrow_batches"),
        (lambda: _written("zstd", max_chunksize=3), "arrow_batches"),
    ],
    ids=["zstd", "lz4", "undecompressable_zstd", "two_batches", "two_zstd_batches"],
)
def test_decode_refuses_compressed_and_multi_batch_files(build, reason):
    # The contract's file is uncompressed and in one batch, so the 256 MiB byte bound is also a
    # memory bound; a compressed body could decompress to far more than its file size.
    with pytest.raises(WireError) as caught:
        decode_bars(build())
    assert caught.value.reason == reason


def test_decode_refuses_a_footer_block_that_points_at_no_message():
    data = _written()
    source = pa.BufferReader(data)
    source.seek(8)  # past the "ARROW1" magic and padding: the schema message, then the batch
    ipc.read_message(source)
    offset = source.tell()
    body = ipc.read_message(source).body.size
    metadata = source.tell() - offset - body
    # The footer's Block struct for the batch: offset, metaDataLength (+4 padding), bodyLength.
    block = struct.pack("<qiiq", offset, metadata, 0, body)
    eos = struct.pack("<qiiq", data.rindex(b"\xff\xff\xff\xff\0\0\0\0"), metadata, 0, body)
    assert data.count(block) == 1

    with pytest.raises(WireError) as caught:
        decode_bars(data.replace(block, eos))
    assert caught.value.reason == "invalid_arrow"


def test_decode_refuses_stream_format_and_garbage():
    table = _table()
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    for data in (sink.getvalue().to_pybytes(), b"", b"ARROW1\x00\x00garbage"):
        with pytest.raises(WireError) as caught:
            decode_bars(data)
        assert caught.value.reason == "invalid_arrow"
