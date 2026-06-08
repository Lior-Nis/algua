import pandas as pd
import pytest

from algua.data.contracts import ImportRequest
from algua.data.importers import get_importer, register_importer
from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.data.store import DataStore


def test_get_importer_unknown_raises():
    with pytest.raises(ValueError, match="unsupported bar importer: nope"):
        get_importer("nope")


def test_register_and_get_importer_roundtrip():
    class _Dummy:
        name = "dummy"

        def import_bars(self, request):
            return iter(())

    register_importer("dummy", lambda: _Dummy())
    try:
        assert get_importer("dummy").name == "dummy"
    finally:
        from algua.data.importers import _REGISTRY

        del _REGISTRY["dummy"]


def test_import_request_defaults(tmp_path):
    req = ImportRequest(raw_dir=tmp_path / "raw", adjusted_dir=tmp_path / "adj")
    assert req.timeframe == "1d"
    assert req.adjustment == "split_div"
    assert req.symbols is None
    assert req.as_of is None


def _chunk(symbol, dates_prices):
    n = len(dates_prices)
    return pd.DataFrame({
        "ts": [pd.Timestamp(d, tz="UTC") for d, _ in dates_prices],
        "symbol": [symbol] * n,
        "open": [p for _, p in dates_prices], "high": [p for _, p in dates_prices],
        "low": [p for _, p in dates_prices], "close": [p for _, p in dates_prices],
        "adj_close": [p / 2 for _, p in dates_prices], "volume": [100.0] * n,
    })


def _two_symbol_chunks():
    return [
        _chunk("AAPL", [("2024-07-01", 100.0), ("2024-07-02", 101.0)]),
        _chunk("MSFT", [("2024-07-01", 200.0), ("2024-07-02", 201.0)]),
    ]


def _ingest_streamed(store, chunks, **kw):
    params = dict(
        provider="firstrate", symbols=["AAPL", "MSFT"], as_of="2024-07-03T00:00:00+00:00",
        source="firstratedata-import", timeframe="1d", adjustment="split_div",
    )
    params.update(kw)
    return store.ingest_bars_streamed(chunks=iter(chunks), **params)


def test_streamed_ingest_one_snapshot_reads_canonical(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    out = store.read_bars(rec.snapshot_id)
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert rec.row_count == 4
    assert rec.start == "2024-07-01" and rec.end == "2024-07-02"


def test_streamed_ingest_is_idempotent(tmp_path):
    store = DataStore(tmp_path)
    a = _ingest_streamed(store, _two_symbol_chunks())
    b = _ingest_streamed(store, _two_symbol_chunks())
    assert a.snapshot_id == b.snapshot_id
    assert len(store.list_snapshots("bars")) == 1


def test_streamed_ingest_no_orphan_on_empty(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="no bars"):
        _ingest_streamed(store, [])
    staging = tmp_path / "snapshots" / "_staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_requested_bounds_mismatch_errors(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError, match="observed coverage"):
        _ingest_streamed(store, _two_symbol_chunks(), start="2020-01-01", end="2024-07-02")


def test_large_row_warning_flag(tmp_path, monkeypatch):
    import algua.data.store as store_mod
    monkeypatch.setattr(store_mod, "IMPORT_WARN_ROWS", 1)
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    assert rec.metadata.source_metadata["servable"] == "deferred-130"


def test_clear_staging_removes_residue(tmp_path):
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging" / "leftover"
    staging.mkdir(parents=True)
    (staging / "bars.parquet").write_text("partial", encoding="utf-8")
    store.clear_staging()
    assert not (tmp_path / "snapshots" / "_staging").exists()


def test_clear_staging_noop_when_absent(tmp_path):
    DataStore(tmp_path).clear_staging()  # must not raise when nothing to clean
