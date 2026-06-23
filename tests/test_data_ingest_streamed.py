import pandas as pd
import pytest

from algua.data.contracts import FirstRateImportRequest
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
    req = FirstRateImportRequest(raw_dir=tmp_path / "raw", adjusted_dir=tmp_path / "adj")
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
    assert rec.storage_format == "parquet_dataset"
    out = store.read_bars(rec.snapshot_id)
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert rec.row_count == 4
    assert rec.start == "2024-07-01" and rec.end == "2024-07-02"


def test_streamed_ingest_pushdown_subset(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    # symbol pushdown
    only_aapl = store.read_bars(rec.snapshot_id, symbols=["AAPL"])
    assert set(only_aapl["symbol"]) == {"AAPL"}
    assert len(only_aapl) == 2
    # half-open [start, end) ts pushdown: only 2024-07-01 bars
    first_day = store.read_bars(
        rec.snapshot_id,
        start=pd.Timestamp("2024-07-01", tz="UTC"),
        end=pd.Timestamp("2024-07-02", tz="UTC"),
    )
    validate_bars(first_day)
    assert set(first_day["symbol"]) == {"AAPL", "MSFT"}
    assert len(first_day) == 2
    assert first_day.index.unique().tolist() == [pd.Timestamp("2024-07-01", tz="UTC")]
    # combined symbol + ts pushdown
    aapl_first = store.read_bars(
        rec.snapshot_id,
        symbols=["AAPL"],
        start=pd.Timestamp("2024-07-01", tz="UTC"),
        end=pd.Timestamp("2024-07-02", tz="UTC"),
    )
    assert set(aapl_first["symbol"]) == {"AAPL"}
    assert len(aapl_first) == 1


def test_streamed_ingest_id_independent_of_chunk_order(tmp_path):
    store_ab = DataStore(tmp_path / "ab")
    store_ba = DataStore(tmp_path / "ba")
    chunks = _two_symbol_chunks()  # [AAPL, MSFT]
    rec_ab = _ingest_streamed(store_ab, chunks)
    rec_ba = _ingest_streamed(store_ba, list(reversed(chunks)))  # [MSFT, AAPL]
    assert rec_ab.snapshot_id == rec_ba.snapshot_id
    assert rec_ab.content_hash == rec_ba.content_hash


def test_streamed_ingest_adopts_orphan_dataset(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest_streamed(store, _two_symbol_chunks())
    dataset_dir = tmp_path / rec.data_path
    assert dataset_dir.is_dir()

    # Simulate an orphan: the committed dataset dir survives a crash but the manifest record for it
    # never landed. Rewrite the manifest jsonl without that record's line.
    manifest_path = tmp_path / "manifest.jsonl"
    lines = [
        ln
        for ln in manifest_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and rec.snapshot_id not in ln
    ]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    assert store.manifest.find(rec.snapshot_id) is None
    assert dataset_dir.is_dir()  # the orphan data dir is still on disk

    # Re-ingesting the same chunks must adopt the orphan dir (not raise on os.replace onto a
    # non-empty target dir), end with exactly one record, and serve the data.
    re_rec = _ingest_streamed(store, _two_symbol_chunks())
    assert re_rec.snapshot_id == rec.snapshot_id
    assert len(store.list_snapshots("bars")) == 1
    out = store.read_bars(re_rec.snapshot_id)
    validate_bars(out)
    assert len(out) == 4


def test_streamed_ingest_propagates_unexpected_replace_error(tmp_path, monkeypatch):
    # An os.replace failure that is NOT "target dir exists" (e.g. permission) must propagate, not be
    # mistaken for an adoptable orphan.
    import errno
    import os

    store = DataStore(tmp_path)

    def boom(src, dst):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(PermissionError):
        _ingest_streamed(store, _two_symbol_chunks())


def test_streamed_ingest_dotted_symbol_roundtrips(tmp_path):
    store = DataStore(tmp_path)
    chunk = _chunk("BRK.B", [("2024-07-01", 410.0), ("2024-07-02", 412.0)])
    rec = store.ingest_bars_streamed(
        provider="firstrate",
        symbols=["BRK.B"],
        as_of="2024-07-03T00:00:00+00:00",
        source="firstratedata-import",
        chunks=iter([chunk]),
        timeframe="1d",
        adjustment="split_div",
    )
    assert rec.storage_format == "parquet_dataset"
    out = store.read_bars(rec.snapshot_id, symbols=["BRK.B"])
    validate_bars(out)
    assert set(out["symbol"]) == {"BRK.B"}
    assert len(out) == 2


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


def test_clear_staging_removes_stale_residue(tmp_path):
    import os
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging" / "leftover"
    staging.mkdir(parents=True)
    (staging / "bars.parquet").write_text("partial", encoding="utf-8")
    old = __import__("time").time() - 7200  # 2h ago, older than the 1h default
    os.utime(staging, (old, old))
    store.clear_staging()
    assert not staging.exists()


def test_clear_staging_keeps_fresh_residue(tmp_path):
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging" / "active"
    staging.mkdir(parents=True)
    store.clear_staging()  # fresh dir must survive (could be a concurrent active import)
    assert staging.exists()


def test_clear_staging_noop_when_absent(tmp_path):
    DataStore(tmp_path).clear_staging()  # must not raise when nothing to clean


def test_clear_staging_spares_old_dir_with_held_lease(tmp_path):
    # #255: a >1h in-progress import's staging-root mtime is stale, but its flock lease is held —
    # clear_staging must NOT delete it out from under the writer.
    import fcntl
    import os as _os
    import time as _time
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging"
    active = staging / "deadbeef"
    active.mkdir(parents=True)
    (active / "symbol=AAA").mkdir()
    lock_path = staging / "deadbeef.lock"
    fd = _os.open(lock_path, _os.O_RDWR | _os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)  # simulate a live writer holding the lease
    try:
        old = _time.time() - 7200  # 2h ago, older than the 1h default
        _os.utime(active, (old, old))
        _os.utime(lock_path, (old, old))
        store.clear_staging()
        assert active.exists()  # spared: an active lease overrides the stale mtime
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        _os.close(fd)


def test_clear_staging_sweeps_old_dir_with_unheld_lease(tmp_path):
    # #255: crash residue — the dir + an unheld lock marker, both old — is swept, marker and all.
    import os as _os
    import time as _time
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging"
    dead = staging / "c0ffee"
    dead.mkdir(parents=True)
    lock_path = staging / "c0ffee.lock"
    lock_path.write_bytes(b"")  # marker left by a crash, no writer holding it
    old = _time.time() - 7200
    for p in (dead, lock_path):
        _os.utime(p, (old, old))
    store.clear_staging()
    assert not dead.exists()
    assert not lock_path.exists()  # orphan marker cleaned too


def test_clear_staging_fails_closed_when_lock_probe_errors(tmp_path):
    # #255 (Codex GATE-2): _lock_held must FAIL CLOSED on a probe error that isn't "file absent".
    # A .lock that is a directory makes os.open(O_RDWR) raise IsADirectoryError (an OSError, not
    # FileNotFoundError) — cleanup must treat that as "can't prove abandoned" and spare the dir.
    import os as _os
    import time as _time
    store = DataStore(tmp_path)
    staging = tmp_path / "snapshots" / "_staging"
    dirpath = staging / "feed"
    dirpath.mkdir(parents=True)
    (staging / "feed.lock").mkdir()  # unprobeable marker (a dir, not a lock file)
    old = _time.time() - 7200
    for p in (dirpath, staging / "feed.lock"):
        _os.utime(p, (old, old))
    store.clear_staging()
    assert dirpath.exists()  # spared: an unprobeable lease marker is not provably abandoned


def test_new_leased_staging_cleans_up_on_lock_failure(tmp_path, monkeypatch):
    # #255 (Codex round 2): the lease helper must be self-cleaning if flock fails before the
    # caller's try/finally takes over — no leaked fd, no orphan dir/marker residue.
    store = DataStore(tmp_path)

    def _boom(*_a, **_k):
        raise OSError("flock unsupported here")

    monkeypatch.setattr("algua.data.store.fcntl.flock", _boom)
    with pytest.raises(OSError, match="flock"):
        store._new_leased_staging()
    staging = tmp_path / "snapshots" / "_staging"
    assert not staging.exists() or list(staging.iterdir()) == []  # nothing left behind


def test_streamed_ingest_leaves_no_lock_residue(tmp_path):
    # the sibling lease marker is removed in the finally; a successful ingest leaves no .lock cruft.
    store = DataStore(tmp_path)
    _ingest_streamed(store, _two_symbol_chunks())
    staging = tmp_path / "snapshots" / "_staging"
    leftover = list(staging.iterdir()) if staging.exists() else []
    assert leftover == [], f"unexpected staging residue: {leftover}"


def test_streamed_ingest_rejects_symbol_split_across_chunks(tmp_path):
    store = DataStore(tmp_path)
    chunks = [
        _chunk("AAPL", [("2024-07-01", 100.0)]),
        _chunk("AAPL", [("2024-07-02", 101.0)]),  # same symbol again -> reject
    ]
    with pytest.raises(ValueError, match="more than one chunk"):
        _ingest_streamed(store, chunks)
