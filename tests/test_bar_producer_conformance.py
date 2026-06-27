from algua.data.contracts import FirstRateImportRequest
from algua.data.importers.firstrate import FirstRateImporter
from algua.data.schema import to_bar_schema, validate_bars
from algua.data.store import DataStore


def _write_pair(raw, adj, sym, price):
    (raw / f"{sym}_full_1day_UNADJUSTED.txt").write_text(
        f"2024-07-01,{price},{price},{price},{price},10\n"
        f"2024-07-02,{price},{price},{price},{price},10\n", encoding="utf-8")
    # Adjusted series ANCHORED at the last bar (adj close == raw close `price`); the older bar
    # carries the split adjustment (price/2). A back-adjusted full series is anchored at its most
    # recent bar, which #265's import-time check now enforces.
    (adj / f"{sym}_full_1day_adjsplitdiv.txt").write_text(
        f"2024-07-01,{price / 2},{price / 2},{price / 2},{price / 2},10\n"
        f"2024-07-02,{price},{price},{price},{price},10\n", encoding="utf-8")


def _dirs(tmp_path, name):
    raw = tmp_path / f"{name}_raw"
    adj = tmp_path / f"{name}_adj"
    raw.mkdir()
    adj.mkdir()
    return raw, adj


def test_importer_output_is_bar_schema_valid(tmp_path):
    raw, adj = _dirs(tmp_path, "c")
    for sym in ["AAPL", "MSFT"]:
        _write_pair(raw, adj, sym, 100)
    req = FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)
    for chunk in FirstRateImporter().import_bars(req):
        # Same terminal boundary both seams must satisfy.
        validate_bars(to_bar_schema(chunk.frame))


def test_snapshot_id_is_discovery_order_invariant(tmp_path):
    # Two dirs with identical content; the importer must sort symbols so the snapshot_id is stable
    # regardless of os listing order. Ingest twice and assert equal ids (dedup proves stability).
    raw, adj = _dirs(tmp_path, "d")
    for sym in ["MSFT", "AAPL", "GOOG"]:
        _write_pair(raw, adj, sym, 100)
    store = DataStore(tmp_path / "store")

    def _ingest():
        chunks = (c.frame for c in FirstRateImporter().import_bars(
            FirstRateImportRequest(raw_dir=raw, adjusted_dir=adj)))
        return store.ingest_bars_streamed(
            provider="firstrate", symbols=["AAPL", "GOOG", "MSFT"],
            as_of="2024-07-03T00:00:00+00:00", source="firstratedata-import",
            chunks=chunks, timeframe="1d", adjustment="split_div",
        )

    assert _ingest().snapshot_id == _ingest().snapshot_id
