import pytest

from algua.data.store import DataStore


def _ingest(store, syms, eff):
    return store.ingest_universe(
        universe="SP500", symbols=syms, effective_date=eff,
        as_of="2026-01-01T00:00:00+00:00", source="test", require_immutable=True,
    )


def test_same_date_same_membership_is_idempotent(tmp_path):
    store = DataStore(tmp_path)
    a = _ingest(store, ["AAPL", "MSFT"], "2000-01-01")
    b = _ingest(store, ["MSFT", "AAPL"], "2000-01-01")  # same set, different order
    assert a.snapshot_id == b.snapshot_id  # content-hash dedup


def test_same_date_different_membership_rejected_before_write(tmp_path):
    store = DataStore(tmp_path)
    _ingest(store, ["AAPL", "MSFT"], "2000-01-01")
    before = store.data_dir.joinpath("manifest.jsonl").read_text()
    with pytest.raises(ValueError, match="immutab|conflict|differ"):
        _ingest(store, ["AAPL", "GOOG"], "2000-01-01")
    after = store.data_dir.joinpath("manifest.jsonl").read_text()
    assert before == after  # rejected import left the manifest unmutated


def test_non_immutable_path_unaffected(tmp_path):
    store = DataStore(tmp_path)
    store.ingest_universe(
        universe="U", symbols=["A"], effective_date="2000-01-01",
        as_of="2026-01-01T00:00:00+00:00", source="test",
    )  # require_immutable defaults False — no conflict checking
