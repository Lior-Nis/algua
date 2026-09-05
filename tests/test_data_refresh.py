"""refresh_bars (#556): resolve-or-ingest behind a per-symbol coverage wall."""
from __future__ import annotations

import pandas as pd
import pytest

from algua.data.contracts import ProviderBars
from algua.data.models import Dataset
from algua.data.refresh import RefreshError, refresh_bars
from algua.data.store import DataStore

_START, _END, _REQ = "2026-01-05", "2026-01-09", "2026-01-08"


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(symbol, iso_date), ...] -> one bar per row, daily UTC-midnight stamps."""
    n = len(rows)
    return pd.DataFrame({
        "ts": [f"{d}T00:00:00+00:00" for _, d in rows],
        "symbol": [s for s, _ in rows],
        "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n, "close": [1.0] * n,
        "adj_close": [1.0] * n, "volume": [1.0] * n,
    })


class _Provider:
    name = "fake"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame, self.calls = frame, 0

    def get_bars(self, request):
        self.calls += 1
        return ProviderBars(frame=self.frame.copy(), source_metadata={"provider": "fake"})


_FULL = _frame([("AAPL", "2026-01-07"), ("AAPL", "2026-01-08"),
                ("MSFT", "2026-01-07"), ("MSFT", "2026-01-08")])


def _call(store, provider, **kw):
    args = dict(symbols=["msft", "AAPL"], start=_START, end=_END, require_bar_on=_REQ)
    args.update(kw)
    return refresh_bars(store, provider, **args)


def test_miss_ingests_and_reports_refreshed(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    rec, refreshed = _call(store, provider)
    assert refreshed is True and provider.calls == 1
    assert rec.dataset is Dataset.BARS and tuple(rec.symbols) == ("AAPL", "MSFT")
    assert rec.provider == "fake"
    assert (rec.metadata.start, rec.metadata.end) == (_START, _END)


def test_hit_reuses_without_provider_call(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    second, refreshed = _call(store, provider, symbols=["AAPL", "MSFT"])  # order-insensitive
    assert refreshed is False and provider.calls == 1
    assert second.snapshot_id == first.snapshot_id


def test_different_end_is_a_miss(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    second, refreshed = _call(store, provider, end="2026-01-10", require_bar_on="2026-01-08")
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_same_key_candidate_failing_current_coverage_is_not_reused(tmp_path):
    """A lax earlier refresh (accepted for 01-07) under the SAME key must not satisfy a later
    01-08 requirement: the candidate is re-validated and a fresh snapshot minted."""
    store = DataStore(tmp_path)
    lax = _Provider(_frame([("AAPL", "2026-01-07"), ("MSFT", "2026-01-07")]))
    first, _ = _call(store, lax, require_bar_on="2026-01-07")
    fresh = _Provider(_FULL)
    second, refreshed = _call(store, fresh, require_bar_on="2026-01-08")
    assert refreshed is True and fresh.calls == 1
    assert second.snapshot_id != first.snapshot_id


def test_same_key_candidate_with_short_history_is_not_reused(tmp_path):
    store = DataStore(tmp_path)
    first, _ = _call(store, _Provider(_FULL))  # 2 rows per symbol
    fresh = _Provider(_frame([("AAPL", "2026-01-06"), ("AAPL", "2026-01-07"),
                              ("AAPL", "2026-01-08"), ("MSFT", "2026-01-06"),
                              ("MSFT", "2026-01-07"), ("MSFT", "2026-01-08")]))
    second, refreshed = _call(store, fresh, min_rows={"AAPL": 3, "MSFT": 3})
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_tampered_candidate_with_equal_row_count_is_not_reused(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    # Rewrite every partition with the same row count but different closes.
    for p in first.data_path.rglob("*.parquet") if first.data_path.is_absolute() else \
            (tmp_path / first.data_path).rglob("*.parquet"):
        df = pd.read_parquet(p)
        df["close"] = df["close"] + 1.0
        df.to_parquet(p, index=False)
    second, refreshed = _call(store, provider)
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_missing_symbol_fails_closed_and_mints_nothing(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08")])))
    assert exc.value.kind == "missing" and exc.value.symbols == ["MSFT"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_stale_symbol_fails_closed_and_mints_nothing(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08"), ("MSFT", "2026-01-07")])))
    assert exc.value.kind == "stale" and exc.value.symbols == ["MSFT"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_misdated_newer_bar_fails_closed(tmp_path):
    # A row dated AFTER require_bar_on but before end (a weekend row from a bad vendor).
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08"), ("MSFT", "2026-01-08"),
                                       ("MSFT", "2026-01-09")])),
              require_bar_on="2026-01-08", end="2026-01-10", timeframe="1d")
    # to_bar_schema may reject the intraday stamp first
    assert exc.value.kind in {"misdated", "stale"}
    assert store.list_snapshots(Dataset.BARS) == []


def test_short_history_fails_closed(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_FULL), min_rows={"AAPL": 5})
    assert exc.value.kind == "short_history" and exc.value.symbols == ["AAPL"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_rows_at_or_after_end_are_clipped(tmp_path):
    store = DataStore(tmp_path)
    provider = _Provider(_frame([("AAPL", "2026-01-08"), ("AAPL", "2026-01-09"),
                                 ("MSFT", "2026-01-08"), ("MSFT", "2026-01-09")]))
    rec, _ = _call(store, provider)
    assert store.read_bars(rec.snapshot_id).index.max().date().isoformat() == "2026-01-08"


def test_provider_error_propagates_and_mints_nothing(tmp_path):
    class _Boom:
        name = "boom"

        def get_bars(self, request):
            raise ConnectionError("vendor down")

    store = DataStore(tmp_path)
    with pytest.raises(ConnectionError):
        _call(store, _Boom())
    assert store.list_snapshots(Dataset.BARS) == []
