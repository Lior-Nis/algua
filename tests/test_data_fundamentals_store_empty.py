"""FIX G: ingest_fundamentals rejects an empty frame."""
from __future__ import annotations

import pytest

from algua.data.fundamentals_schema import empty_fundamentals
from algua.data.store import DataStore


def test_ingest_empty_fundamentals_raises(tmp_path):
    store = DataStore(tmp_path)
    empty = empty_fundamentals()
    with pytest.raises(ValueError, match="empty"):
        store.ingest_fundamentals(
            provider="test",
            symbols=[],
            as_of="2025-09-01T00:00:00Z",
            source="test",
            frame=empty,
        )
