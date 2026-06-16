from datetime import date

import pandas as pd
import pytest

from algua.backtest.delisting import DelistingRecord
from algua.data.store import DataStore


def test_ingest_and_read_delistings_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["ENRN", "WCOM", "ENRN"],
         "delisting_date": ["2001-11-28", "2002-07-01", "1990-01-01"],
         "delisting_value": [0.25, 0.10, 3.0]}
    )
    store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="vendor")
    recs = store.read_delistings()
    assert set(recs) == {"ENRN", "WCOM"}
    assert len(recs["ENRN"]) == 2  # two events for one symbol -> list
    enrn_dates = {r.delisting_date for r in recs["ENRN"]}
    assert enrn_dates == {date(2001, 11, 28), date(1990, 1, 1)}
    assert all(isinstance(r, DelistingRecord) for r in recs["WCOM"])


def test_ingest_delistings_rejects_nonpositive_value(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["X"], "delisting_date": ["2001-01-01"], "delisting_value": [0.0]}
    )
    with pytest.raises(ValueError, match="terminal_price|> 0|zero-proceeds"):
        store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="v")


def test_ingest_delistings_rejects_duplicate_event(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["X", "X"], "delisting_date": ["2001-01-01", "2001-01-01"],
         "delisting_value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="v")


def test_read_delistings_empty_when_none(tmp_path):
    assert DataStore(tmp_path).read_delistings() == {}
