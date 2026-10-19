"""DATA-01: a raw provider response becomes the approved canonical bars snapshot."""
from __future__ import annotations

from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.data.contracts import BarRequest, ProviderBars
from algua.data.refresh import refresh_bars
from algua.data.store import DataStore
from tests.acceptance.evidence import attach_frame, attach_json

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"


class FixtureBarProvider:
    """External-provider test double; all Algua normalization/storage remains real."""

    name = "acceptance-fixture"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_bars(self, request: BarRequest) -> ProviderBars:
        return ProviderBars(
            frame=self.frame.copy(),
            source_metadata={"fixture": "reference_market_v1"},
        )


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    out = frame.reset_index().copy()
    out["timestamp"] = out["timestamp"].map(lambda value: value.isoformat())
    return out.to_dict(orient="records")


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Data ingestion")
@allure.title("DATA-01 — raw bars become the approved canonical snapshot")
def test_data_01_raw_bars_become_the_approved_canonical_snapshot(tmp_path: Path) -> None:
    raw = pd.read_csv(FIXTURE_ROOT / "raw_bars.csv")
    expected = pd.read_csv(FIXTURE_ROOT / "expected" / "canonical_bars.csv").to_dict(
        orient="records"
    )
    store = DataStore(tmp_path / "data")

    record, refreshed = refresh_bars(
        store,
        FixtureBarProvider(raw),
        symbols=["MSFT", "AAPL"],
        start="2026-01-05",
        end="2026-01-07",
        require_bar_on="2026-01-06",
        min_rows={"AAPL": 2, "MSFT": 2},
    )
    actual = store.read_bars(record.snapshot_id)

    attach_frame("Given — raw provider bars", raw)
    attach_frame("Actual — canonical stored bars", actual.reset_index())
    attach_json(
        "Actual — stable snapshot evidence",
        {
            "snapshot_id": record.snapshot_id,
            "dataset": record.dataset.value,
            "provider": record.provider,
            "symbols": list(record.symbols),
            "start": record.start,
            "end": record.end,
            "timeframe": record.metadata.timeframe,
            "adjustment": record.metadata.adjustment,
            "row_count": record.row_count,
            "content_hash": record.content_hash,
            "storage_format": record.storage_format,
            "refreshed": refreshed,
        },
    )

    assert refreshed is True
    assert _records(actual) == expected
    assert record.row_count == 4
    assert tuple(record.symbols) == ("AAPL", "MSFT")
    assert store.verify_snapshots(record.snapshot_id)[0]["ok"] is True
