# algua/data/capabilities.py
from __future__ import annotations

from algua.contracts.idea import DataCapability
from algua.data.models import Dataset

# Maps a platform Dataset (an ingestion/serving path that EXISTS) to the strategy-input
# DataCapability it provides. Extend this when a new ingestion path lands (e.g. a filings
# Dataset -> FORM_13F); that single edit lets parked ideas needing it become testable.
_DATASET_CAPABILITY: dict[Dataset, DataCapability] = {
    Dataset.BARS: DataCapability.OHLCV,
}


def supported_capabilities() -> frozenset[DataCapability]:
    """DataCapability values the platform can provide to a backtest, derived from the data
    layer's dataset support. "Supported" = an ingestion/serving path EXISTS (demo mode serves
    OHLCV with no loaded snapshot), NOT "a snapshot is currently loaded". Today: {OHLCV}."""
    return frozenset(_DATASET_CAPABILITY.values())
