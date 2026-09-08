# algua/data/capabilities.py
from __future__ import annotations

from algua.contracts.idea import DataCapability, Horizon, Market
from algua.data.models import Dataset

# Maps a platform Dataset (an ingestion/serving path that EXISTS) to the strategy-input
# DataCapability it provides. Extend this when a new ingestion path lands (e.g. a filings
# Dataset -> FORM_13F); that single edit lets parked ideas needing it become testable.
_DATASET_CAPABILITY: dict[Dataset, DataCapability] = {
    Dataset.BARS: DataCapability.OHLCV,
}

# Markets a backtest can run against today. PRD step 5 adds crypto/forex/prediction data lanes;
# flipping a member here re-opens every idea parked on it (research idea reclassify).
_SUPPORTED_MARKETS: frozenset[Market] = frozenset({Market.US_EQUITIES, Market.ANY})
# Horizons the daily execution contract serves. PRD step 7 (intraday contract) adds INTRADAY.
_SUPPORTED_HORIZONS: frozenset[Horizon] = frozenset(
    {Horizon.DAILY, Horizon.WEEKLY, Horizon.MONTHLY, Horizon.EVENT})


def supported_capabilities() -> frozenset[DataCapability]:
    """DataCapability values the platform can provide to a backtest, derived from the data
    layer's dataset support. "Supported" = an ingestion/serving path EXISTS (demo mode serves
    OHLCV with no loaded snapshot), NOT "a snapshot is currently loaded". Today: {OHLCV}."""
    return frozenset(_DATASET_CAPABILITY.values())


def supported_markets() -> frozenset[Market]:
    return _SUPPORTED_MARKETS


def supported_horizons() -> frozenset[Horizon]:
    return _SUPPORTED_HORIZONS
