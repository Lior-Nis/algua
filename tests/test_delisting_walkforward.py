"""Integration tests for delisting overlay threaded through walk_forward() — Task 6 of #212."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from algua.backtest.delisting import DelistingRecord
from algua.backtest.walkforward import walk_forward
from tests.test_delisting_simulate import _FakeProvider, _equal_weight_ab_strategy


def _long_bars() -> pd.DataFrame:
    """25-bar panel for A (all 25) and B (first 5, then delisted after 2020-01-05).

    walk_forward needs >=5 bars per window; with windows=2 + holdout_frac=0.2 we get
    20 train bars (10/window) + 5 holdout bars — comfortably above the floor.
    """
    idx = pd.date_range("2020-01-01", periods=25, freq="D", tz="UTC")
    rows: list[dict[str, Any]] = []
    # A: all 25 bars
    for i, ts in enumerate(idx):
        px = 10.0 + i * 0.1
        rows.append({
            "timestamp": ts, "symbol": "A",
            "open": px, "high": px, "low": px,
            "close": px, "adj_close": px, "volume": 100.0,
        })
    # B: first 5 bars only (delists after 2020-01-05)
    for i, ts in enumerate(idx[:5]):
        px = 10.0 + i * 0.1
        rows.append({
            "timestamp": ts, "symbol": "B",
            "open": px, "high": px, "low": px,
            "close": px, "adj_close": px, "volume": 100.0,
        })
    return pd.DataFrame(rows).set_index("timestamp").sort_values(["timestamp", "symbol"])


def test_walk_forward_threads_delisting_records() -> None:
    """walk_forward must accept delisting_records and pass them through to simulate().

    B disappears after bar 5.  Without a delisting record simulate() raises BacktestError.
    With the record the run completes end-to-end and the holdout sharpe is finite.
    """
    strat = _equal_weight_ab_strategy()
    provider = _FakeProvider(_long_bars())
    recs = {"B": [DelistingRecord(date(2020, 1, 5), 5.0, "vendor")]}

    wf = walk_forward(
        strat, provider,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 25, tzinfo=UTC),
        windows=2,
        holdout_frac=0.2,
        delisting_records=recs,
    )

    # Ran cleanly end-to-end → finite holdout sharpe.
    assert np.isfinite(wf.holdout_metrics["sharpe"]), (
        f"expected finite holdout sharpe, got {wf.holdout_metrics['sharpe']}"
    )
