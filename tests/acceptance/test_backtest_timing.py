"""BT-02: a decision on closed bar t becomes executable only on bar t+1."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.backtest.engine import simulate
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import (
    attach_comparison,
    attach_frame,
    attach_json,
    stable_frame_hash,
)
from tests.acceptance.reference_market import decision_bars

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Backtest")
@allure.title("BT-02 — a close-of-bar decision executes on the next bar")
def test_bt_02_decision_at_t_executes_at_t_plus_one(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "backtest_timing.json").read_text(encoding="utf-8")
    )
    bars = decision_bars(periods=62)
    store = DataStore(tmp_path / "data")
    record = store.ingest_bars(
        provider="acceptance-fixture",
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        start=bars.index.min().date().isoformat(),
        end=(bars.index.max() + timedelta(days=1)).date().isoformat(),
        as_of="2026-04-01T00:00:00+00:00",
        source="reference_market_v1",
        frame=bars.reset_index().rename(columns={"timestamp": "ts"}),
        timeframe="1d",
        adjustment="none",
    )
    provider = StoreBackedProvider(store, record.snapshot_id)
    strategy = load_strategy("cross_sectional_momentum", reload=True)
    start = bars.index.min().to_pydatetime()
    end = (bars.index.max() + timedelta(days=1)).to_pydatetime()

    portfolio, executed_weights, forced_exits = simulate(strategy, provider, start, end)
    decision_session = pd.Timestamp(expected["decision_session"])
    execution_session = pd.Timestamp(expected["execution_session"])
    decision_weights = strategy.target_weights(bars.loc[:decision_session])
    orders = portfolio.orders.records_readable
    actual = {
        "decision_session": decision_session.isoformat(),
        "execution_session": execution_session.isoformat(),
        "order_count": len(orders),
        "target_weights": decision_weights.sort_index().to_dict(),
        "order_list_hash": stable_frame_hash(orders),
    }

    attach_frame("Given — bars around the decision", bars.reset_index().tail(15))
    attach_frame(
        "Actual — executed weights around the decision",
        executed_weights.loc[decision_session:execution_session].reset_index(),
    )
    attach_frame("Actual — orders", orders)
    attach_json("Actual — backtest provenance", {"snapshot_id": record.snapshot_id})
    attach_comparison("Expected vs actual timing", expected, actual)

    assert forced_exits == []
    assert executed_weights.loc[decision_session].abs().sum() == pytest.approx(0.0)
    assert executed_weights.loc[execution_session].to_dict() == pytest.approx(
        {
            "AAPL": expected["target_weights"]["AAPL"],
            "AMZN": 0.0,
            "GOOGL": 0.0,
            "MSFT": expected["target_weights"]["MSFT"],
            "NVDA": expected["target_weights"]["NVDA"],
        }
    )
    assert len(orders) == expected["order_count"]
