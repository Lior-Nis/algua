"""FWD-01: backtest and paper rehearsal agree at the same information cutoff."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.backtest.engine import simulate
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import run_paper
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_frame, attach_json
from tests.acceptance.reference_market import decision_bars

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Forward testing")
@allure.title("FWD-01 — backtest and paper rehearsal agree at the same cutoff")
def test_fwd_01_backtest_and_paper_rehearsal_agree_at_the_same_cutoff(
    tmp_path: Path,
) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "forward_parity.json").read_text(encoding="utf-8")
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

    _, executed_weights, forced_exits = simulate(strategy, provider, start, end)
    paper_decisions: dict[datetime, pd.Series] = {}
    paper = run_paper(
        strategy,
        SimBroker(cash=1_000_000.0),
        provider,
        start,
        end,
        on_decision=lambda decision_ts, weights: paper_decisions.__setitem__(
            decision_ts, weights
        ),
    )

    decision_session = pd.Timestamp(expected["decision_session"])
    execution_session = pd.Timestamp(expected["execution_session"])
    paper_weights = paper_decisions[decision_session]
    backtest_weights = executed_weights.loc[execution_session]
    intents = [
        {
            "symbol": order.intent.symbol,
            "side": order.intent.side.value,
            "target_weight": order.intent.target_weight,
            "decision_session": pd.Timestamp(order.intent.decision_ts).isoformat(),
        }
        for order in paper.orders
        if order.intent.decision_ts == decision_session
    ]
    fill_sessions = sorted(
        {
            pd.Timestamp(fill.fill_ts).isoformat()
            for fill in paper.fills
            if fill.decision_ts == decision_session
        }
    )
    actual = {
        "decision_session": decision_session.isoformat(),
        "execution_session": execution_session.isoformat(),
        "available_symbols": sorted(strategy.universe),
        "backtest_target_weights": backtest_weights.sort_index().to_dict(),
        "paper_target_weights": paper_weights.reindex(
            sorted(strategy.universe), fill_value=0.0
        ).to_dict(),
        "paper_order_intents": intents,
        "paper_fill_sessions": fill_sessions,
    }

    attach_frame("Given — bars available through the cutoff", bars.reset_index().tail(15))
    attach_frame(
        "Actual — backtest execution weights",
        executed_weights.loc[decision_session:execution_session].reset_index(),
    )
    attach_json("Actual — immutable snapshot provenance", {"snapshot_id": record.snapshot_id})
    attach_comparison("Expected vs actual forward parity", expected, actual)

    assert forced_exits == []
    assert paper.reconcile_ok is True
    assert actual == expected
    assert actual["paper_target_weights"] == pytest.approx(
        actual["backtest_target_weights"]
    )
