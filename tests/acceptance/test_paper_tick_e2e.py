"""One deterministic strategy-to-reconciled-portfolio paper tick."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import allure
import pandas as pd
import pytest

from algua.contracts.types import OrderIntent
from algua.execution import paper_reconcile
from algua.execution.alpaca_broker import AlpacaPaperBroker, TickSnapshot
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_paper_venue_broker_order_id,
    fill_cursor,
    paper_believed_positions,
    record_paper_venue_order,
)
from algua.execution.venue_sync import ingest_paper_venue
from algua.live.live_loop import SubmittedOrder, TickHooks, run_tick
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_frame, attach_json
from tests.acceptance.reference_market import decision_bars

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"


class _ReferenceProvider:
    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars

    def get_bars(
        self,
        symbols: list[str],
        start: object,
        end: object,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        assert timeframe == "1d"
        return self.bars[self.bars["symbol"].isin(symbols)].copy()


class _PaperVenue(AlpacaPaperBroker):
    def __init__(self) -> None:
        super().__init__(api_key="acceptance-key", api_secret="acceptance-secret")
        self.submissions: list[dict[str, object]] = []
        self.activities: list[dict[str, object]] = []
        self.positions: dict[str, float] = {}

    def get_positions(self) -> pd.Series:
        return pd.Series(self.positions, dtype="float64")

    def snapshot(self, universe: list[str]) -> TickSnapshot:
        return TickSnapshot(
            equity=1_000_000.0,
            market_values={symbol: 0.0 for symbol in universe},
            qtys={symbol: 0.0 for symbol in universe},
        )

    def cancel_open_orders(self) -> None:
        return None

    def submit_sized(
        self,
        intent: OrderIntent,
        _snapshot: TickSnapshot,
        client_order_id: str | None = None,
        reserve: Callable[[str, float], float] | None = None,
        release: Callable[[str, float], None] | None = None,
    ) -> str:
        order_id = f"paper-order-{intent.symbol}"
        self.submissions.append(
            {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "target_weight": intent.target_weight,
                "client_order_id": client_order_id,
                "broker_order_id": order_id,
            }
        )
        return order_id

    def account_activities_window(
        self, _after: str, _until: str
    ) -> list[dict[str, object]]:
        return self.activities


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Paper tick end to end")
@allure.title("TICK-01 — one paper tick reaches a persisted reconciled portfolio")
def test_tick_01_strategy_to_reconciled_paper_portfolio(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "paper_tick_e2e.json").read_text(encoding="utf-8")
    )
    bars = decision_bars(periods=61)
    strategy = load_strategy("cross_sectional_momentum", reload=True)
    venue = _PaperVenue()
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    strategy_record = SqliteStrategyRepository(conn).add(strategy.name)
    conn.execute("UPDATE strategies SET stage = 'paper' WHERE id = ?", (strategy_record.id,))
    conn.commit()

    def client_order_id_for(name: str, decision_ts: Any, symbol: str) -> str:
        return f"{name}-{pd.Timestamp(decision_ts).date()}-{symbol}"

    def before_submit(intent: Any, client_order_id: str | None) -> None:
        assert client_order_id is not None
        record_paper_venue_order(
            conn,
            strategy.name,
            intent.symbol,
            intent.side.value,
            1_000_000.0 * abs(intent.target_weight),
            client_order_id,
            strategy_id=strategy_record.id,
        )

    def on_submitted(order: SubmittedOrder) -> None:
        backfill_paper_venue_broker_order_id(
            conn, order.client_order_id, order.order_id
        )

    start = bars.index.min().to_pydatetime()
    end = (bars.index.max() + timedelta(days=1)).to_pydatetime()
    tick = run_tick(
        strategy,
        venue,
        _ReferenceProvider(bars),
        start,
        end,
        now=end,
        hooks=TickHooks(
            client_order_id_for=client_order_id_for,
            before_submit=before_submit,
            on_submitted=on_submitted,
        ),
    )

    for index, submission in enumerate(venue.submissions, start=1):
        symbol = str(submission["symbol"])
        order_id = str(submission["broker_order_id"])
        venue.activities.append(
            {
                "id": f"tick-fill-{index:03d}",
                "activity_type": "FILL",
                "side": "buy",
                "qty": "10.0",
                "price": "100.00",
                "symbol": symbol,
                "order_id": order_id,
                "transaction_time": "2026-03-27T20:01:00Z",
            }
        )
        venue.positions[symbol] = 10.0

    ingest_paper_venue(conn, venue, "2026-03-27T20:02:00+00:00")
    reconciled = paper_reconcile.reconcile(conn, venue.positions, cycle=1)
    actual = {
        "decision_session": pd.Timestamp(tick.decision_ts).isoformat(),
        "target_weights": tick.target_weights,
        "submitted_orders": venue.submissions,
        "venue_fills": venue.activities,
        "persisted_positions": paper_believed_positions(conn, strategy.name),
        "fill_cursor": fill_cursor(conn, LedgerKind.PAPER),
        "reconciliation": {
            "clean": reconciled.clean,
            "halt": reconciled.halt,
            "mismatches": reconciled.mismatches,
        },
    }
    conn.close()

    attach_frame("Given — canonical market bars", bars.reset_index().tail(15))
    attach_json("Actual — submitted paper orders", venue.submissions)
    attach_comparison("Expected vs actual complete paper tick", expected, actual)

    assert actual == expected
