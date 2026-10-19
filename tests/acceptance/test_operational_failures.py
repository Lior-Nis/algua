"""Human-reviewable fail-closed operational scenarios."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.contracts.types import OrderIntent
from algua.execution import paper_reconcile
from algua.execution.alpaca_broker import AlpacaPaperBroker, TickSnapshot
from algua.live.live_loop import TickHalted, TickHooks, run_tick
from algua.registry.db import connect, migrate
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_json
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
        return self.bars[self.bars["symbol"].isin(symbols)].copy()


class _ObservingVenue(AlpacaPaperBroker):
    def __init__(self, positions: dict[str, float] | None = None) -> None:
        super().__init__(api_key="acceptance-key", api_secret="acceptance-secret")
        self.positions = dict(positions or {})
        self.submitted: list[str] = []

    def get_positions(self) -> pd.Series:
        return pd.Series(self.positions, dtype="float64")

    def snapshot(self, universe: list[str]) -> TickSnapshot:
        symbols = set(universe) | set(self.positions)
        return TickSnapshot(
            equity=100_000.0,
            market_values={symbol: 0.0 for symbol in symbols},
            qtys={symbol: self.positions.get(symbol, 0.0) for symbol in symbols},
        )

    def cancel_open_orders(self) -> None:
        return None

    def submit_sized(
        self,
        intent: OrderIntent,
        snap: TickSnapshot,
        client_order_id: str | None = None,
        reserve: Callable[[str, float], float] | None = None,
        release: Callable[[str, float], None] | None = None,
    ) -> str:
        self.submitted.append(intent.symbol)
        return f"unexpected-order-{intent.symbol}"


def _failed_tick(
    bars: pd.DataFrame,
    venue: _ObservingVenue,
    *,
    now: pd.Timestamp,
) -> dict[str, object]:
    strategy = load_strategy("cross_sectional_momentum", reload=True)
    start = bars.index.min().to_pydatetime()
    end = (bars.index.max() + timedelta(days=1)).to_pydatetime()
    try:
        run_tick(
            strategy,
            venue,
            _ReferenceProvider(bars),
            start,
            end,
            now=now.to_pydatetime(),
        )
    except RiskBreach as exc:
        return {
            "status": "blocked",
            "breach_kind": exc.kind,
            "detail": exc.detail,
            "submitted_symbols": venue.submitted,
        }
    return {"status": "unexpected_success", "submitted_symbols": venue.submitted}


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Operational failures")
@allure.title("OPS-01 — stale or missing marks prevent every venue submission")
def test_ops_01_untrustworthy_market_marks_fail_closed() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "operational_mark_failures.json").read_text(
            encoding="utf-8"
        )
    )
    bars = decision_bars(periods=61)
    stale_now = pd.Timestamp("2026-04-20T20:00:00Z")

    actual = {
        "stale_feed": _failed_tick(bars, _ObservingVenue(), now=stale_now),
        "missing_held_symbol": _failed_tick(
            bars,
            _ObservingVenue({"ORPHAN": 2.0}),
            now=pd.Timestamp("2026-03-28T20:00:00Z"),
        ),
    }

    attach_json("Given — stale and missing market-mark cases", {
        "latest_bar": bars.index.max().isoformat(),
        "stale_now": stale_now.isoformat(),
        "missing_held_symbol": "ORPHAN",
    })
    attach_comparison("Expected vs actual mark-safety response", expected, actual)

    assert actual == expected


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Operational failures")
@allure.title("OPS-02 — a persisted kill switch blocks the next process before submission")
def test_ops_02_persisted_kill_switch_blocks_next_tick(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "operational_kill_switch.json").read_text(
            encoding="utf-8"
        )
    )
    strategy = load_strategy("cross_sectional_momentum", reload=True)
    db_path = tmp_path / "registry.db"
    conn = connect(db_path)
    migrate(conn)
    kill_switch.trip(
        conn,
        strategy.name,
        reason="acceptance risk breach",
        actor="system",
    )
    conn.close()

    restarted = connect(db_path)
    migrate(restarted)
    venue = _ObservingVenue()
    bars = decision_bars(periods=61)
    start = bars.index.min().to_pydatetime()
    end = (bars.index.max() + timedelta(days=1)).to_pydatetime()
    reason: str | None
    try:
        run_tick(
            strategy,
            venue,
            _ReferenceProvider(bars),
            start,
            end,
            now=end,
            hooks=TickHooks(
                should_halt=lambda: kill_switch.is_tripped(restarted, strategy.name)
            ),
        )
    except TickHalted as exc:
        status = "blocked"
        reason = str(exc)
    else:
        status = "unexpected_success"
        reason = None
    switch = kill_switch.get(restarted, strategy.name)
    assert switch is not None
    actual = {
        "after_restart": {
            "strategy": switch["strategy"],
            "reason": switch["reason"],
            "actor": switch["actor"],
            "is_tripped": kill_switch.is_tripped(restarted, strategy.name),
        },
        "next_tick": {
            "status": status,
            "reason": reason,
            "submitted_symbols": venue.submitted,
        },
    }
    restarted.close()

    attach_comparison("Expected vs actual persisted halt", expected, actual)

    assert actual == expected


def _reconcile_record(result, cycle: int) -> dict[str, object]:
    return {
        "cycle": cycle,
        "clean": result.clean,
        "halt": result.halt,
        "trading_allowed": result.clean,
        "mismatches": result.mismatches,
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Operational failures")
@allure.title("OPS-03 — unresolved broker drift survives restart and escalates to halt")
def test_ops_03_persistent_reconciliation_drift_escalates_after_restart(
    tmp_path: Path,
) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "operational_reconcile_drift.json").read_text(
            encoding="utf-8"
        )
    )
    db_path = tmp_path / "registry.db"
    conn = connect(db_path)
    migrate(conn)
    first_cycle = paper_reconcile.next_cycle(conn)
    first = paper_reconcile.reconcile(conn, {"AAPL": 5.0}, first_cycle)
    conn.close()

    restarted = connect(db_path)
    migrate(restarted)
    second_cycle = paper_reconcile.next_cycle(restarted)
    paper_reconcile.reconcile(restarted, {"AAPL": 5.0}, second_cycle)
    third_cycle = paper_reconcile.next_cycle(restarted)
    paper_reconcile.reconcile(restarted, {"AAPL": 5.0}, third_cycle)
    fourth_cycle = paper_reconcile.next_cycle(restarted)
    escalated = paper_reconcile.reconcile(restarted, {"AAPL": 5.0}, fourth_cycle)
    actual = {
        "first_observation": _reconcile_record(first, first_cycle),
        "after_restart_and_grace": _reconcile_record(escalated, fourth_cycle),
    }
    restarted.close()

    attach_json("Given — unexplained broker holding", {
        "ledger_expected": {},
        "broker_positions": {"AAPL": 5.0},
        "grace_cycles": 3,
    })
    attach_comparison("Expected vs actual reconcile escalation", expected, actual)

    assert actual == expected
