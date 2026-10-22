"""Human-reviewable paper-venue ingestion and reconciliation evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import allure
import pytest

from algua.execution import paper_reconcile
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_paper_venue_broker_order_id,
    fill_cursor,
    paper_believed_positions,
    record_paper_venue_order,
)
from algua.execution.venue_sync import ingest_paper_venue, recover_stranded
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from tests.acceptance.evidence import attach_comparison, attach_json

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"
STRATEGY = "acceptance_reconciliation"
CLIENT_ORDER_ID = "acceptance-recon-AAPL"
BROKER_ORDER_ID = "venue-order-AAPL"


class _ActivityVenue:
    """External venue boundary with an explicitly selected activity window."""

    def __init__(self, activities: list[dict[str, Any]]) -> None:
        self.activities = activities

    def account_activities_window(self, _after: str, _until: str) -> list[dict[str, Any]]:
        return self.activities


class _RecoveryVenue(_ActivityVenue):
    def __init__(
        self, activities: list[dict[str, Any]], existing_order: dict[str, Any]
    ) -> None:
        super().__init__(activities)
        self.existing_order = existing_order

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any]:
        assert client_order_id == CLIENT_ORDER_ID
        return self.existing_order


def _fill(activity_id: str, qty: float, timestamp: str) -> dict[str, object]:
    return {
        "id": activity_id,
        "activity_type": "FILL",
        "side": "buy",
        "qty": str(qty),
        "price": "100.00",
        "symbol": "AAPL",
        "order_id": BROKER_ORDER_ID,
        "transaction_time": timestamp,
    }


def _open_book(path: Path):
    conn = connect(path)
    migrate(conn)
    return conn


def _seed_order(conn) -> None:
    strategy = SqliteStrategyRepository(conn).add(STRATEGY)
    conn.execute("UPDATE strategies SET stage = 'paper' WHERE id = ?", (strategy.id,))
    conn.commit()
    assert record_paper_venue_order(
        conn,
        STRATEGY,
        "AAPL",
        "buy",
        1_000.0,
        CLIENT_ORDER_ID,
        strategy_id=strategy.id,
    )
    backfill_paper_venue_broker_order_id(conn, CLIENT_ORDER_ID, BROKER_ORDER_ID)


def _book_state(conn, broker_qty: float, cycle: int) -> dict[str, object]:
    reconciliation = paper_reconcile.reconcile(conn, {"AAPL": broker_qty}, cycle)
    return {
        "believed_positions": paper_believed_positions(conn, STRATEGY),
        "fill_cursor": fill_cursor(conn, LedgerKind.PAPER),
        "reconciliation": {
            "clean": reconciliation.clean,
            "halt": reconciliation.halt,
            "mismatches": reconciliation.mismatches,
        },
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Paper reconciliation")
@allure.title("RECON-01 — partial fills survive restart and replay without double counting")
def test_recon_01_partial_fill_restart_and_replay_are_idempotent(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "paper_fill_reconciliation.json").read_text(
            encoding="utf-8"
        )
    )
    db_path = tmp_path / "registry.db"
    first_fill = _fill("fill-001", 4.0, "2026-03-27T14:31:00Z")
    final_fill = _fill("fill-002", 6.0, "2026-03-27T14:32:00Z")

    conn = _open_book(db_path)
    _seed_order(conn)
    ingest_paper_venue(conn, _ActivityVenue([first_fill]), "2026-03-27T14:31:30+00:00")
    after_partial_fill = _book_state(conn, broker_qty=4.0, cycle=1)
    conn.close()

    restarted = _open_book(db_path)
    ingest_paper_venue(
        restarted,
        _ActivityVenue([first_fill]),
        "2026-03-27T14:32:00+00:00",
    )
    after_restart_replay = _book_state(restarted, broker_qty=4.0, cycle=2)

    ingest_paper_venue(
        restarted,
        _ActivityVenue([first_fill, final_fill]),
        "2026-03-27T14:33:00+00:00",
    )
    after_final_fill = _book_state(restarted, broker_qty=10.0, cycle=3)
    restarted.close()

    actual = {
        "submitted_order": {
            "strategy": STRATEGY,
            "symbol": "AAPL",
            "side": "buy",
            "intended_notional": 1_000.0,
            "client_order_id": CLIENT_ORDER_ID,
            "broker_order_id": BROKER_ORDER_ID,
        },
        "venue_activities": [first_fill, final_fill],
        "after_partial_fill": after_partial_fill,
        "after_restart_replay": after_restart_replay,
        "after_final_fill": after_final_fill,
    }

    attach_json("Given — submitted order and venue fills", {
        "order": actual["submitted_order"],
        "activities": actual["venue_activities"],
    })
    attach_comparison("Expected vs actual reconciled book", expected, actual)

    assert actual == expected


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Paper reconciliation")
@allure.title("RECON-02 — rejected and cancelled venue events never become positions")
def test_recon_02_rejected_and_cancelled_orders_do_not_create_positions(
    tmp_path: Path,
) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "paper_terminal_order_events.json").read_text(
            encoding="utf-8"
        )
    )
    terminal_events = [
        {
            "id": "event-rejected-001",
            "activity_type": "ORDER_REJECTED",
            "symbol": "AAPL",
            "order_id": BROKER_ORDER_ID,
            "transaction_time": "2026-03-27T14:31:00Z",
        },
        {
            "id": "event-cancelled-001",
            "activity_type": "ORDER_CANCELED",
            "symbol": "AAPL",
            "order_id": BROKER_ORDER_ID,
            "transaction_time": "2026-03-27T14:32:00Z",
        },
    ]

    conn = _open_book(tmp_path / "registry.db")
    _seed_order(conn)
    ingest_paper_venue(
        conn,
        _ActivityVenue(terminal_events),
        "2026-03-27T14:33:00+00:00",
    )
    actual = {
        "venue_activities": terminal_events,
        "book": _book_state(conn, broker_qty=0.0, cycle=1),
    }
    conn.close()

    attach_json("Given — terminal non-fill venue events", terminal_events)
    attach_comparison("Expected vs actual terminal-event book", expected, actual)

    assert actual == expected


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Paper reconciliation")
@allure.title("RECON-03 — restart recovers an order stranded between submit and backfill")
def test_recon_03_restart_recovers_crash_stranded_order(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "paper_stranded_order_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    db_path = tmp_path / "registry.db"
    conn = _open_book(db_path)
    strategy = SqliteStrategyRepository(conn).add(STRATEGY)
    conn.execute("UPDATE strategies SET stage = 'paper' WHERE id = ?", (strategy.id,))
    conn.commit()
    record_paper_venue_order(
        conn,
        STRATEGY,
        "AAPL",
        "buy",
        1_000.0,
        CLIENT_ORDER_ID,
        strategy_id=strategy.id,
    )
    conn.close()

    landed_order = {
        "id": BROKER_ORDER_ID,
        "client_order_id": CLIENT_ORDER_ID,
        "symbol": "AAPL",
        "side": "buy",
        "status": "filled",
    }
    fill = _fill("fill-after-crash-001", 10.0, "2026-03-27T14:31:00Z")
    venue = _RecoveryVenue([fill], landed_order)
    restarted = _open_book(db_path)
    recover_stranded(restarted, venue, LedgerKind.PAPER)
    ingest_paper_venue(restarted, venue, "2026-03-27T14:32:00+00:00")
    actual = {
        "crash_point": {
            "intent_persisted": True,
            "broker_order_id_backfilled": False,
        },
        "venue_order_found_after_restart": landed_order,
        "recovered_book": _book_state(restarted, broker_qty=10.0, cycle=1),
    }
    restarted.close()

    attach_json("Given — crash-stranded intent and landed venue order", {
        "crash_point": actual["crash_point"],
        "venue_order": landed_order,
    })
    attach_comparison("Expected vs actual stranded-order recovery", expected, actual)

    assert actual == expected
