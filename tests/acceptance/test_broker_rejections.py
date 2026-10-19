"""Human-reviewable evidence for paper-broker rejection behavior."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import allure
import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution import alpaca_broker
from algua.execution.alpaca_broker import AlpacaPaperBroker, TickSnapshot
from algua.execution.errors import BrokerError
from tests.acceptance.evidence import attach_comparison, attach_json

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"
DECISION_TS = datetime(2026, 3, 27, tzinfo=UTC)


class _VenueResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, separators=(",", ":"))

    def json(self) -> dict[str, Any]:
        return self._payload


class _RejectingVenue:
    """Deterministic replacement for the external Alpaca HTTP boundary."""

    def __init__(self, response: _VenueResponse) -> None:
        self.response = response
        self.posted: list[dict[str, Any]] = []

    def post(self, _url: str, **kwargs: Any) -> _VenueResponse:
        body = kwargs.get("json")
        assert isinstance(body, dict)
        self.posted.append(body)
        return self.response


class _DuplicateVenue(_RejectingVenue):
    def __init__(self, existing_order: dict[str, Any]) -> None:
        super().__init__(
            _VenueResponse(
                422,
                {"code": 42210000, "message": "client_order_id must be unique"},
            )
        )
        self.existing_order = existing_order
        self.lookups: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _VenueResponse:
        self.lookups.append(url.rsplit("/", maxsplit=1)[-1])
        return _VenueResponse(200, self.existing_order)


def _broker() -> AlpacaPaperBroker:
    return AlpacaPaperBroker(api_key="acceptance-key", api_secret="acceptance-secret")


def _intent() -> OrderIntent:
    return OrderIntent("AAPL", Side.BUY, 0.5, DECISION_TS)


def _snapshot() -> TickSnapshot:
    return TickSnapshot(
        equity=10_000.0,
        market_values={"AAPL": 0.0},
        qtys={"AAPL": 0.0},
    )


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Broker boundary")
@allure.title("BROKER-01 — an ordinary venue rejection cannot become a successful order")
def test_broker_01_ordinary_venue_rejection_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "broker_venue_rejection.json").read_text(
            encoding="utf-8"
        )
    )
    rejection = _VenueResponse(
        422,
        {"code": 40310000, "message": "insufficient buying power"},
    )
    venue = _RejectingVenue(rejection)
    monkeypatch.setattr(alpaca_broker, "requests", venue)

    try:
        returned_order_id = _broker().submit_sized(
            _intent(), _snapshot(), "acceptance-20260327-AAPL"
        )
    except BrokerError as exc:
        outcome = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "returned_order_id": None,
        }
    else:
        outcome = {
            "status": "accepted",
            "error_type": None,
            "error": None,
            "returned_order_id": returned_order_id,
        }

    actual = {
        "submitted_order": venue.posted[0],
        "venue_response": {
            "status_code": rejection.status_code,
            "body": rejection.json(),
        },
        "adapter_outcome": outcome,
    }

    attach_json("Given — deterministic venue rejection", actual["venue_response"])
    attach_comparison("Expected vs actual broker outcome", expected, actual)

    assert actual == expected


def _duplicate_outcome(monkeypatch: pytest.MonkeyPatch, status: str) -> dict[str, object]:
    client_order_id = f"acceptance-20260327-AAPL-{status}"
    existing_order = {
        "id": f"venue-order-{status}",
        "client_order_id": client_order_id,
        "symbol": "AAPL",
        "side": "buy",
        "status": status,
    }
    venue = _DuplicateVenue(existing_order)
    monkeypatch.setattr(alpaca_broker, "requests", venue)

    result = _broker().submit_sized(_intent(), _snapshot(), client_order_id)

    return {
        "existing_order": existing_order,
        "submitted_order": venue.posted[0],
        "lookup": venue.lookups[0],
        "adapter_result": result,
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Broker boundary")
@allure.title("BROKER-02 — duplicate IDs recover only orders that can still execute")
def test_broker_02_duplicate_id_distinguishes_active_from_dead_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "broker_duplicate_recovery.json").read_text(
            encoding="utf-8"
        )
    )

    actual = {
        "active_duplicate": _duplicate_outcome(monkeypatch, "accepted"),
        "dead_duplicate": _duplicate_outcome(monkeypatch, "rejected"),
    }

    attach_json(
        "Given — duplicate client-order-id venue response",
        {
            "status_code": 422,
            "body": {
                "code": 42210000,
                "message": "client_order_id must be unique",
            },
        },
    )
    attach_comparison("Expected vs actual duplicate recovery", expected, actual)

    assert actual == expected
