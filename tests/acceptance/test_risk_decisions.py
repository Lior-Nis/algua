"""Human-reviewable target-weight risk decisions and order intents."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import allure
import pandas as pd
import pytest

from algua.audit import log as audit_log
from algua.contracts.types import ExecutionContract, OrderIntent
from algua.live.planner import PlannerInput, plan
from algua.registry.db import connect, migrate
from algua.risk import kill_switch
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig
from tests.acceptance.evidence import attach_comparison, attach_json

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"
DECISION_TS = datetime(2026, 3, 27, tzinfo=UTC)


def _identity(
    scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]
) -> pd.Series:
    return scores


def _fixed_weight_strategy(
    weights: dict[str, float], *, max_weight_per_symbol: float = 0.5
) -> LoadedStrategy:
    config = StrategyConfig(
        name="acceptance_risk_policy",
        universe=["AAPL", "GOOGL", "MSFT", "NVDA"],
        execution=ExecutionContract(
            rebalance_frequency="1d",
            decision_lag_bars=1,
            max_gross_exposure=1.0,
            max_weight_per_symbol=max_weight_per_symbol,
        ),
        params={},
        construction="acceptance_identity",
    )
    return LoadedStrategy(
        config=config,
        signal_fn=lambda _view, _params: pd.Series(weights, dtype="float64"),
        construct_fn=_identity,
    )


def _intent_record(intent: OrderIntent) -> dict[str, object]:
    return {
        "symbol": intent.symbol,
        "side": intent.side.value,
        "target_weight": intent.target_weight,
        "decision_session": pd.Timestamp(intent.decision_ts).isoformat(),
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Risk decisions")
@allure.title("RISK-01 — an approved allocation becomes explicit order intents")
def test_risk_01_approved_allocation_becomes_explicit_order_intents() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "risk_safe_intents.json").read_text(encoding="utf-8")
    )
    strategy = _fixed_weight_strategy({"AAPL": 0.4, "MSFT": 0.1, "NVDA": 0.3})
    current_weights = {"AAPL": 0.2, "GOOGL": 0.05, "MSFT": 0.2}

    result = plan(
        strategy,
        PlannerInput(
            view=pd.DataFrame(),
            current_weights=current_weights,
            decision_ts=DECISION_TS,
        ),
    )
    actual = {
        "risk_limits": {
            "max_gross_exposure": strategy.execution.max_gross_exposure,
            "max_weight_per_symbol": strategy.execution.max_weight_per_symbol,
            "allow_short": strategy.execution.allow_short,
        },
        "current_weights": current_weights,
        "approved_target_weights": result.weights.to_dict(),
        "approved_target_gross": float(result.weights.abs().sum()),
        "order_intents": [_intent_record(intent) for intent in result.intents],
    }

    attach_json("Given — proposed and current weights", {
        "proposed": {"AAPL": 0.4, "MSFT": 0.1, "NVDA": 0.3},
        "current": current_weights,
    })
    attach_comparison("Expected vs actual safe intents", expected, actual)

    assert actual == expected


def _rejection_record(
    proposed: dict[str, float], *, max_weight_per_symbol: float
) -> dict[str, object]:
    strategy = _fixed_weight_strategy(
        proposed, max_weight_per_symbol=max_weight_per_symbol
    )
    try:
        result = plan(
            strategy,
            PlannerInput(view=pd.DataFrame(), current_weights={}, decision_ts=DECISION_TS),
        )
    except RiskBreach as exc:
        return {
            "status": "rejected",
            "breach_kind": exc.kind,
            "detail": exc.detail,
            "order_intents": [],
        }
    return {
        "status": "accepted",
        "breach_kind": None,
        "detail": None,
        "order_intents": [_intent_record(intent) for intent in result.intents],
    }


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Risk decisions")
@allure.title("RISK-02 — concentration and gross breaches fail closed")
def test_risk_02_concentration_and_gross_breaches_fail_closed() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "risk_rejections.json").read_text(encoding="utf-8")
    )
    proposals = {
        "concentration": {
            "weights": {"AAPL": 0.6, "MSFT": 0.2},
            "max_weight_per_symbol": 0.5,
        },
        "gross_exposure": {
            "weights": {"AAPL": 0.45, "MSFT": 0.45, "NVDA": 0.2},
            "max_weight_per_symbol": 0.5,
        },
    }

    actual = {
        name: _rejection_record(
            case["weights"],
            max_weight_per_symbol=case["max_weight_per_symbol"],
        )
        for name, case in proposals.items()
    }

    attach_json("Given — unsafe allocation proposals", proposals)
    attach_comparison("Expected vs actual risk rejections", expected, actual)

    assert actual == expected


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Risk decisions")
@allure.title("RISK-03 — a risk breach trips a persistent audited kill switch")
def test_risk_03_breach_trips_persistent_audited_kill_switch(tmp_path: Path) -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "risk_kill_switch.json").read_text(encoding="utf-8")
    )
    strategy = _fixed_weight_strategy({"AAPL": 0.6, "MSFT": 0.2})
    conn = connect(tmp_path / "registry.db")
    migrate(conn)

    with pytest.raises(RiskBreach) as raised:
        plan(
            strategy,
            PlannerInput(view=pd.DataFrame(), current_weights={}, decision_ts=DECISION_TS),
        )
    trip_for_breach(conn, strategy.name, raised.value)

    switch = kill_switch.get(conn, strategy.name)
    assert switch is not None
    audit_rows = audit_log.read(conn, strategy=strategy.name)
    trip_row = next(row for row in audit_rows if row["action"] == "kill_switch_trip")
    actual = {
        "breach": {
            "kind": raised.value.kind,
            "detail": raised.value.detail,
        },
        "kill_switch": {
            "strategy": switch["strategy"],
            "actor": switch["actor"],
            "reason": switch["reason"],
            "is_tripped": kill_switch.is_tripped(conn, strategy.name),
        },
        "audit": {
            "strategy": trip_row["strategy"],
            "actor": trip_row["actor"],
            "action": trip_row["action"],
            "reason": trip_row["reason"],
        },
    }

    attach_comparison("Expected vs actual kill-switch response", expected, actual)

    assert actual == expected
