"""STRAT-01: obvious price paths produce explainable scores and target weights."""
from __future__ import annotations

import json
from pathlib import Path

import allure
import pytest

from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_frame, attach_json
from tests.acceptance.reference_market import decision_bars

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"

@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Strategy decision")
@allure.title("STRAT-01 — obvious price paths produce explainable scores and weights")
def test_strat_01_obvious_paths_produce_approved_scores_and_weights() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "strategy_decision.json").read_text(encoding="utf-8")
    )
    bars = decision_bars()
    strategy = load_strategy("cross_sectional_momentum", reload=True)

    scores = strategy.signal(bars)
    weights = strategy.target_weights(bars)
    actual = {
        "scores": scores.sort_index().to_dict(),
        "target_weights": weights.sort_index().to_dict(),
        "target_gross": float(weights.abs().sum()),
        "excluded_symbols": {"GOOGL": "insufficient 60-bar history"},
    }

    attach_frame("Given — final 65 bars", bars.reset_index().tail(65))
    attach_json("Given — strategy configuration", strategy.config.model_dump(mode="json"))
    attach_comparison("Expected vs actual decision", expected, actual)

    assert actual["excluded_symbols"] == expected["excluded_symbols"]
    assert actual["scores"] == pytest.approx(expected["scores"])
    assert actual["target_weights"] == pytest.approx(expected["target_weights"])
    assert actual["target_gross"] == pytest.approx(expected["target_gross"])
