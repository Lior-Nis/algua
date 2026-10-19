"""STRAT-01: obvious price paths produce explainable scores and target weights."""
from __future__ import annotations

import json
from pathlib import Path

import allure
import pandas as pd
import pytest

from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.strategies.loader import load_strategy
from tests.acceptance.evidence import attach_comparison, attach_frame, attach_json

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "reference_market_v1"


def _decision_bars() -> pd.DataFrame:
    """Literal, inspectable paths: +60%, +30%, flat, -30%, and one short-history name."""
    sessions = pd.bdate_range("2026-01-02", periods=61, tz="UTC", name="timestamp")
    paths = {
        "AAPL": [100.0 + i for i in range(61)],
        "NVDA": [100.0 + 0.5 * i for i in range(61)],
        "MSFT": [100.0] * 61,
        "AMZN": [100.0 - 0.5 * i for i in range(61)],
        "GOOGL": [None] * 30 + [80.0 + i for i in range(31)],
    }
    rows: list[dict[str, object]] = []
    for symbol, closes in paths.items():
        for timestamp, close in zip(sessions, closes, strict=True):
            if close is None:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000.0,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).set_index("timestamp")
    return validate_bars(frame[BAR_COLUMNS], timeframe="1d")


@pytest.mark.acceptance
@allure.parent_suite("Algua acceptance")
@allure.suite("Strategy decision")
@allure.title("STRAT-01 — obvious price paths produce explainable scores and weights")
def test_strat_01_obvious_paths_produce_approved_scores_and_weights() -> None:
    expected = json.loads(
        (FIXTURE_ROOT / "expected" / "strategy_decision.json").read_text(encoding="utf-8")
    )
    bars = _decision_bars()
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
