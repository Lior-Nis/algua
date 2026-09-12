"""Overlays end to end: the example loads, backtests, and passes the exhaustive parity gate."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.decision_path import verify_signal_panel_parity
from algua.backtest.engine import run
from algua.portfolio.overlays import regime_gate, trailing_stop
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy

START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


def test_example_loads_with_resolved_overlay_fns():
    strat = load_strategy("momentum_regime_stop")
    assert strat.overlay_fns == (regime_gate, trailing_stop)
    assert strat.config.feature_lookback == 252


def test_example_backtests_and_never_exceeds_the_unoverlaid_gross():
    provider = SyntheticProvider(seed=7)
    with_overlays = run(load_strategy("momentum_regime_stop"), provider, START, END)
    without = run(load_strategy("cross_sectional_momentum"), provider, START, END)
    gross_cap = without.metrics["avg_gross_exposure"] + 1e-9
    assert with_overlays.metrics["avg_gross_exposure"] <= gross_cap
    assert with_overlays.metrics["avg_gross_exposure"] > 0.0  # it did trade


def test_example_passes_the_exhaustive_parity_gate():
    assert verify_signal_panel_parity(
        load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END
    ) is None


def test_example_identity_differs_from_the_plain_momentum():
    assert config_hash(load_strategy("momentum_regime_stop")) != config_hash(
        load_strategy("cross_sectional_momentum")
    )


def test_run_is_deterministic_with_overlays():
    a = run(load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END)
    b = run(load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END)
    assert a.metrics == b.metrics
    assert isinstance(a.metrics["sharpe"], float) or pd.isna(a.metrics["sharpe"])
