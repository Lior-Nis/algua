"""Provenance fields forced_exits + delisting_snapshot in BacktestResult — Task 10 of #212."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import run
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig
from tests.test_delisting_simulate import (
    _bars,
    _equal_weight_ab_strategy,
    _FakeProvider,
)


def test_forced_exits_in_result_provenance() -> None:
    strat = _equal_weight_ab_strategy()
    provider = _FakeProvider(_bars())
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    result = run(
        strat, provider, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
        delisting_records=recs, delisting_snapshot="vendor-2026",
    )
    d = result.to_dict()
    assert d["delisting_snapshot"] == "vendor-2026"
    assert d["forced_exits"] and d["forced_exits"][0]["symbol"] == "B"
    assert d["forced_exits"][0]["terminal_price"] == 5.0


def test_no_delisting_defaults() -> None:
    """When no delisting is involved, both fields are falsy/None."""
    # Single-symbol to avoid the residual-position error on missing B.
    cfg = StrategyConfig(
        name="ew_a",
        universe=["A"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict) -> pd.Series:
        return scores

    strat = LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)
    all_bars = _bars()
    a_only = all_bars[all_bars["symbol"] == "A"]
    provider = _FakeProvider(a_only)

    result = run(
        strat, provider, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
    )
    d = result.to_dict()
    assert d["delisting_snapshot"] is None
    assert d["forced_exits"] == []
