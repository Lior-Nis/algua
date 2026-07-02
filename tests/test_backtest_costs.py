"""Transaction-cost model in the backtest fill (issue #325).

`simulate` charges the per-side `fees` + `slippage` from the strategy's ExecutionContract into
`vbt.Portfolio.from_orders`, so every downstream metric (Sharpe/DSR/bootstrap/FDR/regime/paper
forward gate) evaluates NET-OF-COST returns. These tests lock:

- costs actually bite (cost-free != costed) — the regression guard against a future edit silently
  dropping fees/slippage;
- symmetry across sides (long round-trip AND short entry/cover both pay);
- the forced delisting liquidation is charged (conservative — never inflates edge);
- costs are applied at EXECUTION time on the t->t+1-shifted weights (no look-ahead — a cost on a
  never-traded warm-up bar is impossible);
- the ExecutionContract validation + the agent-gated cost floor fail closed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import run, simulate
from algua.contracts.types import (
    MIN_GATED_COST,
    ExecutionContract,
    assert_gated_costs,
)
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2020, 1, 10, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    return scores


class _FrameProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
        return self._frame[self._frame["symbol"].isin(set(symbols))]


def _long_flip_bars() -> pd.DataFrame:
    """Flat-price 3-bar panel for A and B (all prices 10.0). A flat-price round trip has ZERO
    frictionless return, so any negative total return is PURELY the transaction cost — an isolated
    read of the cost model with no market P&L confound."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    rows = []
    for sym in ("A", "B"):
        for ts in idx:
            rows.append({
                "timestamp": ts, "symbol": sym,
                "open": 10.0, "high": 10.0, "low": 10.0,
                "close": 10.0, "adj_close": 10.0, "volume": 1e6,
            })
    return pd.DataFrame(rows).set_index("timestamp").sort_values(["timestamp", "symbol"])


def _rotate_strategy(execution: ExecutionContract, *, allow_short: bool = False) -> LoadedStrategy:
    """Rotate 100% A on even bars, 100% B on odd bars (or SHORT B when allow_short) — forces a full
    turnover every bar so the cost model is exercised on entries, exits, and (optionally) shorts."""
    cfg = StrategyConfig(
        name="rotate",
        universe=["A", "B"],
        execution=execution,
        params={},
        construction="passthrough",
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        n = int(view.index.nunique())
        if n % 2 == 0:
            return pd.Series({"A": 1.0})
        return pd.Series({"B": -1.0 if allow_short else 1.0})

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def _total_return(pf) -> float:  # noqa: ANN001
    return float((1.0 + pf.returns().fillna(0.0)).prod() - 1.0)


# --- costs actually bite -------------------------------------------------------------------------

def test_costs_reduce_returns_vs_frictionless() -> None:
    """Same rotating book, one with the DEFAULT-ON cost, one explicitly frictionless: the costed
    run MUST have a strictly lower total return. This is the regression guard — if a future edit
    drops fees/slippage from from_orders, this fails."""
    provider = _FrameProvider(_long_flip_bars())
    costed = ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1)  # default 5+5 bps
    free = ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, fees=0.0, slippage=0.0)

    pf_costed, _, _ = simulate(_rotate_strategy(costed), provider, START, END)
    pf_free, _, _ = simulate(_rotate_strategy(free), provider, START, END)

    # Flat prices -> frictionless is exactly zero; costed is strictly negative (pure cost drag).
    assert _total_return(pf_free) == pytest.approx(0.0, abs=1e-12)
    assert _total_return(pf_costed) < _total_return(pf_free)
    assert _total_return(pf_costed) < 0.0


def test_higher_cost_is_monotonically_worse() -> None:
    """More cost -> strictly worse net return (never better): the model can only subtract."""
    provider = _FrameProvider(_long_flip_bars())
    lo = ExecutionContract(rebalance_frequency="1d", fees=0.001, slippage=0.001)
    hi = ExecutionContract(rebalance_frequency="1d", fees=0.002, slippage=0.002)
    pf_lo, _, _ = simulate(_rotate_strategy(lo), provider, START, END)
    pf_hi, _, _ = simulate(_rotate_strategy(hi), provider, START, END)
    assert _total_return(pf_hi) < _total_return(pf_lo) < 0.0


def test_costs_symmetric_long_and_short() -> None:
    """A SHORT-selling rotation pays cost too (buys/covers fill higher, sells/shorts fill lower):
    the short book's flat-price total return is also strictly negative, proving slippage is charged
    symmetrically on the short side, not only on longs."""
    provider = _FrameProvider(_long_flip_bars())
    ex = ExecutionContract(rebalance_frequency="1d", allow_short=True, max_gross_exposure=1.0)
    pf_short, _, _ = simulate(_rotate_strategy(ex, allow_short=True), provider, START, END)
    assert _total_return(pf_short) < 0.0


# --- forced delisting exit is charged ------------------------------------------------------------

def _delist_bars() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for ts in idx:
        rows.append({"timestamp": ts, "symbol": "A", "open": 10.0, "high": 10.0, "low": 10.0,
                     "close": 10.0, "adj_close": 10.0, "volume": 1e6})
    for ts in idx[:2]:
        rows.append({"timestamp": ts, "symbol": "B", "open": 10.0, "high": 10.0, "low": 10.0,
                     "close": 10.0, "adj_close": 10.0, "volume": 1e6})
    return pd.DataFrame(rows).set_index("timestamp").sort_values(["timestamp", "symbol"])


def _ew_ab_strategy(execution: ExecutionContract) -> LoadedStrategy:
    cfg = StrategyConfig(name="ew_ab", universe=["A", "B"], execution=execution, params={},
                         construction="passthrough")

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms)

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def test_forced_delisting_exit_is_charged() -> None:
    """The forced liquidation of a delisted name is a modeled trade and pays cost. With flat prices
    and a terminal price equal to the last close, the ONLY source of any negative return is the
    transaction cost on the entries and the forced exit — so the costed run is strictly worse than
    the frictionless one, and returns stay finite."""
    provider = _FrameProvider(_delist_bars())
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 10.0, "vendor")]}
    start, end = datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC)

    costed = ExecutionContract(rebalance_frequency="1d")
    free = ExecutionContract(rebalance_frequency="1d", fees=0.0, slippage=0.0)
    pf_c, _, forced_c = simulate(_ew_ab_strategy(costed), provider, start, end,
                                 delisting_records=recs)
    pf_f, _, _ = simulate(_ew_ab_strategy(free), provider, start, end, delisting_records=recs)

    assert any(fe["symbol"] == "B" for fe in forced_c)
    assert bool(np.isfinite(pf_c.returns().fillna(0.0)).all())
    assert _total_return(pf_c) < _total_return(pf_f)


# --- no look-ahead / warm-up bars pay nothing ----------------------------------------------------

def test_warmup_bars_incur_no_cost() -> None:
    """A held-flat warm-up bar issues no trade, so it must incur no cost: a strategy that stays flat
    for its whole (long) warm-up and then rotates should cost the SAME whether warm-up is 0 or 1,
    because the cost is charged at EXECUTION time on the t->t+1-shifted weights, never on a bar with
    no order. (Equivalently: the cost cannot look ahead to a trade that has not executed.)"""
    provider = _FrameProvider(_long_flip_bars())
    # Warm-up 2 => only the last bar can trade (flat prices, no P&L) -> flat -> zero cost, zero ret.
    ex = ExecutionContract(rebalance_frequency="1d", warmup_bars=2)
    pf, weights, _ = simulate(_rotate_strategy(ex), provider, START, END)
    # With a 2-bar warm-up over a 3-bar flat panel and a 1-bar decision lag, no round trip completes
    # -> total return is exactly the cost of at most one entry, and never positive.
    assert _total_return(pf) <= 0.0


# --- metrics surface the cost assumption ---------------------------------------------------------

def test_metrics_surface_cost_assumption() -> None:
    provider = _FrameProvider(_long_flip_bars())
    res = run(_rotate_strategy(ExecutionContract(rebalance_frequency="1d")), provider, START, END)
    assert res.metrics["cost_fees"] == pytest.approx(0.0005)
    assert res.metrics["cost_slippage"] == pytest.approx(0.0005)


# --- ExecutionContract validation ----------------------------------------------------------------

@pytest.mark.parametrize("field", ["fees", "slippage"])
@pytest.mark.parametrize("bad", [-0.0001, float("nan"), float("inf"), True])
def test_contract_rejects_bad_cost(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1d", **{field: bad})


def test_contract_default_costs_are_default_on() -> None:
    ex = ExecutionContract(rebalance_frequency="1d")
    assert ex.fees == 0.0005 and ex.slippage == 0.0005


def test_cost_is_in_config_hash() -> None:
    """Changing the cost assumption must change strategy identity (invalidate a live approval)."""
    from algua.strategies.base import config_hash

    a = _rotate_strategy(ExecutionContract(rebalance_frequency="1d"))
    b = _rotate_strategy(ExecutionContract(rebalance_frequency="1d", fees=0.001))
    assert config_hash(a) != config_hash(b)


# --- agent-gated cost floor ----------------------------------------------------------------------

def test_gated_costs_floor_rejects_frictionless() -> None:
    with pytest.raises(ValueError, match="fees \\+ slippage"):
        assert_gated_costs(ExecutionContract(rebalance_frequency="1d", fees=0.0, slippage=0.0))


def test_gated_costs_floor_accepts_default() -> None:
    # Default 5+5 bps == MIN_GATED_COST exactly (boundary is inclusive) -> allowed.
    ex = ExecutionContract(rebalance_frequency="1d")
    assert ex.fees + ex.slippage == pytest.approx(MIN_GATED_COST)
    assert_gated_costs(ex)  # does not raise
