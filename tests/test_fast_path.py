"""Vectorized fast-path (#6): panel_fn detection, full parity, warmup/risk integration,
fail-closed runtime parity guard, PIT-forces-loop, and the t->t+1 shift.

The fast path must never become a second, silently-divergeable signal definition: it is used
ONLY behind a fail-closed parity guard against the canonical per-bar `_decision_weights` loop.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import algua.strategies.examples as examples
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import (
    BacktestError,
    _decision_weights,
    _decision_weights_fast_or_loop,
    simulate,
)
from algua.contracts.types import ExecutionContract
from algua.risk.limits import WEIGHT_TOL
from algua.strategies.base import LoadedStrategy, StrategyConfig
from algua.strategies.loader import load_strategy

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


def _bars_adj(symbols: list[str], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = SyntheticProvider(seed=seed).get_bars(symbols, START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return bars, adj.sort_index()


# --- 1. loader detection -------------------------------------------------------------------


def test_example_strategy_exposes_panel_fn() -> None:
    strat = load_strategy("cross_sectional_momentum")
    # cross_sectional_momentum DOES define a panel fn (added in this change); use a strategy
    # module without one for the None case.
    assert strat.panel_fn is not None  # sanity: the example exposes it


def test_loader_binds_panel_fn_when_present() -> None:
    mod_path = Path(examples.__path__[0]) / "_tmp_with_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_with_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1))\n"
        "def compute_weights(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
        "def compute_weights_panel(bars, params):\n"
        "    return pd.DataFrame()\n"
    )
    try:
        strat = load_strategy("_tmp_with_panel")
        assert strat.panel_fn is not None
        assert callable(strat.panel_fn)
    finally:
        mod_path.unlink()


def test_loader_panel_fn_none_when_module_omits_it() -> None:
    mod_path = Path(examples.__path__[0]) / "_tmp_no_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_no_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1))\n"
        "def compute_weights(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        strat = load_strategy("_tmp_no_panel")
        assert strat.panel_fn is None
    finally:
        mod_path.unlink()


def test_loader_rejects_non_callable_panel() -> None:
    mod_path = Path(examples.__path__[0]) / "_tmp_bad_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_bad_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1))\n"
        "def compute_weights(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
        "compute_weights_panel = 42\n"
    )
    try:
        with pytest.raises(Exception, match="compute_weights_panel"):
            load_strategy("_tmp_bad_panel")
    finally:
        mod_path.unlink()


# --- 2. full parity: example loop vs panel raw weights -------------------------------------


def test_cross_sectional_momentum_full_parity() -> None:
    strat = load_strategy("cross_sectional_momentum")
    syms = strat.universe
    bars, adj = _bars_adj(syms, seed=7)
    loop = _decision_weights(strat, bars, adj)
    panel_fn = strat.panel_fn
    assert panel_fn is not None
    panel = panel_fn(bars, strat.params)
    # Reindex panel like the fast path will, then compare to the loop output.
    aligned = panel.reindex(index=adj.index, columns=adj.columns).fillna(0.0)
    warmup = strat.execution.warmup_bars
    aligned.iloc[:warmup] = 0.0
    pd.testing.assert_frame_equal(loop, aligned, check_exact=False, atol=WEIGHT_TOL, rtol=0.0)


def test_fast_or_loop_matches_loop_for_example() -> None:
    strat = load_strategy("cross_sectional_momentum")
    bars, adj = _bars_adj(strat.universe, seed=11)
    loop = _decision_weights(strat, bars, adj)
    fast = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    pd.testing.assert_frame_equal(loop, fast, check_exact=False, atol=WEIGHT_TOL, rtol=0.0)


# --- 3. warmup + risk integration on the fast path -----------------------------------------


def test_fast_path_applies_warmup_flat_period() -> None:
    """A panel fn that always wants 100% AAA still gets the first warmup_bars rows zeroed."""

    def panel_fn(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(1.0, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="warm", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=5),
        params={},
    )
    strat = LoadedStrategy(
        config=cfg, fn=lambda v, p: pd.Series([1.0], index=["AAA"]), panel_fn=panel_fn
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    fast = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    assert (fast.iloc[:5]["AAA"] == 0.0).all()
    assert (fast.iloc[5:]["AAA"] == 1.0).all()


def test_fast_path_gross_breach_raises() -> None:
    def panel_fn(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(1.5, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="lev", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=0, max_gross_exposure=1.0),
        params={},
    )
    strat = LoadedStrategy(
        config=cfg, fn=lambda v, p: pd.Series(dtype="float64"), panel_fn=panel_fn
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="gross"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


def test_fast_path_long_only_breach_raises() -> None:
    def panel_fn(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(-1.0, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="short", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={},
    )
    strat = LoadedStrategy(
        config=cfg, fn=lambda v, p: pd.Series(dtype="float64"), panel_fn=panel_fn
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    with pytest.raises(BacktestError, match="long-only"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


# --- 4. fail-closed parity guard: a divergent panel fn raises (no silent fallback) ---------


def test_divergent_panel_raises_parity_error() -> None:
    """A panel fn whose answer disagrees with the canonical per-bar definition must RAISE,
    never silently fall back to either answer."""

    def good_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms)

    def bad_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        # Disagrees: puts everything in AAA instead of equal-weight.
        out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
        out["AAA"] = 1.0
        return out

    cfg = StrategyConfig(
        name="liar", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=good_loop, panel_fn=bad_panel)
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="parity"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


# --- 5. PIT mode forces the loop even when panel_fn exists ---------------------------------


def test_pit_mode_forces_loop_even_with_panel_fn() -> None:
    """With a universe_by_date map, the fast path must NOT run the panel fn; it falls back to
    the per-bar loop (which can reproduce as-of masking)."""
    calls = {"panel": 0}

    def panel_fn(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        calls["panel"] += 1
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.0, index=adj.index, columns=adj.columns)

    def loop_fn(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms)

    cfg = StrategyConfig(
        name="pit", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=loop_fn, panel_fn=panel_fn)
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    universe_by_date: dict[date, set[str]] = {ts.date(): {"AAA", "BBB"} for ts in adj.index}
    out = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=universe_by_date)
    loop = _decision_weights(strat, bars, adj, universe_by_date=universe_by_date)
    assert calls["panel"] == 0  # panel fn never invoked in PIT mode
    pd.testing.assert_frame_equal(out, loop, check_exact=False, atol=WEIGHT_TOL, rtol=0.0)


# --- 6. simulate still applies the t->t+1 shift after the fast path ------------------------


def test_simulate_applies_lag_after_fast_path() -> None:
    strat = load_strategy("cross_sectional_momentum")
    _pf, weights_eff = simulate(strat, SyntheticProvider(seed=7), START, END)
    bars, adj = _bars_adj(strat.universe, seed=7)
    raw = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    expected = raw.shift(strat.execution.decision_lag_bars).fillna(0.0)
    pd.testing.assert_frame_equal(weights_eff, expected, check_exact=False, atol=WEIGHT_TOL,
                                  rtol=0.0)
    # First row is flat (shifted-in NaN -> 0) — proves the lag was applied.
    assert (weights_eff.iloc[0] == 0.0).all()
