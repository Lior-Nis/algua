"""Vectorized fast-path (#6): signal_panel detection, full parity, warmup/risk integration,
fail-closed runtime parity guard, PIT-forces-loop, and the t->t+1 shift.

The fast path must never become a second, silently-divergeable signal definition: it is used
ONLY behind a fail-closed WEIGHT-level parity guard against the canonical per-bar
`_decision_weights` loop (construct(signal(view), view)).
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import algua.strategies.momentum as momentum_pkg
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import (
    BacktestError,
    _assert_parity,
    _decision_weights,
    _decision_weights_fast_or_loop,
    _fast_weights,
    _parity_sample_positions,
    simulate,
    verify_signal_panel_parity,
)
from algua.contracts.types import ExecutionContract
from algua.risk.limits import WEIGHT_TOL
from algua.strategies.base import LoadedStrategy, StrategyConfig
from algua.strategies.loader import load_strategy

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identity construction: scores ARE the desired raw weights. Lets these fast-path tests drive
    exact panel weight vectors at the risk rails (the signal/panel emit weights directly)."""
    return scores


def _bars_adj(symbols: list[str], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = SyntheticProvider(seed=seed).get_bars(symbols, START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return bars, adj.sort_index()


# --- 1. loader detection -------------------------------------------------------------------


def test_example_strategy_exposes_signal_panel_fn() -> None:
    strat = load_strategy("cross_sectional_momentum")
    # cross_sectional_momentum DOES define a signal_panel fn; use a strategy module without one
    # for the None case.
    assert strat.signal_panel_fn is not None  # sanity: the example exposes it


def test_loader_binds_signal_panel_fn_when_present() -> None:
    mod_path = Path(momentum_pkg.__path__[0]) / "tmp_with_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_with_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
        "def signal_panel(bars, params):\n"
        "    return pd.DataFrame()\n"
    )
    try:
        strat = load_strategy("tmp_with_panel")
        assert strat.signal_panel_fn is not None
        assert callable(strat.signal_panel_fn)
    finally:
        mod_path.unlink()


def test_loader_signal_panel_fn_none_when_module_omits_it() -> None:
    mod_path = Path(momentum_pkg.__path__[0]) / "tmp_no_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_no_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        strat = load_strategy("tmp_no_panel")
        assert strat.signal_panel_fn is None
    finally:
        mod_path.unlink()


def test_loader_rejects_non_callable_signal_panel() -> None:
    mod_path = Path(momentum_pkg.__path__[0]) / "tmp_bad_panel.py"
    mod_path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_bad_panel', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
        "signal_panel = 42\n"
    )
    try:
        with pytest.raises(Exception, match="signal_panel"):
            load_strategy("tmp_bad_panel")
    finally:
        mod_path.unlink()


# --- 2. full parity: example loop vs fast path ---------------------------------------------


def test_cross_sectional_momentum_full_parity() -> None:
    strat = load_strategy("cross_sectional_momentum")
    syms = strat.universe
    bars, adj = _bars_adj(syms, seed=7)
    loop = _decision_weights(strat, bars, adj)
    fast = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    pd.testing.assert_frame_equal(loop, fast, check_exact=False, atol=WEIGHT_TOL, rtol=0.0)


def test_fast_or_loop_matches_loop_for_example() -> None:
    strat = load_strategy("cross_sectional_momentum")
    bars, adj = _bars_adj(strat.universe, seed=11)
    loop = _decision_weights(strat, bars, adj)
    fast = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    pd.testing.assert_frame_equal(loop, fast, check_exact=False, atol=WEIGHT_TOL, rtol=0.0)


# --- 3. warmup + risk integration on the fast path -----------------------------------------


def test_fast_path_applies_warmup_flat_period() -> None:
    """A panel that always wants 100% AAA still gets the first warmup_bars rows zeroed."""

    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(1.0, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="warm", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=5),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series([1.0], index=["AAA"]),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    fast = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    assert (fast.iloc[:5]["AAA"] == 0.0).all()
    assert (fast.iloc[5:]["AAA"] == 1.0).all()


def test_fast_path_gross_breach_raises() -> None:
    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        # 0.6 per name is within the default per-symbol cap (1.0); the sum (1.2) breaches gross,
        # isolating the gross-exposure rail rather than tripping the concentration cap first.
        return pd.DataFrame(0.6, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="lev", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=0, max_gross_exposure=1.0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.6, "BBB": 0.6}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="gross"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


def test_fast_path_long_only_breach_raises() -> None:
    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(-1.0, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="short", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": -1.0}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    with pytest.raises(BacktestError, match="long-only"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


# --- 4. fail-closed parity guard: a divergent panel raises (no silent fallback) ------------


def test_divergent_panel_raises_parity_error() -> None:
    """A signal_panel whose answer disagrees with the canonical per-bar definition must RAISE,
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
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=good_loop, signal_panel_fn=bad_panel, construct_fn=_passthrough
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="parity"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


# --- 5. PIT mode forces the loop even when signal_panel_fn exists --------------------------


def test_pit_mode_forces_loop_even_with_signal_panel_fn() -> None:
    """With a universe_by_date map, the fast path must NOT run the panel fn; it falls back to
    the per-bar loop (which can reproduce as-of masking)."""
    calls = {"panel": 0}

    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        calls["panel"] += 1
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.0, index=adj.index, columns=adj.columns)

    def loop_fn(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms)

    cfg = StrategyConfig(
        name="pit", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=loop_fn, signal_panel_fn=signal_panel, construct_fn=_passthrough
    )
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


# --- new rails (#135) enforced in the vectorized fast-path, not just the loop ----------------


def test_fast_path_concentration_cap_breach_raises() -> None:
    """The single-name cap is enforced in the fast-path, identically to the loop: a panel putting
    0.9 in a name when the cap is 0.5 must raise (cap trips before gross given the bundle order)."""
    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.9, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="conc", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=0, max_weight_per_symbol=0.5),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.9, "BBB": 0.9}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="max_weight_per_symbol"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


def test_fast_path_inf_weight_raises_non_finite() -> None:
    """A non-finite VALUE survives the panel's reindex (dropna only drops NaN), so the fail-closed
    finite guard in the risk walls must catch an inf in the fast-path rather than let it through."""
    import numpy as np

    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(np.inf, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="inf", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": np.inf}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    with pytest.raises(BacktestError, match="non-finite"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)


def test_fast_path_omitted_cell_nan_stays_flat() -> None:
    """An omitted score (NaN) is DROPPED before construction (missing score != 0), so it must NOT
    breach — only real values do. BBB omitted -> flat."""
    import numpy as np

    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        df = pd.DataFrame(0.5, index=adj.index, columns=adj.columns)
        df["BBB"] = np.nan  # BBB omitted -> flat by convention, not a breach
        return df

    cfg = StrategyConfig(
        name="sparse", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    # The per-bar fn must agree with the panel for the parity guard: AAA at 0.5, BBB omitted (flat).
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.5}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    # Runs clean: AAA at 0.5 (within cap + gross), BBB held flat. No raise is the assertion.
    weights = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    assert (weights["BBB"] == 0.0).all()


def test_fast_path_allow_short_permits_negative_panel() -> None:
    """With allow_short=True a negative panel weight passes the fast-path short policy (the cap
    still bounds |weight|); the default-False rejection is pinned by the long-only test."""
    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(-0.5, index=adj.index, columns=adj.columns)

    cfg = StrategyConfig(
        name="shortable", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=0, allow_short=True),
        params={}, construction="passthrough",
    )
    # The per-bar fn must agree with the panel for the parity guard: AAA short at -0.5.
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": -0.5}),
        signal_panel_fn=signal_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA"], seed=2)
    weights = _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
    assert (weights["AAA"] == -0.5).all()


def test_fast_weights_skips_bounded_guard() -> None:
    """`_fast_weights` returns the fast-path matrix WITHOUT the bounded parity guard, so a panel
    that diverges only where the bounded sample does not look does NOT raise here — the guard is
    `_decision_weights_fast`'s job, decoupled so the exhaustive verifier owns its own comparison."""
    def good_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    n = len(adj.index)
    sample = set(_parity_sample_positions(0, n))
    target = next(i for i in range(0, n) if i not in sample)  # an UNSAMPLED evaluated bar
    target_ts = adj.index[target]

    def sneaky_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        out = pd.DataFrame(0.5, index=a.index, columns=a.columns)
        out.loc[target_ts, "AAA"] = 1.0  # diverge from equal-weight only here
        out.loc[target_ts, "BBB"] = 0.0
        return out

    cfg = StrategyConfig(
        name="sneaky", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=good_loop, signal_panel_fn=sneaky_panel, construct_fn=_passthrough
    )
    # `_fast_weights` does NOT raise (no bounded guard); it returns the divergent matrix as-is.
    fast = _fast_weights(strat, bars, adj)
    assert fast.loc[target_ts, "AAA"] == 1.0
    # The bounded guard at the unsampled bar also passes (documents the gap the verifier closes).
    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")
    _assert_parity(strat, bars_sorted, end_pos, fast, 0)  # no raise — sample misses target


# --- exhaustive parity gate (#178): every-bar panel-vs-loop check for promotion ---------------


def test_verifier_catches_divergence_on_unsampled_bar() -> None:
    """The crux: a panel diverging on a bar the bounded sample never inspects passes the runtime
    guard but MUST be caught by the exhaustive verifier."""
    def good_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    _, adj = _bars_adj(["AAA", "BBB"], seed=2)
    n = len(adj.index)
    sample = set(_parity_sample_positions(0, n))
    target_ts = adj.index[next(i for i in range(0, n) if i not in sample)]

    def sneaky_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        out = pd.DataFrame(0.5, index=a.index, columns=a.columns)
        out.loc[target_ts, "AAA"] = 1.0
        out.loc[target_ts, "BBB"] = 0.0
        return out

    cfg = StrategyConfig(
        name="sneaky", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=good_loop, signal_panel_fn=sneaky_panel, construct_fn=_passthrough
    )
    with pytest.raises(BacktestError, match="parity"):
        verify_signal_panel_parity(strat, SyntheticProvider(seed=2), START, END)


def test_verifier_passes_for_faithful_panel() -> None:
    """A panel equal to its per-bar twin everywhere passes (returns None, no raise)."""
    def equal_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    def faithful_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.5, index=a.index, columns=a.columns)

    cfg = StrategyConfig(
        name="faithful", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=equal_loop, signal_panel_fn=faithful_panel, construct_fn=_passthrough
    )
    assert verify_signal_panel_parity(strat, SyntheticProvider(seed=2), START, END) is None


def test_verifier_noop_when_no_signal_panel_fn() -> None:
    """No signal_panel_fn -> nothing to verify; returns WITHOUT touching the provider."""
    class _BoomProvider:
        def get_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            raise AssertionError("provider must not be called when there is no signal_panel_fn")

    cfg = StrategyConfig(
        name="nopanel", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 1.0}),
        signal_panel_fn=None, construct_fn=_passthrough,
    )
    assert verify_signal_panel_parity(strat, _BoomProvider(), START, END) is None


def test_verifier_raises_on_empty_provider() -> None:
    """A provider with no bars for a panel strategy fails the gate (mirrors simulate's guard)."""
    class _EmptyProvider:
        def get_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            return pd.DataFrame()

    def faithful_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.5, index=a.index, columns=a.columns)

    cfg = StrategyConfig(
        name="empty", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 1.0}),
        signal_panel_fn=faithful_panel, construct_fn=_passthrough,
    )
    with pytest.raises(BacktestError, match="no bars"):
        verify_signal_panel_parity(strat, _EmptyProvider(), START, END)
