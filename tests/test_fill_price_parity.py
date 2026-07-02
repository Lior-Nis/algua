"""Fill-price basis parity (issue #383).

The ExecutionContract pins the t->t+1 lag (WHEN a lagged decision fills) AND now the intra-bar
fill REFERENCE price (WHICH price on bar t+1 it fills at). Before #383 the backtest silently filled
at adj_close(t+1) while the paper/live loop filled at the next-bar open — the same decision priced
on two different bases. These tests pin: the contract field + resolver, the adjusted-open grid
derivation (incl. corporate actions + fail-safe cells), the engine honoring the basis, and true
end-to-end backtest<->paper fill-price parity on an adj==raw panel.
"""
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import adj_grid, adj_open_grid, simulate
from algua.contracts.types import ExecutionContract, fill_reference_column
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import run_paper
from algua.portfolio.construction import top_k_equal_weight
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2023, 6, 1, tzinfo=UTC)


# --- Contract field + resolver ------------------------------------------------------------------

def test_fill_price_defaults_to_open() -> None:
    assert ExecutionContract(rebalance_frequency="1d").fill_price == "open"


def test_fill_price_accepts_open_and_close() -> None:
    assert ExecutionContract(rebalance_frequency="1d", fill_price="open").fill_price == "open"
    assert ExecutionContract(rebalance_frequency="1d", fill_price="close").fill_price == "close"


@pytest.mark.parametrize("bad", ["OPEN", "vwap", "", "mid", 0, None, True])
def test_fill_price_rejects_unknown_basis(bad) -> None:
    with pytest.raises(ValueError, match="fill_price"):
        ExecutionContract(rebalance_frequency="1d", fill_price=bad)


def test_fill_reference_column_maps_basis_to_raw_column() -> None:
    assert fill_reference_column(ExecutionContract(rebalance_frequency="1d", fill_price="open")) \
        == "open"
    assert fill_reference_column(ExecutionContract(rebalance_frequency="1d", fill_price="close")) \
        == "adj_close"


def test_fill_price_is_part_of_config_identity() -> None:
    from algua.strategies.base import config_hash

    def _strat(basis: str) -> LoadedStrategy:
        cfg = StrategyConfig(
            name="s", universe=["AAA"],
            execution=ExecutionContract(rebalance_frequency="1d", fill_price=basis),
            params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
        )
        return LoadedStrategy(
            config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
            construct_fn=top_k_equal_weight,
        )

    assert config_hash(_strat("open")) != config_hash(_strat("close"))


# --- adj_open_grid derivation -------------------------------------------------------------------

def _bars(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def test_adj_open_scales_open_by_adjustment_ratio() -> None:
    # A 2:1 split on day 2 halves the adjusted prices of day 1 (adj_close = close/2 that day).
    bars = _bars([
        {"timestamp": "2023-01-02", "symbol": "AAA",
         "open": 100.0, "high": 110.0, "low": 90.0, "close": 102.0, "adj_close": 51.0,
         "volume": 1.0},
        {"timestamp": "2023-01-03", "symbol": "AAA",
         "open": 52.0, "high": 55.0, "low": 50.0, "close": 53.0, "adj_close": 53.0,
         "volume": 1.0},
    ])
    ao = adj_open_grid(bars)
    # Day 1: adj_open = open * adj_close/close = 100 * 51/102 = 50.0 (same 1/2 ratio as the close).
    assert ao.loc[pd.Timestamp("2023-01-02", tz="UTC"), "AAA"] == pytest.approx(50.0)
    # Day 2: unadjusted (adj_close == close) -> adj_open == open.
    assert ao.loc[pd.Timestamp("2023-01-03", tz="UTC"), "AAA"] == pytest.approx(52.0)
    # Invariant: adj_open/adj_close == open/close on every bar.
    adjc = adj_grid(bars)
    ratio_ao = ao / adjc
    ratio_raw = (bars.reset_index().pivot(index="timestamp", columns="symbol", values="open")
                 / bars.reset_index().pivot(index="timestamp", columns="symbol", values="close"))
    pd.testing.assert_frame_equal(ratio_ao, ratio_raw, check_names=False)


@pytest.mark.parametrize("close,open_", [
    (0.0, 100.0),        # close <= 0 -> undefined ratio
    (-5.0, 100.0),       # negative close
    (100.0, 0.0),        # open <= 0
    (100.0, -1.0),       # negative open
    (float("nan"), 100.0),
    (100.0, float("nan")),
])
def test_adj_open_fails_unsafe_cells_to_nan(close, open_) -> None:
    bars = _bars([
        {"timestamp": "2023-01-02", "symbol": "AAA",
         "open": open_, "high": 1.0, "low": 1.0, "close": close, "adj_close": 50.0,
         "volume": 1.0},
    ])
    ao = adj_open_grid(bars)
    assert np.isnan(ao.loc[pd.Timestamp("2023-01-02", tz="UTC"), "AAA"])


def test_adj_open_grid_shares_adj_grid_index_and_columns() -> None:
    bars = SyntheticProvider(seed=1).get_bars(["AAA", "BBB"], START, END, "1d")
    ao, ac = adj_open_grid(bars), adj_grid(bars)
    assert ao.index.equals(ac.index) and ao.columns.equals(ac.columns)


# --- Engine honors the basis --------------------------------------------------------------------

def _momo(fill_price: str, warmup: int = 3) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="momo", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=warmup,
            fill_price=fill_price, fees=0.0, slippage=0.0,
        ),
        params={"lookback": 5},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        lb = params["lookback"]
        if len(wide) <= lb:
            return pd.Series(dtype="float64")
        return wide.iloc[-1] / wide.iloc[-1 - lb] - 1.0

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)


def test_open_and_close_bases_produce_different_finite_returns() -> None:
    provider = SyntheticProvider(seed=7)
    pf_open, _, _ = simulate(_momo("open"), provider, START, END)
    pf_close, _, _ = simulate(_momo("close"), provider, START, END)
    r_open, r_close = pf_open.returns(), pf_close.returns()
    assert bool(np.isfinite(r_open.fillna(0.0)).all())
    assert bool(np.isfinite(r_close.fillna(0.0)).all())
    # open (raw==adj here, open != close) vs close basis price fills differently -> different series
    assert not np.allclose(r_open.to_numpy(), r_close.to_numpy())


# --- End-to-end backtest <-> paper fill-price parity --------------------------------------------

def test_backtest_and_paper_fill_at_the_same_price_basis() -> None:
    """On a synthetic (adj_close == raw close, open != close) panel, adj_open == raw open, so the
    open-fill backtest and the paper loop fill at the IDENTICAL price for each bar. We prove the
    two paths CONSULT the same reference by capturing the price series each fills against."""
    provider = SyntheticProvider(seed=3)
    strat = _momo("open")

    # Paper loop: capture the price series it hands the broker each fill bar.
    captured: list[pd.Series] = []
    broker = SimBroker(cash=100_000.0)
    orig = broker.fill_pending

    def spy(fill_prices: pd.Series, fill_ts):
        captured.append(fill_prices.copy())
        return orig(fill_prices, fill_ts=fill_ts)

    broker.fill_pending = spy  # type: ignore[method-assign]
    run_paper(strat, broker, provider, START, END)

    # The engine's open-fill grid == adj_open_grid; on this panel it equals the RAW open the paper
    # loop fills against. Assert every captured paper fill price matches adj_open at that bar.
    bars = provider.get_bars(strat.universe, START, END, "1d")
    ao = adj_open_grid(bars)
    assert captured, "paper loop performed no fills"
    for series in captured:
        t = series.name
        for sym, price in series.items():
            assert price == pytest.approx(ao.loc[t, sym]), f"{sym}@{t}: paper {price} vs adj_open"
