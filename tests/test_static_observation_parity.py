"""Issue #208: static-mode observation parity. A misbehaving provider that returns an UNDECLARED
symbol (one not in strategy.universe) must never have that symbol's data reach the strategy's view,
panel, weights/grid, or fundamentals/news sidecars. Mirror of #179, which closed the out-of-universe
WEIGHT path; this closes the OBSERVATION path."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import run
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identity construction: scores ARE the desired raw weights."""
    return scores


class _ExtraSymbolProvider:
    """A misbehaving provider: returns bars for the requested symbols PLUS an undeclared `extra`."""

    def __init__(self, extra: str = "ZZZ", seed: int = 0) -> None:
        self.extra = extra
        self.seed = seed

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
        requested = list(symbols)
        return SyntheticProvider(seed=self.seed).get_bars(
            requested + [self.extra], start, end, timeframe
        )


class _ViewRecorder:
    """A 2-arg signal that records every symbol it is shown and returns FLAT weights (so the
    observation check is isolated from the #179 weight-rejection path)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        self.seen.update(view["symbol"].unique())
        return pd.Series(dtype="float64")


def _loop_strategy(signal) -> LoadedStrategy:  # noqa: ANN001
    cfg = StrategyConfig(
        name="obs_loop", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def test_loop_view_excludes_undeclared_symbol() -> None:
    recorder = _ViewRecorder()
    strat = _loop_strategy(recorder)
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert recorder.seen, "signal was never invoked"
    assert "ZZZ" not in recorder.seen
    assert recorder.seen <= {"AAA", "BBB"}


class _PanelRecorder:
    """A signal_panel that records the symbols it is handed and returns a FLAT (all-zero) scores
    matrix. Paired with a flat 2-arg signal so the fast-path parity guard (which compares the panel
    against the per-bar loop) holds."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        self.seen.update(bars["symbol"].unique())
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.0, index=adj.index, columns=adj.columns)


def _fast_strategy(panel: _PanelRecorder) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="obs_fast", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )

    # Flat loop twin so the fast path's parity guard agrees with the panel (both produce 0.0).
    def flat_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.0, index=syms) if syms else pd.Series(dtype="float64")

    return LoadedStrategy(
        config=cfg, signal_fn=flat_loop, signal_panel_fn=panel, construct_fn=_passthrough
    )


def test_fast_path_panel_excludes_undeclared_symbol() -> None:
    panel = _PanelRecorder()
    strat = _fast_strategy(panel)
    # run() drives simulate(), which selects the fast path (signal_panel_fn set, static mode).
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert panel.seen, "signal_panel was never invoked"
    assert "ZZZ" not in panel.seen
    assert panel.seen <= {"AAA", "BBB"}
