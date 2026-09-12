"""The PAPER lane resolves weights THROUGH the overlay chain, not around it.

The branch's central architectural claim is that `LoadedStrategy.construct()` is the one
chokepoint every path resolves weights through, so the overlay stage needs no lane change. Until
now that rested on a code reading: a future refactor reintroducing a direct `construct_fn` call,
or a lane building its own `LoadedStrategy`, would silently drop overlays in LIVE TRADING with
nothing red. This test makes the claim executable at the lane boundary.

The LIVE lane shares the same decision core (`live_loop` -> `paper_loop.decide` -> the strategy's
`target_weights` -> `construct()`), so pinning the paper tick pins both; the live tick adds broker
authorization and reconciliation on top of the SAME decided vector.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import run_paper
from algua.portfolio.construction import score_proportional_long
from algua.portfolio.overlays import OverlaySpec, trailing_stop
from algua.strategies.base import LoadedStrategy, StrategyConfig

_SYMBOLS = ("AAA", "BBB")
_DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in range(2, 14)]
# Strictly DECREASING every bar for every symbol: a hair-thin trailing stop then fires on every
# name on every decided bar (except the first, where the rolling high IS the current price).
_STOP = {"lookback": 2, "stop_pct": 1e-6, "cooldown_bars": 0}


def _falling_bars() -> pd.DataFrame:
    rows = []
    for i, sym in enumerate(_SYMBOLS):
        for j, ts in enumerate(_DATES):
            px = 100.0 * (1.0 + i) * (0.99**j)
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1000.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class _FakeProvider:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def get_bars(self, symbols: list[str], start: datetime, end: datetime, tf: str) -> pd.DataFrame:
        return self._bars


def _signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Last adj_close per symbol — always positive, so construction always wants a full book."""
    return view.reset_index().pivot(
        index="timestamp", columns="symbol", values="adj_close"
    ).sort_index().iloc[-1]


def _strategy(*, overlaid: bool) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="paper_overlaid" if overlaid else "paper_bare", universe=list(_SYMBOLS),
        # warmup_bars=1 skips the first bar, where nothing can be below its own rolling high yet.
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=1),
        construction="score_proportional_long",
        overlays=[OverlaySpec(policy="trailing_stop", params=_STOP)] if overlaid else [],
        feature_lookback=2,
    )
    return LoadedStrategy(
        config=cfg, signal_fn=_signal, construct_fn=score_proportional_long,
        overlay_fns=(trailing_stop,) if overlaid else (),
    )


def _decisions(strategy: LoadedStrategy) -> list[pd.Series]:
    seen: list[pd.Series] = []
    run_paper(
        strategy, SimBroker(cash=100_000.0), _FakeProvider(_falling_bars()),
        _DATES[0], _DATES[-1], on_decision=lambda ts, w: seen.append(w),
    )
    return seen


def test_paper_tick_applies_the_overlay_chain() -> None:
    """Every weight the paper tick decides is zeroed by the stop — the tick went THROUGH
    `construct()`'s overlay chain. The control run (same signal, same construction, `overlays=[]`)
    decides a non-zero book on the same bars, so the assertion is about the overlays and not about
    a strategy that simply never wants a position."""
    overlaid = _decisions(_strategy(overlaid=True))
    bare = _decisions(_strategy(overlaid=False))

    assert len(overlaid) == len(bare) >= 5  # the lanes decided on the same bars
    for w in overlaid:
        assert set(w.index) == set(_SYMBOLS)  # not vacuous: both names are in the vector...
        assert all(v == 0.0 for v in w.to_numpy()), f"overlay chain skipped: {w.to_dict()}"
    for w in bare:
        assert sum(w.to_numpy()) > 0.0  # without overlays the same tick holds a full book
        assert set(w.index) == set(_SYMBOLS)
