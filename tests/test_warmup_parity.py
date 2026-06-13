"""#166 gap 6: ONE unified three-way warmup regression.

Algua's "one execution contract" claim means `warmup_bars = N` must mean the SAME thing in all
three lanes — backtest, paper, AND live: hold the first N sessions flat and FIRST DECIDE on session
index N (the bar that sees N+1 sessions of history). Today only PAIRWISE guards exist
(`test_decision_parity.test_warmup_means_the_same_number_of_flat_bars_in_both_paths` for
backtest<->paper, `test_live_loop.test_run_tick_warmup_boundary_matches_backtest_paper_semantic`
for live). This pins the single boundary across all three at once, over the same strategy + data.
"""
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import _decision_weights
from algua.contracts.types import ExecutionContract
from algua.execution.alpaca_broker import TickSnapshot
from algua.execution.sim_broker import SimBroker
from algua.live.live_loop import run_tick
from algua.live.paper_loop import run_paper
from algua.portfolio.construction import top_k_equal_weight
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _momentum_strategy(warmup_bars: int) -> LoadedStrategy:
    """Long-only, path-dependent 5-bar momentum (same shape as the pairwise parity tests)."""
    cfg = StrategyConfig(
        name="momo", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=warmup_bars
        ),
        params={"lookback": 5},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        lookback = params["lookback"]
        if len(wide) <= lookback:
            return pd.Series(dtype="float64")
        return wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)


def _record_into(sink: dict[datetime, pd.Series]) -> Callable[[datetime, pd.Series], None]:
    def record(decision_ts: datetime, weights: pd.Series) -> None:
        sink[decision_ts] = weights
    return record


class _FlatBroker:
    """Minimal flat-book broker — enough for run_tick's decide path and warm-up early return.
    The warm-up parity assertion is about the decision POINT, not fills, so orders are no-ops."""

    def __init__(self) -> None:
        self.snapshots = 0

    def get_positions(self) -> pd.Series:
        return pd.Series(dtype="float64")

    def cancel_open_orders(self) -> None:
        pass

    def snapshot(self, universe: list[str]) -> TickSnapshot:
        self.snapshots += 1
        qtys = {s: 0.0 for s in universe}
        return TickSnapshot(equity=1_000_000.0, market_values=dict(qtys), qtys=qtys)

    def submit_sized(self, intent, snap, client_order_id=None, reserve=None) -> str:
        return "noop"


def test_backtest_paper_live_share_one_warmup_boundary() -> None:
    warmup = 4
    strat = _momentum_strategy(warmup_bars=warmup)
    provider = SyntheticProvider(seed=0)

    bars = provider.get_bars(strat.universe, START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    sessions = list(adj.sort_index().index)
    boundary = sessions[warmup]   # session index N — the shared first-decision bar

    # --- Backtest: the first N sessions are forced flat. ---
    bt_weights = _decision_weights(strat, bars, adj.sort_index())
    for t in sessions[:warmup]:
        assert (bt_weights.loc[t] == 0.0).all(), f"backtest not flat during warm-up at {t}"

    # --- Paper: never decides during warm-up; first decision is exactly the boundary. ---
    recorder: dict[datetime, pd.Series] = {}
    run_paper(strat, SimBroker(cash=1_000_000.0), provider, START, END,
              on_decision=_record_into(recorder))
    for t in sessions[:warmup]:
        assert t not in recorder, f"paper decided during warm-up at {t}"
    assert sorted(recorder)[0] == boundary

    # --- Live: N closed sessions => warm-up not met; N+1 => first decides on the boundary. ---
    # now = sessions[N] makes sessions[0..N-1] (N sessions) the only fully-closed bars.
    not_met = run_tick(strat, _FlatBroker(), provider, START, END,
                       now=sessions[warmup].to_pydatetime())
    assert not_met.target_weights == {} and not_met.submitted == []

    # now = sessions[N+1] makes sessions[0..N] (N+1 sessions) closed => decides on session index N.
    broker = _FlatBroker()
    met = run_tick(strat, broker, provider, START, END,
                   now=sessions[warmup + 1].to_pydatetime())
    assert met.decision_ts == boundary
    assert broker.snapshots == 1   # warm-up satisfied: a sizing snapshot was taken
