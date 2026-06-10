"""Backtest <-> paper decision parity.

Algua's central claim is "one signal definition + one execution contract": the SAME
per-bar decision (target weights + risk checks + warm-up) drives both the backtester and
the paper/live loop, so a result discovered in research transfers to live. These tests pin
that the two paths make the IDENTICAL decision bar-for-bar, share a single risk-check source
of truth, and agree on what "warmup_bars = N" means.
"""
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError, _decision_weights, run, simulate
from algua.contracts.types import ExecutionContract
from algua.execution.sim_broker import SimBroker
from algua.live.paper_loop import run_paper
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _identity(scores: pd.Series, view: pd.DataFrame, params: dict) -> pd.Series:
    """Test-local construction policy: the authored scores ARE the target weights. Used where a test
    injects a PRECISE weight vector to exercise the risk rails, so no policy re-normalization
    distorts the vector under test."""
    return scores


def decided_weights_by_bar(
    sink: dict[datetime, pd.Series],
) -> Callable[[datetime, pd.Series], None]:
    """Build an `on_decision` callback that records each bar's decided target weights into `sink`.

    A minimal, side-effect-free observation seam: run_paper already computes the decided weights,
    so recording them does not change any production decision. Test-only — reads out the live
    path's per-bar decision to assert backtest<->paper parity."""

    def record(decision_ts: datetime, weights: pd.Series) -> None:
        sink[decision_ts] = weights

    return record


def _momentum_strategy(warmup_bars: int = 3) -> LoadedStrategy:
    """A long-only, path-dependent strategy: hold the single best 5-bar performer at 100%.

    Path-dependent so the per-bar decision actually varies bar-to-bar (a genuine parity test,
    not a constant-weights tautology), and long-only so it passes both paths' risk checks.
    """
    cfg = StrategyConfig(
        name="momo",
        universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=warmup_bars
        ),
        params={"lookback": 5},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        wide = view.reset_index().pivot(
            index="timestamp", columns="symbol", values="adj_close"
        )
        lookback = params["lookback"]
        if len(wide) <= lookback:
            return pd.Series(dtype="float64")
        return wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0

    from algua.portfolio.construction import top_k_equal_weight
    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)


def test_backtest_and_paper_decide_identical_target_weights_bar_for_bar() -> None:
    """KEYSTONE: one strategy, one deterministic dataset, run through BOTH the backtest decision
    path and run_paper, and assert the per-bar TARGET WEIGHTS (the decision, pre-fill) match
    bar-for-bar. PnL/fills are NOT compared (different fill models)."""
    strat = _momentum_strategy()
    provider = SyntheticProvider(seed=0)

    # Backtest pre-lag per-bar target weights (the raw decision, before the t->t+1 shift).
    bars = provider.get_bars(strat.universe, START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()
    bt_weights = _decision_weights(strat, bars, adj)

    # Paper per-bar decided weights, captured via a minimal recorder.
    recorder: dict[datetime, pd.Series] = {}
    run_paper(
        strat, SimBroker(cash=1_000_000.0), provider, START, END,
        on_decision=decided_weights_by_bar(recorder),
    )

    # The paper loop only evaluates bars with a successor (it needs t+1 to fill) and skips
    # warm-up. Compare exactly the bars the paper loop actually decided on.
    assert recorder, "paper loop decided on no bars"
    for t, paper_w in recorder.items():
        bt_row = bt_weights.loc[t].reindex(sorted(strat.universe)).fillna(0.0)
        paper_row = paper_w.reindex(sorted(strat.universe)).fillna(0.0)
        pd.testing.assert_series_equal(
            bt_row, paper_row, check_names=False,
            obj=f"target weights at {t}",
        )


def test_backtest_enforces_long_only_identically_to_live() -> None:
    """A strategy returning a negative (short) weight must FAIL the backtest, the same way it
    fails paper/live via check_short_policy. Previously the backtest only checked gross exposure."""
    cfg = StrategyConfig(
        name="shorty", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    short = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 1.0, "BBB": -0.5}),
        construct_fn=_identity,
    )
    with pytest.raises(BacktestError, match="long-only"):
        run(short, SyntheticProvider(seed=0), START, END)


def test_backtest_gross_exposure_uses_the_shared_risk_check() -> None:
    """Gross-exposure breach in the backtest surfaces as BacktestError for the CLI contract, but
    the underlying breach is the shared RiskBreach (single source of truth + tolerance)."""
    cfg = StrategyConfig(
        name="overlev", universe=["AAA", "BBB"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, max_gross_exposure=1.0
        ),
        params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    # Each name is within the default per-symbol cap (1.0) so the gross rail is what trips:
    # 0.6 + 0.6 = 1.2 gross > 1.0, isolating the gross-exposure breach.
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series([0.6, 0.6], index=["AAA", "BBB"]),
        construct_fn=_identity,
    )
    with pytest.raises(BacktestError) as ei:
        simulate(strat, SyntheticProvider(seed=1), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "gross_exposure"


def test_warmup_means_the_same_number_of_flat_bars_in_both_paths() -> None:
    """warmup_bars = N must hold the SAME initial bars flat in the backtest and the paper loop.

    Reconciles the historical off-by-one: backtest skipped indices 0..N-1 (N bars) while the
    paper loop skipped 0..N-2 (N-1 bars). The first bar EITHER path decides on must be the same.
    """
    warmup = 4
    strat = _momentum_strategy(warmup_bars=warmup)
    provider = SyntheticProvider(seed=0)

    bars = provider.get_bars(strat.universe, START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()
    sessions = list(adj.index)

    bt_weights = _decision_weights(strat, bars, adj)
    # First N sessions are flat in the backtest.
    for t in sessions[:warmup]:
        assert (bt_weights.loc[t] == 0.0).all(), f"backtest not flat during warm-up at {t}"

    recorder: dict[datetime, pd.Series] = {}
    run_paper(
        strat, SimBroker(cash=1_000_000.0), provider, START, END,
        on_decision=decided_weights_by_bar(recorder),
    )
    decided = sorted(recorder)
    # The paper loop never decides on any of the first N (warm-up) sessions.
    for t in sessions[:warmup]:
        assert t not in recorder, f"paper decided during warm-up at {t}"
    # The first bar the paper loop decides on is exactly session index N (the same bar the
    # backtest first evaluates), not N-1.
    assert decided[0] == sessions[warmup]


def test_concentration_cap_fails_backtest_and_paper_identically() -> None:
    """A weight vector busting the per-symbol cap must FAIL the backtest the same way it fails
    paper/live — the cap is a shared decision-time rail, not a live-only afterthought (#135)."""
    cfg = StrategyConfig(
        name="concentrated", universe=["AAA", "BBB"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, max_weight_per_symbol=0.5
        ),
        params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.9, "BBB": 0.1}),
        construct_fn=_identity,
    )
    with pytest.raises(BacktestError) as ei:
        run(strat, SyntheticProvider(seed=0), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "max_weight_per_symbol"
    with pytest.raises(RiskBreach) as ei_paper:
        run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)
    assert ei_paper.value.kind == "max_weight_per_symbol"


def test_allow_short_lets_a_short_through_in_both_paths() -> None:
    """With allow_short=True a negative weight is permitted in BOTH backtest and paper; the default
    (False) rejects it in both (the latter pinned by the existing long-only parity test)."""
    cfg = StrategyConfig(
        name="ls", universe=["AAA", "BBB"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, allow_short=True
        ),
        params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.5, "BBB": -0.5}),
        construct_fn=_identity,
    )
    # Neither path raises (long/short now declared+permitted); a clean run is the assertion.
    run(strat, SyntheticProvider(seed=0), START, END)
    run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)


def test_named_symbol_nan_breaches_both_paths() -> None:
    """A strategy that names a symbol but returns NaN for it must hard-breach (not silently flatten)
    in backtest and paper — fail-closed finite guard (#135)."""
    import numpy as np
    cfg = StrategyConfig(
        name="nanny", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": np.nan}), construct_fn=_identity,
    )
    with pytest.raises(BacktestError) as ei:
        run(strat, SyntheticProvider(seed=0), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "non_finite_weight"
    with pytest.raises(RiskBreach) as ei_paper:
        run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)
    assert ei_paper.value.kind == "non_finite_weight"
