from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError, run
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Test construction policy: the signal already emits the desired raw weights, so construction
    is the identity. Lets these engine tests drive exact weight vectors at the risk rails."""
    return scores


def _strategy(cfg: StrategyConfig, signal) -> LoadedStrategy:
    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def _equal_weight_strategy() -> LoadedStrategy:
    cfg = StrategyConfig(
        name="ew",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    return _strategy(cfg, signal)


def test_run_produces_metrics_keys() -> None:
    res = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    for key in [
        "total_return", "cagr", "ann_volatility", "sharpe", "max_drawdown",
        "turnover", "avg_gross_exposure", "n_rebalances",
        # Golden-Rule-6 dashboard breadth (#348)
        "sortino", "calmar", "hit_rate", "tail_ratio",
    ]:
        assert key in res.metrics


def test_run_is_deterministic() -> None:
    a = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    b = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    assert a.metrics == b.metrics


def test_t_plus_1_blocks_same_bar_fill():
    """Guards the t->t+1 rule: a position entered (decided) at bar t must fill at t+1,
    so it cannot capture a close[t]->close[t+1] jump. The strategy decides to go long
    while price is still 100 and HOLDS; the price then jumps 100->150 from bar 5 to bar 6.
    Honest (lag=1) fills at 150 and earns ~0; a broken/removed shift would fill at 100 and
    capture +50%, failing the assertion."""

    class JumpProvider:
        seed = 0

        def get_bars(self, symbols, start, end, timeframe):
            ts = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")
            path = [100.0] * 6 + [150.0] * 6  # flat 100 through bar 5, jump to 150 at bar 6
            rows = [{"timestamp": t, "symbol": "AAA", "open": px, "high": px, "low": px,
                     "close": px, "adj_close": px, "volume": 1.0}
                    for t, px in zip(ts, path, strict=True)]
            return pd.DataFrame(rows).set_index("timestamp").sort_index()

    cfg = StrategyConfig(
        name="holder", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )

    def holder(view, params):
        # Deterministically go long once we have >=6 bars of history (decision at bar index 5,
        # price still 100), and HOLD from then on.
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) >= 6:
            return pd.Series([1.0], index=["AAA"])
        return pd.Series(dtype="float64")

    res = run(_strategy(cfg, holder), JumpProvider(),
              datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC))
    # Entered at t+1 (price 150), held flat at 150 -> earns ~0 from the jump it sat through.
    assert res.metrics["total_return"] < 0.01
    # It genuinely held a position (not vacuous).
    assert res.metrics["n_rebalances"] >= 1


def test_static_operating_universe_empty_raises() -> None:
    """A misbehaving provider that returns bars only for UNDECLARED symbols yields an empty
    strategy.universe & adj.columns intersection -> fail closed rather than run a flat backtest."""
    class _WrongSymbolProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            return SyntheticProvider(seed=0).get_bars(["ZZZ"], start, end, timeframe)

    cfg = StrategyConfig(
        name="wrongdata", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series(dtype="float64"))
    with pytest.raises(BacktestError, match="no fetched price data for any symbol"):
        run(strat, _WrongSymbolProvider(), START, END)


def test_empty_universe_data_raises() -> None:
    cfg = StrategyConfig(
        name="x",
        universe=[],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={}, construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series(dtype="float64"))
    with pytest.raises(BacktestError):
        run(strat, SyntheticProvider(), START, END)


def test_rejects_weights_exceeding_max_gross_exposure():
    cfg = StrategyConfig(
        name="overlev", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    max_gross_exposure=1.0),
        params={}, construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series([1.5, 1.5], index=["AAA", "BBB"]))
    with pytest.raises(BacktestError):
        run(strat, SyntheticProvider(seed=1), START, END)


def test_rejects_unsupported_cadence():
    cfg = StrategyConfig(
        name="weekly", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1w", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series(dtype="float64"))
    with pytest.raises(BacktestError):
        run(strat, SyntheticProvider(seed=1), START, END)


def test_run_stamps_snapshot_id_when_provider_exposes_it():
    class StampedProvider:
        snapshot_id = "snap-123"

        def get_bars(self, symbols, start, end, timeframe):
            return SyntheticProvider(seed=1).get_bars(symbols, start, end, timeframe)

    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
        construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series(
        1.0 / len(v["symbol"].unique()), index=sorted(v["symbol"].unique())))
    res = run(strat, StampedProvider(), START, END)
    assert res.snapshot_id == "snap-123"


def test_run_stamps_code_and_dependency_identity():
    res = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END)
    payload = res.to_dict()
    assert isinstance(payload["code_hash"], str) and len(payload["code_hash"]) >= 7
    assert isinstance(payload["dependency_hash"], str) and len(payload["dependency_hash"]) == 64


def test_explicit_seed_recorded_when_provider_has_none():
    # #43: provider w/o a seed attr -> the run's explicit seed is the provenance seed.
    class NoSeedProvider:
        def get_bars(self, symbols, start, end, timeframe):
            return SyntheticProvider(seed=1).get_bars(symbols, start, end, timeframe)

    res = run(_equal_weight_strategy(), NoSeedProvider(), START, END, seed=99)
    assert res.seed == 99


def test_provider_seed_wins_over_explicit_seed():
    res = run(_equal_weight_strategy(), SyntheticProvider(seed=3), START, END, seed=99)
    assert res.seed == 3


def test_warmup_bars_zeroes_weights_in_warmup_window():
    """#41: warmup_bars must suppress positions for the first `warmup_bars` bars.

    A strategy that always wants 100% AAA, with warmup_bars=5 and lag=1, cannot hold
    anything until after the warmup window, so its first rebalance is delayed.
    """
    cfg_warm = StrategyConfig(
        name="warm", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=5),
        params={}, construction="passthrough",
    )
    cfg_none = StrategyConfig(
        name="nowarm", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                    warmup_bars=0),
        params={}, construction="passthrough",
    )
    fn = lambda v, p: pd.Series([1.0], index=["AAA"])  # noqa: E731
    warm = run(_strategy(cfg_warm, fn), SyntheticProvider(seed=2), START, END)
    none = run(_strategy(cfg_none, fn), SyntheticProvider(seed=2), START, END)
    # Warmup suppresses early holding -> lower average gross exposure than the no-warmup run.
    assert warm.metrics["avg_gross_exposure"] < none.metrics["avg_gross_exposure"]


def test_simulate_label_aligns_misordered_weights():
    """Direct check of label-aligned assignment in the simulation step (#42)."""
    from algua.backtest import simulate

    cfg = StrategyConfig(
        name="rev", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    # Strategy always wants 100% BBB but returns it as the only label.
    strat = _strategy(cfg, lambda v, p: pd.Series([1.0], index=["BBB"]))
    _pf, weights_eff, _forced = simulate(strat, SyntheticProvider(seed=4), START, END)
    # Every non-zero weight must sit in the BBB column; AAA must remain flat zero.
    assert (weights_eff["AAA"] == 0.0).all()
    assert weights_eff["BBB"].abs().sum() > 0


# --- #7: point-in-time universe membership (survivorship-bias fix) --------------------------


def _pit_provider():
    """Synthetic provider over a fixed AAA+BBB panel; records the symbols arg it was asked for."""

    class PITProvider:
        seed = 0

        def __init__(self):
            self.requested_symbols = None

        def get_bars(self, symbols, start, end, timeframe):
            self.requested_symbols = list(symbols)
            return SyntheticProvider(seed=0).get_bars(["AAA", "BBB"], start, end, timeframe)

    return PITProvider()


def test_decision_weights_masks_symbol_before_effective_date():
    """A symbol added mid-period is ABSENT from the strategy view before its effective date and
    PRESENT on/after it. A spy strategy records the symbols it sees per bar."""
    from datetime import date

    from algua.backtest.engine import _decision_weights

    provider = SyntheticProvider(seed=0)
    bars = provider.get_bars(["AAA", "BBB"], START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    sessions = list(adj.index)
    cutover = sessions[len(sessions) // 2].date()
    # Before cutover only AAA is a member; on/after cutover AAA+BBB.
    universe_by_date = {}
    for ts in adj.index:
        d = ts.date()
        universe_by_date[d] = {"AAA"} if d < cutover else {"AAA", "BBB"}

    seen: dict = {}

    def spy(view, params):
        d = view.index[-1].date()
        seen[d] = set(view["symbol"].unique())
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms)

    cfg = StrategyConfig(
        name="spy", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    _decision_weights(_strategy(cfg, spy), bars, adj,
                      universe_by_date=universe_by_date)

    before = [d for d in seen if d < cutover]
    after = [d for d in seen if d >= cutover]
    assert before and after
    assert all("BBB" not in seen[d] for d in before), "BBB leaked before its effective date"
    assert all("BBB" in seen[d] for d in after), "BBB missing on/after its effective date"
    assert isinstance(cutover, date)


def test_decision_weights_flat_before_earliest_membership():
    """Bars whose date precedes the earliest effective date have empty membership -> flat,
    and the strategy is never even called for them."""
    from algua.backtest.engine import _decision_weights

    provider = SyntheticProvider(seed=0)
    bars = provider.get_bars(["AAA"], START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    sessions = list(adj.index)
    start_membership = sessions[len(sessions) // 2].date()
    universe_by_date = {ts.date(): {"AAA"} for ts in adj.index if ts.date() >= start_membership}

    cfg = StrategyConfig(
        name="x", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    fn = lambda v, p: pd.Series([1.0], index=["AAA"])  # noqa: E731
    weights = _decision_weights(_strategy(cfg, fn), bars, adj,
                                universe_by_date=universe_by_date)

    pre = weights[weights.index.map(lambda ts: ts.date() < start_membership)]
    post = weights[weights.index.map(lambda ts: ts.date() >= start_membership)]
    assert (pre["AAA"] == 0.0).all()
    assert post["AAA"].abs().sum() > 0


def test_decision_weights_rejects_non_member_weight():
    """A weight returned for a symbol that is NOT an as-of member is a strategy bug -> BacktestError
    naming the offending symbol."""
    from algua.backtest.engine import _decision_weights

    provider = SyntheticProvider(seed=0)
    bars = provider.get_bars(["AAA", "BBB"], START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()
    # BBB is never a member, but the strategy insists on holding it.
    universe_by_date = {ts.date(): {"AAA"} for ts in adj.index}

    cfg = StrategyConfig(
        name="cheat", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    fn = lambda v, p: pd.Series([1.0], index=["BBB"])  # noqa: E731
    with pytest.raises(BacktestError, match="BBB"):
        _decision_weights(_strategy(cfg, fn), bars, adj,
                          universe_by_date=universe_by_date)


def test_decision_weights_none_is_unchanged_static_behavior():
    """No PIT map (None) => identical behavior to before: the full panel is visible every bar."""
    from algua.backtest.engine import _decision_weights

    provider = SyntheticProvider(seed=0)
    bars = provider.get_bars(["AAA", "BBB"], START, END, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    fn = lambda v, p: pd.Series(  # noqa: E731
        1.0 / len(v["symbol"].unique()), index=sorted(v["symbol"].unique()))
    strat = _strategy(cfg, fn)
    static = _decision_weights(strat, bars, adj)
    pit_none = _decision_weights(strat, bars, adj, universe_by_date=None)
    assert static.equals(pit_none)


def test_simulate_fetches_union_of_pit_members_not_strategy_universe():
    """#7 union-fetch: in PIT mode `simulate` fetches the UNION of all ever-effective members
    (spy provider records the symbols arg), NOT the static strategy.universe."""
    from algua.backtest.engine import simulate

    provider = _pit_provider()
    # PIT timeline: CCC effective early, DDD added later. strategy.universe is a DIFFERENT,
    # decoy list that must NOT drive the fetch.
    early = START.date()
    universe_by_date = {early: {"CCC"}, START.replace(month=2).date(): {"CCC", "DDD"}}

    cfg = StrategyConfig(
        name="ew", universe=["ZZZ_DECOY"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    # Strategy only ever holds symbols it actually sees; the provider serves AAA/BBB regardless,
    # so the masked view may be empty and the run stays flat — we only assert the FETCH arg here.
    fn = lambda v, p: pd.Series(dtype="float64")  # noqa: E731
    simulate(_strategy(cfg, fn), provider, START, END,
             universe_by_date=universe_by_date)
    assert provider.requested_symbols == ["CCC", "DDD"]  # union, sorted; not ZZZ_DECOY
