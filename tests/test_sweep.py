from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import SweepResult, sweep
from algua.contracts.types import ExecutionContract
from algua.features.indicators import momentum
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _momentum_signal(view, params):
    # Module-level (not a closure) so the strategy pickles to a ProcessPoolExecutor worker,
    # exactly like a real loaded strategy (the loader binds a module-level `signal`).
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= int(params["lookback"]):
        return pd.Series(dtype="float64")
    return momentum(wide, lookback=int(params["lookback"])).iloc[-1].dropna()


def _momentum():
    cfg = StrategyConfig(
        name="m", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 40},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    return LoadedStrategy(
        config=cfg, signal_fn=_momentum_signal,
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_sweep_ranks_and_counts():
    res = sweep(_momentum(), SyntheticProvider(seed=3), START, END,
                grid={"lookback": [20, 40], "construction.top_k": [1, 2]}, windows=4,
                holdout_frac=0.2)
    assert isinstance(res, SweepResult)
    d = res.to_dict()
    assert d["n_combos"] == 4
    assert len(d["ranked"]) == 4
    scores = [r["score"] for r in d["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert d["ranked"][0]["score"] == d["best"]["score"]
    top = d["ranked"][0]
    # The holdout is WITHHELD from sweep records (reserved for `research promote`); only the
    # window/stability ranking signal is exposed.
    assert "holdout" not in top and "stability" in top
    assert top["score"] == top["stability"]["mean_sharpe"]
    assert set(top["params"]) == {"lookback", "construction.top_k"}
    assert d["code_hash"] and d["dependency_hash"]


def test_sweep_is_deterministic():
    kw = dict(
        grid={"lookback": [20, 40], "construction.top_k": [1, 2]}, windows=4, holdout_frac=0.2
    )
    a = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    b = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    assert a.to_dict() == b.to_dict()


def test_sweep_rejects_bad_rank_by():
    with pytest.raises(ValueError):
        sweep(_momentum(), SyntheticProvider(seed=3), START, END,
              grid={"lookback": [20, 40]}, rank_by="holdout_sharpe")


def test_sweep_records_windows_and_holdout_frac():
    res = sweep(_momentum(), SyntheticProvider(seed=3), START, END,
                grid={"lookback": [20, 40]}, windows=3, holdout_frac=0.25)
    d = res.to_dict()
    assert d["windows"] == 3
    assert d["holdout_frac"] == 0.25


def test_evaluate_combo_returns_record_without_holdout():
    from algua.backtest.sweep import _evaluate_combo, _override

    ov = _override(_momentum(), {"lookback": 20})
    rec = _evaluate_combo(
        ov, provider=SyntheticProvider(seed=3), start=START, end=END,
        windows=4, holdout_frac=0.2,
        universe_by_date=None, universe_name=None, universe_snapshots=None,
        rank_by="mean_sharpe",
        delisting_records=None, assume_terminal_last_close=False,
    )
    # The rankable fields are present; the holdout never leaves the worker.
    assert set(rec) == {"config_hash", "n_windows", "stability", "score", "meta"}
    assert "holdout_metrics" not in rec and "holdout" not in rec
    assert rec["score"] == rec["stability"]["mean_sharpe"]
    assert set(rec["meta"]) == {
        "data_source", "snapshot_id", "timeframe", "seed", "code_hash",
        "dependency_hash", "period", "universe_name", "universe_snapshots",
    }


def _run_kwargs():
    return dict(
        provider=SyntheticProvider(seed=3), start=START, end=END,
        windows=4, holdout_frac=0.2,
        universe_by_date=None, universe_name=None, universe_snapshots=None,
        rank_by="mean_sharpe",
        delisting_records=None, assume_terminal_last_close=False,
    )


def test_run_combos_inline_single_combo():
    from algua.backtest.sweep import _override, _run_combos

    overridden = [_override(_momentum(), {"lookback": 20})]  # len==1 -> inline path
    results = _run_combos(overridden, _run_kwargs())
    assert len(results) == 1
    assert results[0]["score"] == results[0]["stability"]["mean_sharpe"]


def _force_pool(monkeypatch):
    # The pool path only engages when n_workers > 1. On a single-core CI host os.cpu_count() == 1
    # would silently route these "pool" tests through the inline branch, so pin it to 2 to actually
    # exercise ProcessPoolExecutor.
    from algua.backtest import sweep as sweep_mod
    monkeypatch.setattr(sweep_mod.os, "cpu_count", lambda: 2)


def test_run_combos_pool_preserves_order(monkeypatch):
    from algua.backtest.sweep import _evaluate_combo, _override, _run_combos

    _force_pool(monkeypatch)
    combos = [{"lookback": 20}, {"lookback": 30}, {"lookback": 40}]  # >1 -> pool path
    overridden = [_override(_momentum(), c) for c in combos]
    pooled = _run_combos(overridden, _run_kwargs())
    # Ground truth: sequential evaluation in combo order. The pool MUST match it position-for-
    # position — proving the result is combo-ordered, not completion-ordered or reversed (comparing
    # two pooled runs to each other would pass even for a consistently-wrong ordering).
    sequential = [_evaluate_combo(ov, **_run_kwargs()) for ov in overridden]
    assert [r["config_hash"] for r in pooled] == [r["config_hash"] for r in sequential]
    assert [r["score"] for r in pooled] == [r["score"] for r in sequential]
    assert len(pooled) == 3


def test_run_combos_non_picklable_strategy_raises_backtest_error(monkeypatch):
    from algua.backtest.engine import BacktestError
    from algua.backtest.sweep import _override, _run_combos
    from algua.portfolio.construction import get_construction_policy

    _force_pool(monkeypatch)

    # A signal defined inside this test is a LOCAL object -> pickle raises AttributeError (not
    # PicklingError). The parent-side preflight must convert that into a JSON-safe BacktestError,
    # never let the raw AttributeError escape past the CLI's @json_errors.
    def _closure_signal(view, params):
        return pd.Series(dtype="float64")

    cfg = StrategyConfig(
        name="closure", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 5},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=_closure_signal,
        construct_fn=get_construction_policy(cfg.construction),
    )
    overridden = [_override(strat, {"lookback": 5}), _override(strat, {"lookback": 6})]
    with pytest.raises(BacktestError, match="picklable"):
        _run_combos(overridden, _run_kwargs())


def test_sweep_combo_error_surfaces_as_backtest_error(monkeypatch):
    from algua.backtest.engine import BacktestError

    _force_pool(monkeypatch)
    # >1 combo through the pool; `windows` far too large for the period forces walk_forward to raise
    # BacktestError ("not enough bars") inside a worker. It must come back as BacktestError
    # (CLI-wrappable), not a BrokenProcessPool and not a partial result.
    with pytest.raises(BacktestError):
        sweep(_momentum(), SyntheticProvider(seed=3), START, END,
              grid={"lookback": [20, 40]}, windows=500, holdout_frac=0.2)
