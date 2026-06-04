"""The public backtest seam a stacked follow-up PR (and walk-forward/sweep) depend on.

Promotes the formerly-private engine internals (#38) to a stable public surface and
pins the one canonical metrics module (#37).
"""
from __future__ import annotations

import inspect

from algua.backtest import (
    BacktestError,
    build_portfolio,
    config_hash,
    metrics_from_returns,
    portfolio_metrics,
    simulate,
)


def test_public_names_are_exported_from_package():
    # The follow-up PR imports these from the package root, not private modules.
    assert callable(build_portfolio)
    assert callable(simulate)
    assert callable(config_hash)
    assert callable(metrics_from_returns)
    assert callable(portfolio_metrics)
    assert issubclass(BacktestError, RuntimeError)


def test_simulate_is_the_public_name_for_build_portfolio():
    # build_portfolio is the explicit public alias of the simulation step.
    assert build_portfolio is simulate


def test_no_private_engine_internals_imported_by_walkforward_or_sweep():
    import algua.backtest.sweep as sweep_mod
    import algua.backtest.walkforward as wf_mod

    for mod in (wf_mod, sweep_mod):
        src = inspect.getsource(mod)
        assert "_build_portfolio" not in src
        assert "_config_hash" not in src


def test_config_hash_lives_with_provenance_in_result_module():
    from algua.backtest import result as result_mod

    assert hasattr(result_mod, "config_hash")
    assert hasattr(result_mod, "provenance")
