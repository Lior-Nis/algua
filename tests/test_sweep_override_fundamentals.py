"""FIX B: _override preserves fundamentals_signal_fn."""
from __future__ import annotations

from algua.backtest.sweep import _override
from algua.strategies.loader import load_strategy


def test_override_preserves_fundamentals_signal_fn():
    """_override must not drop fundamentals_signal_fn when sweeping a needs_fundamentals strat."""
    base = load_strategy("fundamentals_earnings_tilt")
    assert base.fundamentals_signal_fn is not None  # precondition

    out = _override(base, {"metric": "eps_diluted"})

    assert out.fundamentals_signal_fn is not None, (
        "_override dropped fundamentals_signal_fn; the sweep combo silently lost the fundamentals "
        "lane"
    )
    assert out.fundamentals_signal_fn is base.fundamentals_signal_fn  # same object


def test_override_fundamentals_strategy_does_not_raise():
    """_override on a needs_fundamentals strategy must not raise."""
    base = load_strategy("fundamentals_earnings_tilt")
    out = _override(base, {"metric": "eps_diluted"})
    assert out.config.params["metric"] == "eps_diluted"
    assert out.config.needs_fundamentals is True
