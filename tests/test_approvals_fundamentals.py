"""FIX A: authored_signal property + code_hash for fundamentals strategies."""
from __future__ import annotations

import inspect

from algua.registry.approvals import compute_artifact_hashes
from algua.strategies.loader import load_strategy


def test_authored_signal_for_fundamentals_strategy_resolves_to_correct_module():
    """authored_signal on a needs_fundamentals strategy returns the 3-arg fn, not None."""
    strat = load_strategy("fundamentals_earnings_tilt")
    assert strat.config.needs_fundamentals
    # signal_fn is None for a fundamentals strategy; authored_signal must not be None
    assert strat.signal_fn is None
    sfn = strat.authored_signal
    assert sfn is not None
    mod = inspect.getmodule(sfn)
    assert mod is not None
    assert mod.__name__.endswith("fundamentals_earnings_tilt")


def test_compute_artifact_hashes_fundamentals_strategy_does_not_crash():
    """compute_artifact_hashes must not crash on a needs_fundamentals strategy."""
    hashes = compute_artifact_hashes("fundamentals_earnings_tilt")
    assert hashes.code_hash  # non-empty string
    assert hashes.config_hash
    assert hashes.dependency_hash


# sha256("") — what you get when a None fn yields no source
_EMPTY_CLOSURE_HASH = "e3b0c44298fc1c14"


def test_code_hash_differs_from_plain_strategy_and_not_empty_closure():
    """The fundamentals strategy code_hash must be derived from real source (not the empty-closure
    constant produced when inspect.getmodule(None) returns None)."""
    h_fund = compute_artifact_hashes("fundamentals_earnings_tilt")
    h_mom = compute_artifact_hashes("cross_sectional_momentum")
    # Must not be the empty-closure sentinel
    assert h_fund.code_hash != _EMPTY_CLOSURE_HASH, (
        "code_hash for fundamentals_earnings_tilt is the empty-closure constant; "
        "signal_fn is not wired in approvals.py"
    )
    assert h_fund.code_hash != h_mom.code_hash
