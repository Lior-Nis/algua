"""FIX D: target_weights fails closed when fundamentals are required but not supplied."""
from __future__ import annotations

import pandas as pd
import pytest

from algua.strategies.loader import load_strategy


def test_target_weights_raises_when_fundamentals_missing():
    """A needs_fundamentals strategy called without a fundamentals frame must raise ValueError
    matching 'fail closed', not silently forward None to the strategy fn."""
    strat = load_strategy("fundamentals_earnings_tilt")
    assert strat.config.needs_fundamentals

    with pytest.raises(ValueError, match="fail closed"):
        strat.target_weights(pd.DataFrame())
