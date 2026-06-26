from __future__ import annotations

import pytest

from algua.execution.schedule import is_due


def test_daily_is_always_due():
    assert is_due("1d") is True


def test_unknown_frequency_fails_closed():
    with pytest.raises(ValueError, match="unsupported rebalance_frequency"):
        is_due("1w")
