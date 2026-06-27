"""Unit tests for the fundamentals_earnings_tilt signal selection (#274)."""
import pandas as pd

from algua.strategies.fundamentals.fundamentals_earnings_tilt import signal


def _funds(rows):
    return pd.DataFrame(
        rows,
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def test_signal_picks_latest_fiscal_period_regardless_of_frame_order():
    """The newest fiscal period's value wins even when an OLDER period appears last in frame
    order — proving selection is by fiscal_period_end, not by incoming row order (#274)."""
    funds = _funds(
        [
            # Newer period listed FIRST; older period listed LAST. A plain .last() over frame
            # order would wrongly pick the older period's value (1.0).
            ["AAPL", "2025-06-30", "eps_diluted", 5.0, "2025-08-01T13:00:00Z", "v"],
            ["AAPL", "2025-03-31", "eps_diluted", 1.0, "2025-05-01T13:00:00Z", "v"],
        ]
    )
    out = signal(pd.DataFrame(), {"metric": "eps_diluted"}, funds)
    assert out["AAPL"] == 5.0


def test_signal_empty_when_metric_absent():
    funds = _funds([["AAPL", "2025-06-30", "eps_diluted", 5.0, "2025-08-01T13:00:00Z", "v"]])
    out = signal(pd.DataFrame(), {"metric": "revenue"}, funds)
    assert out.empty
