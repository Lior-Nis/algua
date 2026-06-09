from datetime import datetime

import pandas as pd

from algua.backtest.engine import run
from algua.cli._common import resolve_universe_inputs
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy


def test_future_member_fundamentals_do_not_leak(tmp_path, monkeypatch):
    """NVDA joins the universe only AFTER the decision window; even though its EPS is positive and
    knowable, the as-of-member mask must keep it out of the weights while it is not a member."""
    # Point ALGUA_DATA_DIR at tmp_path so resolve_universe_inputs finds the seeded universe.
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))

    store = DataStore(tmp_path)
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "NVDA"] for t in idx]
    bars = pd.DataFrame(
        rows,
        columns=["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"],
    )
    brec = store.ingest_bars(
        provider="t", symbols=["AAPL", "NVDA"], start="2025-01-01",
        end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars,
    )
    # universe: only AAPL is a member during the window; NVDA becomes a member far later.
    store.ingest_universe(
        universe="u", symbols=["AAPL"], effective_date="2024-12-01",
        as_of="2025-01-01T00:00:00Z", source="t",
    )
    funds = pd.DataFrame(
        [
            ["AAPL", "2024-12-31", "eps_diluted", 1.0, "2024-12-31T00:00:00Z", "v"],
            ["NVDA", "2024-12-31", "eps_diluted", 1.0, "2024-12-31T00:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )
    frec = store.ingest_fundamentals(
        provider="v", symbols=["AAPL", "NVDA"],
        as_of="2025-01-01T00:00:00Z", source="v", frame=funds,
    )
    ubd, _ = resolve_universe_inputs("u", datetime(2025, 1, 1), datetime(2025, 1, 10))
    result = run(
        load_strategy("fundamentals_earnings_tilt"),
        StoreBackedProvider(store, brec.snapshot_id),
        datetime(2025, 1, 1), datetime(2025, 1, 10),
        universe_by_date=ubd, universe_name="u",
        fundamentals_provider=StoreBackedFundamentalsProvider(store, frec.snapshot_id),
    )
    # the run completes without a non-member-weight BacktestError => NVDA never got weight
    assert result.strategy == "fundamentals_earnings_tilt"
