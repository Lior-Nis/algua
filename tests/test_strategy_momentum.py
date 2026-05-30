import pandas as pd
from algua.strategies.examples import cross_sectional_momentum as csm


def _bars(prices_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
    # Build a bar-schema long frame from per-symbol adj_close paths.
    ts = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for sym, path in prices_by_symbol.items():
        for t, px in zip(ts, path):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_momentum_picks_top_k_winners_equal_weight():
    # WIN doubles, FLAT flat, LOSE halves over the window -> top_k=1 -> all weight on WIN
    view = _bars({"WIN": [10, 12, 16, 20], "FLAT": [10, 10, 10, 10], "LOSE": [10, 9, 8, 7]})
    params = {"lookback": 3, "top_k": 1}
    w = csm.target_weights(view, params)
    assert w.idxmax() == "WIN"
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w.drop("WIN") == 0).all()


def test_has_config():
    assert csm.CONFIG.name == "cross_sectional_momentum"
    assert csm.CONFIG.execution.decision_lag_bars >= 1
