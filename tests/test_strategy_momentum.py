import pandas as pd

from algua.portfolio.construction import top_k_equal_weight
from algua.strategies.examples import cross_sectional_momentum as csm


def _bars(prices_by_symbol: dict[str, list[float]]) -> pd.DataFrame:
    # Build a bar-schema long frame from per-symbol adj_close paths.
    ts = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for sym, path in prices_by_symbol.items():
        for t, px in zip(ts, path, strict=False):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_signal_returns_trailing_return_scores():
    # WIN doubles, FLAT flat, LOSE halves over the window -> scores ranked WIN > FLAT > LOSE.
    view = _bars({"WIN": [10, 12, 16, 20], "FLAT": [10, 10, 10, 10], "LOSE": [10, 9, 8, 7]})
    params = {"lookback": 3}
    scores = csm.signal(view, params)
    assert scores["WIN"] > scores["FLAT"] > scores["LOSE"]


def test_signal_plus_top_k_construction_picks_winner():
    # top_k=1 over the scores -> all weight on WIN.
    view = _bars({"WIN": [10, 12, 16, 20], "FLAT": [10, 10, 10, 10], "LOSE": [10, 9, 8, 7]})
    scores = csm.signal(view, {"lookback": 3})
    w = top_k_equal_weight(scores, view, {"top_k": 1})
    assert w.idxmax() == "WIN"
    assert abs(w.sum() - 1.0) < 1e-9
    assert set(w.index) == {"WIN"}


def test_has_config():
    assert csm.CONFIG.name == "cross_sectional_momentum"
    assert csm.CONFIG.execution.decision_lag_bars >= 1
    assert csm.CONFIG.construction == "top_k_equal_weight"
    assert csm.CONFIG.construction_params == {"top_k": 3}


def test_authored_module_exposes_signal_not_protocol_name():
    # The authored layer is `signal(view, params)`; the protocol-level
    # `target_weights(features)` lives only on the LoadedStrategy adapter.
    assert hasattr(csm, "signal")
    assert not hasattr(csm, "target_weights")
    assert not hasattr(csm, "compute_weights")
