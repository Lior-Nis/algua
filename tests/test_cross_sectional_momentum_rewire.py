import pandas as pd

from algua.features.alphas import xs_trailing_return
from algua.registry.lineage import factors_used_by
from algua.strategies.momentum import cross_sectional_momentum as csm


def _view():
    rows = []
    for i, ts in enumerate(pd.date_range("2023-01-01", periods=70, freq="D")):
        rows.append((ts, "AAA", 100.0 + i))
        rows.append((ts, "BBB", 100.0 + 2 * i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"]).set_index("timestamp")


def test_signal_delegates_to_alpha_identically():
    view = _view()
    params = {"lookback": 60}
    pd.testing.assert_series_equal(csm.signal(view, params), xs_trailing_return(view, params))


def test_lineage_reports_the_composed_factor():
    used = {s.name for s in factors_used_by("cross_sectional_momentum")}
    assert "xs_trailing_return" in used
