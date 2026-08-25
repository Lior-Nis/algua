"""The simulation price grid: the (timestamp x symbol) pivot every backtest path is built on.

`adj_grid` imports nothing from algua — a leaf, same pattern as `algua.backtest.errors`. It is
needed by `algua.backtest.engine` (which still owns `adj_open_grid` and `simulate`, both built on
this grid), `algua.backtest.walkforward` (whose `holdout_window` reproduces this grid's index to
find the exact OOS boundary WITHOUT running the strategy — the #192 single-use holdout identity),
and `algua.backtest.decision_path` (whose `verify_signal_panel_parity` builds its own grid from
freshly fetched bars). Defining it in any one of those modules would force the others to import it
back — engine.py already imports `decision_path` for the dual-path selector, so decision_path
importing `adj_grid` from engine.py would be a real import cycle, not just a style objection. A
leaf breaks it the same way `errors.BacktestError` does. Moved verbatim out of
`algua.backtest.engine` (stage 7, task 2).
"""
from __future__ import annotations

import pandas as pd


def adj_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """The simulation grid: adj_close pivoted to (timestamp index x symbol columns), sorted by
    time. This index IS the bar date-index `vectorbt` simulates on and `pf.returns()` carries, so
    it is the single source of truth for both `build_portfolio` and `holdout_window`."""
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return adj.sort_index()
