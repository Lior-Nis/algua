"""Shared numeric constants for the backtest core.

One home for the annualization factor and the named tolerances that were previously
duplicated/magic across engine.py and walkforward.py (#47).
"""
from __future__ import annotations

# Trading days per year, used to annualize mean return and volatility.
ANN = 252

# Tolerance when checking that targeted gross exposure does not exceed the cap.
GROSS_EXPOSURE_TOL = 1e-9

# A per-bar weight change larger than this counts as a rebalance.
REBALANCE_EPS = 1e-12
