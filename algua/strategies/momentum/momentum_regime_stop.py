"""Cross-sectional momentum with a regime gate and a trailing stop — the bundled OVERLAYS example.

Same alpha as `cross_sectional_momentum` (trailing return, top-k equal weight); the difference is
the `overlays` chain: `regime_gate` scales the whole book by a universe-derived regime multiplier,
then `trailing_stop` zeroes any name more than 15% off its 60-bar high (with a 5-bar cooldown).
Both are stateless functions of the PIT view, enforced tighten-only inside construct(). It also
exposes `signal_panel`, so the exhaustive parity gate exercises the overlay chain on every bar."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.features.alphas import xs_trailing_return
from algua.portfolio.overlays import OverlaySpec
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): bundled examples are hand-authored.
GENERATED_BY = "human"

_REGIME = {
    "trend_window": 126, "dd_window": 63, "dd_threshold": 0.10,
    "turb_window": 63, "z_window": 126, "turb_z": 3.0,
    "persistence": 5,
    "shock_window": 3, "shock_return": 0.05, "fast_turb_z": 4.0, "fast_lookback": 5,
    "neutral_exposure": 0.6, "risk_off_exposure": 0.2, "fast_exposure": 0.25,
}
_STOP = {"lookback": 60, "stop_pct": 0.15, "cooldown_bars": 5}

CONFIG = StrategyConfig(
    name="momentum_regime_stop",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    overlays=[
        OverlaySpec(policy="regime_gate", params=_REGIME),
        OverlaySpec(policy="trailing_stop", params=_STOP),
    ],
    # max(signal 60, regime_gate max(126, 63, 63+126, 3+5) + 63 = 252, trailing_stop 60+5 = 65)
    feature_lookback=252,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score)."""
    return xs_trailing_return(view, params)


def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Vectorized SCORES twin of `signal`; the overlays run inside construct() per row either
    way."""
    lookback = int(params["lookback"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide / wide.shift(lookback) - 1.0
