"""Overlays in the strategy contract: identity fold, construct() chain, loader validation,
closure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import algua.strategies.momentum as momentum_pkg
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.portfolio.overlays import OverlaySpec, trailing_stop
from algua.registry.approvals import closure_module_names
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash
from algua.strategies.loader import StrategyNotFound, _loaded_for_test, load_strategy

_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def _cfg(**over: Any) -> StrategyConfig:
    base: dict[str, Any] = dict(
        name="s", universe=["A", "B"], execution=ExecutionContract(rebalance_frequency="1d"),
        params={"lookback": 10}, construction="top_k_equal_weight",
        construction_params={"top_k": 2},
    )
    base.update(over)
    return StrategyConfig(**base)


# --- identity --------------------------------------------------------------------------------

def test_empty_overlays_leaves_config_hash_byte_identical():
    # Digest of _cfg() computed on main at 82b8ec7, BEFORE the overlays field existed. An
    # undeclared / empty overlays list must reproduce it exactly (no live-approval churn).
    assert config_hash(_loaded_for_test(_cfg())) == "ea29606c94cca1a5731a1fe4552c9ea5"
    assert config_hash(_loaded_for_test(_cfg(overlays=[]))) == "ea29606c94cca1a5731a1fe4552c9ea5"


def test_non_empty_overlays_change_config_hash_and_order_matters():
    a = OverlaySpec(policy="trailing_stop", params=_TS)
    b = OverlaySpec(policy="trailing_stop", params={**_TS, "stop_pct": 0.2})
    base = config_hash(_loaded_for_test(_cfg()))
    ab = config_hash(_loaded_for_test(_cfg(overlays=[a, b])))
    ba = config_hash(_loaded_for_test(_cfg(overlays=[b, a])))
    assert base != ab and ab != ba


def test_overlay_param_change_changes_config_hash():
    a = config_hash(
        _loaded_for_test(_cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)]))
    )
    b = config_hash(
        _loaded_for_test(
            _cfg(overlays=[OverlaySpec(policy="trailing_stop", params={**_TS, "cooldown_bars": 3})])
        )
    )
    assert a != b


# --- LoadedStrategy ---------------------------------------------------------------------------

def test_loaded_strategy_requires_one_fn_per_declared_overlay():
    cfg = _cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)])
    with pytest.raises(
        ValueError, match="overlay_fns must hold one resolved fn per config overlay"
    ):
        LoadedStrategy(config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
                       construct_fn=get_construction_policy("top_k_equal_weight"))


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = [{"timestamp": t, "symbol": s, "open": px, "high": px, "low": px, "close": px,
             "adj_close": px, "volume": 1.0}
            for s, path in prices.items() for t, px in zip(ts, path, strict=True)]
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_construct_applies_the_overlay_chain_after_construction():
    cfg = _cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)])
    strat = LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series({"A": 2.0, "B": 1.0}),
        construct_fn=get_construction_policy("top_k_equal_weight"),
        overlay_fns=(trailing_stop,),
    )
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})  # A breached
    w = strat.target_weights(view)
    assert w.to_dict() == {"A": 0.0, "B": 0.5}  # top-2 equal weight, then A stopped; B stays 0.5


def test_construct_without_overlays_is_unchanged():
    strat = LoadedStrategy(
        config=_cfg(),
        signal_fn=lambda v, p: pd.Series({"A": 2.0, "B": 1.0}),
        construct_fn=get_construction_policy("top_k_equal_weight"),
    )
    view = _view({"A": [1.0] * 3, "B": [1.0] * 3})
    assert strat.target_weights(view).to_dict() == {"A": 0.5, "B": 0.5}


# --- loader -----------------------------------------------------------------------------------

def _write(stem: str, overlays_src: str, feature_lookback: str = "None") -> Path:
    path = Path(momentum_pkg.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.portfolio.overlays import OverlaySpec\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive',\n"
        f"    feature_lookback={feature_lookback},\n"
        f"    overlays={overlays_src})\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return pd.Series(dtype='float64')\n"
    )
    return path


@pytest.fixture
def tmp_strategy():
    import sys
    made: list[Path] = []

    def make(stem: str, overlays_src: str, feature_lookback: str = "None") -> str:
        made.append(_write(stem, overlays_src, feature_lookback))
        return stem

    yield make
    for p in made:
        p.unlink(missing_ok=True)
        sys.modules.pop(f"algua.strategies.momentum.{p.stem}", None)


def test_loader_binds_overlay_fns(tmp_strategy):
    name = tmp_strategy("tmp_ov_ok", f"[OverlaySpec(policy='trailing_stop', params={_TS!r})]", "7")
    strat = load_strategy(name)
    assert strat.overlay_fns == (trailing_stop,)


def test_loader_rejects_unknown_overlay_policy(tmp_strategy):
    name = tmp_strategy("tmp_ov_bad_id", "[OverlaySpec(policy='nope', params={})]")
    with pytest.raises(StrategyNotFound, match="unknown overlay policy 'nope'"):
        load_strategy(name)


def test_loader_rejects_bad_overlay_params(tmp_strategy):
    name = tmp_strategy(
        "tmp_ov_bad_params", "[OverlaySpec(policy='trailing_stop', params={'lookback': 5})]"
    )
    with pytest.raises(StrategyNotFound, match=r"overlay\[0\] 'trailing_stop': missing"):
        load_strategy(name)


def test_loader_rejects_feature_lookback_below_overlay_window(tmp_strategy):
    name = tmp_strategy(
        "tmp_ov_short_lb", f"[OverlaySpec(policy='trailing_stop', params={_TS!r})]", "6"
    )
    with pytest.raises(
        StrategyNotFound, match="feature_lookback 6 is smaller than the longest overlay window 7"
    ):
        load_strategy(name)


# --- approvals closure ------------------------------------------------------------------------

def test_closure_includes_overlays_and_regime_modules():
    names = closure_module_names(load_strategy("cross_sectional_momentum"))
    assert "algua.portfolio.overlays" in names
    assert "algua.portfolio.overlay_policies" in names
    assert "algua.features.regime" in names
