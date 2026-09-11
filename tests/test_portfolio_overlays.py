"""The overlay seam: tighten-only invariants, validation, resolution, and trailing_stop."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from algua.portfolio.overlays import (
    OVERLAY_POLICIES,
    OverlayError,
    OverlaySpec,
    apply_overlays,
    get_overlay_policy,
    overlay_lookback,
    resolve_overlays,
    trailing_stop,
    validate_overlay_params,
)


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = []
    for sym, path in prices.items():
        for t, px in zip(ts, path, strict=True):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


_W = pd.Series({"A": 0.5, "B": 0.5})
_V = _view({"A": [1.0, 1.0], "B": [1.0, 1.0]})


# --- invariants -------------------------------------------------------------------------------

def _spec(policy: str = "trailing_stop") -> OverlaySpec:
    return OverlaySpec(policy=policy, params={})


def test_apply_overlays_rejects_added_symbol():
    def adds(w: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.concat([w, pd.Series({"C": 0.1})])
    with pytest.raises(OverlayError, match=r"overlay\[0\] 'trailing_stop' added symbol"):
        apply_overlays(_W, _V, [_spec()], [adds])


def test_apply_overlays_rejects_scale_up():
    def up(w, view, params):
        return w * 1.5
    with pytest.raises(OverlayError, match=r"overlay\[0\].*increased"):
        apply_overlays(_W, _V, [_spec()], [up])


def test_apply_overlays_rejects_sign_flip():
    def flip(w, view, params):
        return -w
    with pytest.raises(OverlayError, match=r"overlay\[0\].*flipped"):
        apply_overlays(_W, _V, [_spec()], [flip])


def test_apply_overlays_rejects_non_finite():
    def nan(w, view, params):
        out = w.copy()
        out["A"] = float("nan")
        return out
    with pytest.raises(OverlayError, match=r"overlay\[0\].*non-finite"):
        apply_overlays(_W, _V, [_spec()], [nan])


def test_apply_overlays_names_the_failing_index_in_a_chain():
    ident = lambda w, view, params: w  # noqa: E731
    up = lambda w, view, params: w * 2.0  # noqa: E731
    with pytest.raises(OverlayError, match=r"overlay\[1\]"):
        apply_overlays(_W, _V, [_spec(), _spec()], [ident, up])


def test_apply_overlays_accepts_zeroing_and_dropping():
    def zero_a_drop_b(w, view, params):
        return pd.Series({"A": 0.0})
    out = apply_overlays(_W, _V, [_spec()], [zero_a_drop_b])
    assert out.to_dict() == {"A": 0.0}


def test_apply_overlays_empty_weights_short_circuits():
    def boom(w, view, params):
        raise AssertionError("must not be called on empty weights")
    empty = pd.Series(dtype="float64")
    assert apply_overlays(empty, _V, [_spec()], [boom]).empty


def test_apply_overlays_requires_one_fn_per_spec():
    with pytest.raises(OverlayError, match="one resolved fn per spec"):
        apply_overlays(_W, _V, [_spec()], [])


# --- registry + validation ---------------------------------------------------------------------

def test_registry_is_read_only_and_lists_trailing_stop():
    assert "trailing_stop" in OVERLAY_POLICIES
    with pytest.raises(TypeError):
        OVERLAY_POLICIES["x"] = None  # type: ignore[index]


def test_get_overlay_policy_unknown_id():
    with pytest.raises(OverlayError, match="unknown overlay policy 'nope'"):
        get_overlay_policy("nope")


@pytest.mark.parametrize(
    "params, msg",
    [
        ({"lookback": 20, "stop_pct": 0.1}, "missing"),
        ({"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 0, "x": 1}, "unknown"),
        ({"lookback": 0, "stop_pct": 0.1, "cooldown_bars": 0}, "lookback"),
        ({"lookback": True, "stop_pct": 0.1, "cooldown_bars": 0}, "lookback"),
        ({"lookback": 20, "stop_pct": 1.0, "cooldown_bars": 0}, "stop_pct"),
        ({"lookback": 20, "stop_pct": 0.0, "cooldown_bars": 0}, "stop_pct"),
        ({"lookback": 20, "stop_pct": float("nan"), "cooldown_bars": 0}, "non-finite"),
        ({"lookback": 20, "stop_pct": 0.1, "cooldown_bars": -1}, "cooldown_bars"),
    ],
)
def test_validate_trailing_stop_params_fail_closed(params, msg):
    with pytest.raises(OverlayError, match=msg):
        validate_overlay_params("trailing_stop", params)


def test_validate_trailing_stop_params_ok():
    validate_overlay_params("trailing_stop", {"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3})


def test_overlay_lookback_trailing_stop():
    spec = OverlaySpec(
        policy="trailing_stop", params={"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3}
    )
    assert overlay_lookback(spec) == 23


def test_resolve_overlays_binds_fns_and_checks_feature_lookback():
    spec = OverlaySpec(
        policy="trailing_stop", params={"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3}
    )
    fns = resolve_overlays([spec], feature_lookback=None)
    assert fns == (trailing_stop,)
    assert resolve_overlays([spec], feature_lookback=23) == (trailing_stop,)
    with pytest.raises(
        OverlayError, match="feature_lookback 22 is smaller than the longest overlay window 23"
    ):
        resolve_overlays([spec], feature_lookback=22)
    assert resolve_overlays([], feature_lookback=0) == ()


def test_resolve_overlays_validates_params():
    with pytest.raises(OverlayError, match="overlay\\[0\\] 'trailing_stop': missing"):
        resolve_overlays([OverlaySpec(policy="trailing_stop", params={})], feature_lookback=None)


# --- trailing_stop -----------------------------------------------------------------------------

_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def test_trailing_stop_zeroes_a_name_below_its_rolling_high():
    # A: peak 100 then 85 (-15% off the 5-bar high) -> stopped. B flat -> kept.
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})
    out = trailing_stop(pd.Series({"A": 0.5, "B": 0.5}), view, _TS)
    assert out.to_dict() == {"A": 0.0, "B": 0.5}


def test_trailing_stop_keeps_a_name_within_tolerance():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 92.0]})  # -8% off the high < 10%
    out = trailing_stop(pd.Series({"A": 1.0}), view, _TS)
    assert out["A"] == 1.0


def test_trailing_stop_cooldown_keeps_it_out_after_recovery():
    # Breach at bar 4 (85 vs high 100), recovers to 99 by bar 6. cooldown_bars=2 -> bars 5,6 out.
    path = [90.0, 100.0, 98.0, 95.0, 85.0, 99.0, 99.0]
    for end, expect in ((5, 0.0), (6, 0.0), (7, 0.0)):
        view = _view({"A": path[:end]})
        assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == expect
    # One more bar and the breach (bar 4) is outside the last cooldown_bars+1 bars; the 5-bar
    # high no longer holds 100 either -> back in.
    view = _view({"A": path + [99.0]})
    assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == 1.0


def test_trailing_stop_cooldown_zero_is_only_the_current_bar():
    path = [90.0, 100.0, 98.0, 95.0, 85.0, 99.0]
    params = {**_TS, "cooldown_bars": 0}
    # bar 5: price 99 vs 5-bar high 100 -> within tolerance -> kept, even though bar 4 breached.
    assert trailing_stop(pd.Series({"A": 1.0}), _view({"A": path}), params)["A"] == 1.0


def test_trailing_stop_short_history_uses_available_bars():
    view = _view({"A": [100.0, 85.0]})  # only 2 bars, lookback 5 -> high = 100 -> stopped
    assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == 0.0


def test_trailing_stop_passes_through_a_symbol_absent_from_view():
    view = _view({"A": [100.0] * 5})
    out = trailing_stop(pd.Series({"A": 0.5, "Z": 0.5}), view, _TS)
    assert out.to_dict() == {"A": 0.5, "Z": 0.5}


def test_trailing_stop_preserves_short_sign():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0]})
    out = trailing_stop(pd.Series({"A": -0.5}), view, _TS)
    assert out["A"] == 0.0  # a short is stopped the same way (weight -> 0, never flipped)


def test_trailing_stop_through_apply_overlays_satisfies_invariants():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})
    spec = OverlaySpec(policy="trailing_stop", params=_TS)
    fns = resolve_overlays([spec], feature_lookback=None)
    out = apply_overlays(pd.Series({"A": 0.5, "B": 0.5}), view, [spec], fns)
    assert out.to_dict() == {"A": 0.0, "B": 0.5}


# --- regime_gate --------------------------------------------------------------------------------

from algua.portfolio import overlay_policies  # noqa: E402
from algua.portfolio.overlays import regime_gate  # noqa: E402

# Small windows so a 60-bar synthetic path can traverse the states. Vol/turbulence stresses are
# disabled by absurd thresholds unless a test enables them.
_RG = {
    "trend_window": 10, "dd_window": 10, "dd_threshold": 0.05,
    "turb_window": 5, "z_window": 10, "turb_z": 1e9,
    "persistence": 2,
    "shock_window": 3, "shock_return": 0.9, "fast_turb_z": 1e9, "fast_lookback": 1,
    "neutral_exposure": 0.6, "risk_off_exposure": 0.2, "fast_exposure": 0.25,
}
_W2 = pd.Series({"A": 0.5, "B": 0.5})


def _uni(path: list[float]) -> pd.DataFrame:
    """A 3-symbol universe that all follow `path` (turbulence is then degenerate -> NaN -> off)."""
    return _view({"A": path, "B": [p * 2 for p in path], "C": [p * 3 for p in path]})


def _up(n: int, start: float = 100.0, step: float = 0.002) -> list[float]:
    return [start * (1 + step) ** i for i in range(n)]


def test_regime_gate_risk_on_leaves_weights_untouched():
    out = regime_gate(_W2, _uni(_up(40)), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_risk_off_after_trend_and_drawdown_persist():
    path = _up(40) + [_up(40)[-1] * (0.98 ** i) for i in range(1, 11)]  # -18% over 10 bars
    out = regime_gate(_W2, _uni(path), _RG)  # trend below mean AND dd < -5% for >= 2 bars
    assert out.to_dict() == pytest.approx({"A": 0.1, "B": 0.1})


def test_regime_gate_neutral_on_a_single_stress():
    # 30 flat bars at 100, then 3 bars at 97: a -3% drawdown (under the 5% threshold, so NOT a
    # drawdown stress) but the level sits below its 10-bar mean (a trend stress) for 3 bars
    # >= persistence 2 -> exactly one stress -> neutral_exposure.
    path = [100.0] * 30 + [97.0] * 3
    out = regime_gate(_W2, _uni(path), _RG)
    assert out["A"] == pytest.approx(0.5 * 0.6)


def test_regime_gate_persistence_blocks_a_one_bar_state():
    # One bar of stress (last bar only) with persistence=2 -> the prior risk-on run still stands.
    path = _up(40) + [_up(40)[-1] * 0.90]
    out = regime_gate(_W2, _uni(path), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_fast_shock_applies_fast_exposure():
    params = {**_RG, "shock_window": 1, "shock_return": 0.05, "fast_lookback": 2}
    # -10% shock, then flat: within last 2
    path = _up(40) + [_up(40)[-1] * 0.90, _up(40)[-1] * 0.90]
    out = regime_gate(_W2, _uni(path), params)
    # slow state: trend+dd stressed for 2 bars -> 0.2; fast 0.25 -> min = 0.2
    assert out["A"] == pytest.approx(0.5 * 0.2)
    params2 = {**params, "risk_off_exposure": 0.6, "neutral_exposure": 0.6}
    out2 = regime_gate(_W2, _uni(path), params2)
    assert out2["A"] == pytest.approx(0.5 * 0.25)  # now fast is the binding one


def test_regime_gate_volatility_stress_via_turbulence():
    rng = np.random.default_rng(3)
    n = 60
    rets = rng.normal(0.0, 0.005, size=(n, 3))
    rets[-1] = 0.08  # a 16-sigma common shock on the LAST bar only (its trailing window is calm)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    view = _view({s: prices[:, i].tolist() for i, s in enumerate(["A", "B", "C"])})
    params = {**_RG, "turb_z": 3.0, "persistence": 1, "trend_window": 3, "dd_window": 3,
              "dd_threshold": 0.99}  # trend can't be below a 3-bar mean while rising; dd disabled
    out = regime_gate(_W2, view, params)
    assert out["A"] == pytest.approx(0.5 * 0.6)  # exactly one stress (volatility) -> neutral


def test_regime_gate_no_op_on_short_history():
    out = regime_gate(_W2, _uni([100.0, 99.0, 98.0]), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_empty_view_is_a_noop():
    # An empty view would otherwise hit `.iloc[-1]` on an empty Series and raise IndexError.
    view = _uni([100.0]).iloc[0:0]
    out = regime_gate(_W2, view, _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_rejects_turb_window_not_exceeding_symbol_count():
    # _uni() is a 3-symbol universe: turb_window must exceed 3, else the trailing turbulence
    # covariance can never be full rank (fail closed and LOUD, not a silently-off volatility leg).
    with pytest.raises(OverlayError, match="turb_window"):
        regime_gate(_W2, _uni(_up(40)), {**_RG, "turb_window": 3})
    # 4 > 3 symbols -> a full-rank covariance is possible -> no raise.
    regime_gate(_W2, _uni(_up(40)), {**_RG, "turb_window": 4})


def test_regime_gate_persistence_search_bounded_to_regime_search_bars(monkeypatch):
    # A hand-computed, deterministic 12-bar level path (all bars flat at level 1.0 except where
    # noted below; level == price / price[0] since the universe is proportional, see `_uni`):
    #   idx  0  1  2  3  4  5     6     7      8      9     10     11
    #   lvl  1  1  1  1  1  1  0.95  0.94  1.00  0.95  1.00  0.95
    # trend_window=3 (min_periods=3): trend[t] = level[t] < mean(level[t-2:t+1]).
    #   idx 2..5: flat -> False.  idx 6: 0.95 < mean(1,1,0.95)=0.9833 -> True.
    #   idx 7: 0.94 < mean(1,0.95,0.94)=0.963 -> True.              (2-bar stress RUN at 6,7)
    #   idx 8: 1.00 < mean(0.95,0.94,1.00)=0.963 -> False.
    #   idx 9: 0.95 < mean(0.94,1.00,0.95)=0.963 -> True.
    #   idx 10: 1.00 < mean(1.00,0.95,1.00)=0.983 -> False.
    #   idx 11: 0.95 < mean(0.95,1.00,0.95)=0.967 -> True.
    # -> scores at idx 8,9,10,11 are 0,1,0,1: the last REGIME_SEARCH_BARS=4 bars oscillate, so
    # persistence=2 is never satisfied inside a 4-bar horizon, while the persistence-2 run at
    # 6,7 sits just outside it. dd/vol are disabled (absurd threshold / dd_threshold=0.99), so
    # score == trend alone.
    levels = [1.0] * 6 + [0.95, 0.94, 1.00, 0.95, 1.00, 0.95]
    path = [100.0 * lv for lv in levels]
    view = _uni(path)
    params = {
        **_RG, "trend_window": 3, "dd_window": 3, "dd_threshold": 0.99, "z_window": 3,
        "persistence": 2,
    }
    spec = OverlaySpec(policy="regime_gate", params=params)
    # max(trend 3, dd 3, turb 5 + z 3, shock 3 + fast 1) + persistence 2 = 10, regardless of the
    # search horizon (the horizon does not change the declared lookback formula).
    expected_lookback = 10

    # Default REGIME_SEARCH_BARS (63): the 6,7 run is well inside the horizon -> picked up ->
    # neutral_exposure (0.6) applies.
    assert overlay_lookback(spec) == expected_lookback
    out_default = regime_gate(_W2, view, params)
    assert out_default["A"] == pytest.approx(0.5 * 0.6)

    # Patch the horizon down to 4 bars: now only idx 8..11 (the oscillation) are searched, so no
    # persistence-2 run is visible -> risk-on (weights untouched) even though a valid run exists
    # 6 bars back.
    monkeypatch.setattr(overlay_policies, "REGIME_SEARCH_BARS", 4)
    assert overlay_lookback(spec) == expected_lookback  # lookback still unaffected
    out_bounded = regime_gate(_W2, view, params)
    assert out_bounded.to_dict() == _W2.to_dict()


def test_regime_gate_lookback():
    spec = OverlaySpec(policy="regime_gate", params=_RG)
    # max(trend 10, dd 10, turb 5 + z 10, shock 3 + fast 1) + persistence 2 = 17
    assert overlay_lookback(spec) == 17


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"persistence": 11}, "persistence"),
        ({"risk_off_exposure": 0.7}, "risk_off_exposure"),
        ({"dd_threshold": 1.0}, "dd_threshold"),
        ({"turb_z": 0.0}, "turb_z"),
        ({"fast_exposure": 1.5}, "fast_exposure"),
        ({"trend_window": 0}, "trend_window"),
        ({"persistence": 64, "dd_window": 70}, "REGIME_SEARCH_BARS"),
        ({"fast_lookback": 64}, "REGIME_SEARCH_BARS"),
    ],
)
def test_validate_regime_gate_domains(over, msg):
    with pytest.raises(OverlayError, match=msg):
        validate_overlay_params("regime_gate", {**_RG, **over})


def test_validate_regime_gate_ok():
    validate_overlay_params("regime_gate", _RG)
