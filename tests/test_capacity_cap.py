from __future__ import annotations

import pandas as pd
import pytest

from algua.contracts.types import CapacityLimit
from algua.portfolio.construction import ConstructionError, _dollar_adv, apply_capacity_cap


def _view(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows = [(timestamp_iso, symbol, close, volume), ...] -> a bar-schema-shaped view.

    Only symbol/close/volume are read by the capacity cap; the other schema columns are filled with
    stand-in values so the frame is realistically shaped."""
    ts = pd.to_datetime([r[0] for r in rows], utc=True)
    frame = pd.DataFrame(
        {
            "symbol": [r[1] for r in rows],
            "open": [r[2] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[2] for r in rows],
            "adj_close": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    return frame


# reference_aum=1000, rate=0.1, window=2 => max_notional = 0.1 * dollar_adv; max_weight = that/1000.
_CAP = CapacityLimit(reference_aum=1000.0, max_participation_rate=0.1, adv_window_bars=2)


def test_caps_oversized_long():
    # A: two bars of close*vol=1000 => adv=1000 => max_weight = 0.1*1000/1000 = 0.1.
    view = _view([("2024-01-01", "A", 10.0, 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)
    assert capped["A"] == pytest.approx(0.1)


def test_under_budget_weight_untouched():
    view = _view([("2024-01-01", "A", 10.0, 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    # 0.05 < max_weight 0.1 => unchanged.
    capped = apply_capacity_cap(pd.Series({"A": 0.05}), view, _CAP)
    assert capped["A"] == pytest.approx(0.05)


def test_scales_proportional_to_adv():
    # A adv=1000 -> max 0.1 ; B adv=2000 -> max 0.2. Both requested 0.5 -> capped to their budgets.
    view = _view(
        [
            ("2024-01-01", "A", 10.0, 100.0),
            ("2024-01-02", "A", 10.0, 100.0),
            ("2024-01-01", "B", 20.0, 100.0),
            ("2024-01-02", "B", 20.0, 100.0),
        ]
    )
    capped = apply_capacity_cap(pd.Series({"A": 0.5, "B": 0.5}), view, _CAP)
    assert capped["A"] == pytest.approx(0.1)
    assert capped["B"] == pytest.approx(0.2)


def test_zero_volume_name_forced_flat():
    # dollar_adv = 0 => budget 0 => weight forced to 0 (fail closed, no divide-by-zero).
    view = _view([("2024-01-01", "A", 10.0, 0.0), ("2024-01-02", "A", 10.0, 0.0)])
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)
    assert capped["A"] == 0.0


def test_missing_symbol_forced_flat():
    # Weight for C, but C has no bars in the view => no established ADV => forced flat.
    view = _view([("2024-01-01", "A", 10.0, 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    capped = apply_capacity_cap(pd.Series({"A": 0.05, "C": 0.5}), view, _CAP)
    assert capped["A"] == pytest.approx(0.05)
    assert capped["C"] == 0.0


def test_short_history_forced_flat():
    # A has only 1 bar but window=2 => no full window => forced flat.
    view = _view([("2024-01-02", "A", 10.0, 100.0)])
    capped = apply_capacity_cap(pd.Series({"A": 0.05}), view, _CAP)
    assert capped["A"] == 0.0


def test_preserves_short_sign():
    view = _view([("2024-01-01", "A", 10.0, 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    capped = apply_capacity_cap(pd.Series({"A": -0.5}), view, _CAP)
    assert capped["A"] == pytest.approx(-0.1)


def test_only_trailing_window_feeds_adv():
    # Older bars are huge; the most recent `window`=2 bars are small. ADV must use only the recent 2
    # (mean dollar = 500), so max_weight = 0.1*500/1000 = 0.05, capping a 0.5 request to 0.05.
    view = _view(
        [
            ("2024-01-01", "A", 10.0, 10_000.0),  # old, must be excluded
            ("2024-01-02", "A", 10.0, 10_000.0),  # old, must be excluded
            ("2024-01-03", "A", 10.0, 50.0),      # dollar 500
            ("2024-01-04", "A", 10.0, 50.0),      # dollar 500
        ]
    )
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)
    assert capped["A"] == pytest.approx(0.05)


def test_noop_when_nothing_exceeds():
    view = _view(
        [
            ("2024-01-01", "A", 10.0, 100.0),
            ("2024-01-02", "A", 10.0, 100.0),
            ("2024-01-01", "B", 20.0, 100.0),
            ("2024-01-02", "B", 20.0, 100.0),
        ]
    )
    w = pd.Series({"A": 0.05, "B": 0.05})
    capped = apply_capacity_cap(w, view, _CAP)
    assert capped.to_dict() == pytest.approx(w.to_dict())


def test_empty_weights_unchanged():
    view = _view([("2024-01-01", "A", 10.0, 100.0)])
    capped = apply_capacity_cap(pd.Series(dtype="float64"), view, _CAP)
    assert len(capped) == 0


@pytest.mark.parametrize("missing", ["close", "volume"])
def test_view_missing_column_fails_closed(missing: str):
    view = _view([("2024-01-01", "A", 10.0, 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    view = view.drop(columns=[missing])
    with pytest.raises(ConstructionError):
        apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)


def test_empty_view_with_weights_forces_flat():
    # An empty view (no history) with a non-empty weight vector => every name un-sized => flat.
    empty = _view([]).iloc[0:0]
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), empty, _CAP)
    assert capped["A"] == 0.0


def test_nan_in_window_fails_closed():
    # A NaN dollar value inside the trailing window must fail closed (no partial-sample ADV): the
    # symbol is omitted and its weight forced flat, NOT sized off the one remaining finite bar.
    view = _view([("2024-01-01", "A", float("nan"), 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    assert "A" not in _dollar_adv(view, 2).index
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)
    assert capped["A"] == 0.0


def test_inf_in_window_fails_closed():
    view = _view([("2024-01-01", "A", float("inf"), 100.0), ("2024-01-02", "A", 10.0, 100.0)])
    assert "A" not in _dollar_adv(view, 2).index
    capped = apply_capacity_cap(pd.Series({"A": 0.5}), view, _CAP)
    assert capped["A"] == 0.0


def test_dollar_adv_omits_short_history_symbols():
    view = _view(
        [
            ("2024-01-01", "A", 10.0, 100.0),
            ("2024-01-02", "A", 10.0, 100.0),
            ("2024-01-02", "B", 10.0, 100.0),  # only 1 bar, window=2
        ]
    )
    adv = _dollar_adv(view, 2)
    assert "A" in adv.index
    assert "B" not in adv.index
    assert adv["A"] == pytest.approx(1000.0)


# --- CapacityLimit validation (fail-closed) ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_aum": 0.0, "max_participation_rate": 0.1, "adv_window_bars": 2},
        {"reference_aum": -1.0, "max_participation_rate": 0.1, "adv_window_bars": 2},
        {"reference_aum": float("nan"), "max_participation_rate": 0.1, "adv_window_bars": 2},
        {"reference_aum": float("inf"), "max_participation_rate": 0.1, "adv_window_bars": 2},
        {"reference_aum": 1000.0, "max_participation_rate": 0.0, "adv_window_bars": 2},
        {"reference_aum": 1000.0, "max_participation_rate": 1.5, "adv_window_bars": 2},
        {"reference_aum": 1000.0, "max_participation_rate": float("nan"), "adv_window_bars": 2},
        {"reference_aum": 1000.0, "max_participation_rate": 0.1, "adv_window_bars": 0},
        {"reference_aum": 1000.0, "max_participation_rate": 0.1, "adv_window_bars": True},
        # bool is an int subtype: True would masquerade as 1 on a numeric field and fail OPEN.
        {"reference_aum": True, "max_participation_rate": 0.1, "adv_window_bars": 2},
        {"reference_aum": 1000.0, "max_participation_rate": True, "adv_window_bars": 2},
    ],
)
def test_capacity_limit_rejects_bad_values(kwargs: dict):
    with pytest.raises(ValueError):
        CapacityLimit(**kwargs)


def test_capacity_limit_accepts_valid():
    cap = CapacityLimit(reference_aum=1_000_000.0, max_participation_rate=0.05, adv_window_bars=21)
    assert cap.max_participation_rate == 0.05


# --- integration: identity + engine/live parity --------------------------------------------------


def _strat(capacity: CapacityLimit | None):
    from typing import Any

    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(1.0, index=adj.index, columns=adj.columns)  # always want 100% AAA

    cfg = StrategyConfig(
        name="cap_test", universe=["AAA"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0, capacity=capacity,
        ),
        params={}, construction="passthrough",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series([1.0], index=["AAA"]),
        signal_panel_fn=signal_panel,
        construct_fn=lambda scores, view, params: scores,  # passthrough: scores ARE weights
    )


def test_config_hash_changes_with_capacity_and_stable_when_none():
    from algua.strategies.base import config_hash

    base = _strat(None)
    with_cap = _strat(
        CapacityLimit(reference_aum=1e9, max_participation_rate=0.1, adv_window_bars=5)
    )
    # None default must be identity-neutral vs a freshly-built no-capacity strategy.
    assert config_hash(base) == config_hash(_strat(None))
    # Declaring capacity must change identity (invalidates any prior live approval).
    assert config_hash(base) != config_hash(with_cap)


def test_backtest_loop_and_fast_path_apply_cap_identically():
    from datetime import UTC, datetime

    from algua.backtest._sample import SyntheticProvider
    from algua.backtest.engine import _decision_weights, _fast_weights

    start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)
    # reference_aum huge vs synthetic dollar-ADV (~100*1e6=1e8) so the 100%-AAA target caps.
    cap = CapacityLimit(reference_aum=1e10, max_participation_rate=0.1, adv_window_bars=5)
    strat = _strat(cap)
    bars = SyntheticProvider(seed=2).get_bars(["AAA"], start, end, "1d")
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    loop = _decision_weights(strat, bars, adj)
    fast = _fast_weights(strat, bars, adj)
    # Parity: the cap is applied through the same chokepoint in both paths.
    pd.testing.assert_frame_equal(loop, fast)
    # The cap actually fired: an evaluated bar (>= window) is well below the uncapped 1.0 target.
    evaluated = loop["AAA"].iloc[5:]
    assert (evaluated > 0.0).any()
    assert (evaluated < 1.0).all()


def test_live_decide_applies_same_cap():
    from datetime import UTC, datetime

    from algua.backtest._sample import SyntheticProvider
    from algua.live.paper_loop import decide

    cap = CapacityLimit(reference_aum=1e10, max_participation_rate=0.1, adv_window_bars=5)
    strat = _strat(cap)
    start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)
    bars = SyntheticProvider(seed=2).get_bars(["AAA"], start, end, "1d").sort_index()
    weights, _intents = decide(strat, bars, {}, end)
    # decide() runs the full pipeline incl. the cap: the 100%-AAA target is capacity-limited.
    assert weights["AAA"] < 1.0
    assert weights["AAA"] > 0.0
