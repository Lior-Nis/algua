"""An int written into a float field of ``ExecutionContract``/``CapacityLimit`` is stored as the
equal float (Story 1.3c review).

Two serializers read those fields: ``asdict`` feeds ``config_hash`` and ``model_dump(mode="json")``
records a frozen deployment's config. The first kept an int ``1`` while the second wrote ``1.0``,
so the strict frozen decoder's re-hash could never match a config that wrote a whole number into a
float field, and such a candidate was refused at every intake. Normalising at construction makes
``1`` and ``1.0`` one identity everywhere. It must not re-identify anything that exists: the golden
below pins every repo strategy's config hash as it was before the change.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from algua.contracts.types import CapacityLimit, ExecutionContract
from algua.primitives.module_refresh import serialized_import
from algua.strategies.base import StrategyConfig, strategy_config_hash
from algua.strategies.loader import _index

# Every repo strategy's config hash, computed on the tree immediately BEFORE the normalisation
# (feat/story-1-3c-frozen-paper at 1ed5c22). Strategy modules are additions-only, so a pin only
# moves if a human deliberately edits that strategy; a strategy added later is simply not pinned.
_GOLDEN_CONFIG_HASHES = {
    "cadenced_mad_relative_strength": "70c9a5552628f9e16f482ca1873b0882",
    "cadenced_sharpe_rank_buffer": "31af862ca18a407cac500d48b226e0ec",
    "cadenced_tail_risk_relative_strength": "db7ef2d2eb8a445534e42c3bccd2f9d9",
    "calm_four_day_rebound": "73fa0535642eae8c58b9435f485225a0",
    "cross_horizon_low_vol_consensus": "1439efc33cbb81bed6b7ac06c4ae4638",
    "cross_sectional_momentum": "5e4f08591d72dcf08117abaa75a87bb0",
    "distributed_gains_quality_momentum": "53c630572753e26acd0a39f58809a8b3",
    "distributed_loss_peer_selloff_rebound": "28916e887ddce65677e1f59ed0b83f57",
    "dual_horizon_skip_month_persistence": "e6542693f13fa1c5889dc2501778906f",
    "fundamentals_earnings_tilt": "9dfcfc242bba59ddf9e98d9a1515c546",
    "lagged_rank_persistence_momentum": "5239fe88830da81983c274e98b172635",
    "liquidity_stable_quality_momentum": "fb424c2ba4a23203bc7c66ceed1285a6",
    "market_residual_low_volatility": "b70b0ed446f574e712845d37e1426e0b",
    "model_linear_scores": "dbee045a0c4652aa1d0b15d378e3370e",
    "model_scaled_linear": "040c2ea4608275eb46a67664b9e60d08",
    "momentum_regime_stop": "c267c3cd0483d963cd734bdd57ac15ba",
    "news_coverage_tilt": "eb06346b6374f2d10437da02286a4bdf",
    "orderly_six_day_rebound": "d8d8d4994202bc1a258d874d360eb9c1",
    "path_efficiency_quality_momentum": "61116477f695354c09ac858f26869161",
    "peer_median_oversold_rebound": "d413e75e50e7113ee2ba3b69ebc9daae",
    "robust_dispersion_low_volatility": "95e63d9c3490b35303ba045d90d37b5c",
    "shock_veto_peer_selloff_rebound": "4e0070175e2fd31931327e7365772d00",
    "thresholded_peer_selloff_conviction": "b718083496d59a6a993a1da583ef1aeb",
    "trend_fit_quality_momentum": "86ca721f41724b450424001a55ef13f3",
    "ulcer_quiet_momentum_quality": "80a5029b1d36c0f1d092269773d60224",
}

_EXECUTION_FLOATS = {
    "max_gross_exposure": 1, "max_weight_per_symbol": 1, "fees": 0, "slippage": 0,
    "target_gross_utilization": 1,
}
_CAPACITY_FLOATS = {"reference_aum": 1_000_000, "max_participation_rate": 1}
_CAPACITY = {"reference_aum": 1e6, "max_participation_rate": 0.1, "adv_window_bars": 20}


def _config(execution: ExecutionContract) -> StrategyConfig:
    return StrategyConfig(
        name="s", universe=["AAPL", "MSFT"], execution=execution, construction="top_n")


def test_every_existing_repo_strategy_keeps_its_config_hash():
    """The normalisation re-identifies nothing that exists: no repo strategy wrote an int into a
    float field, so every pinned hash (computed before the change) is reproduced exactly."""
    index = _index()
    present = sorted(set(_GOLDEN_CONFIG_HASHES) & set(index))
    assert present, "no pinned repo strategy is loadable"
    actual = {name: strategy_config_hash(serialized_import(index[name]).CONFIG)
              for name in present}
    assert actual == {name: _GOLDEN_CONFIG_HASHES[name] for name in present}


def test_the_normalised_fields_are_exactly_the_float_typed_fields():
    assert {f.name for f in dataclasses.fields(ExecutionContract) if f.type == "float"} == set(
        _EXECUTION_FLOATS)
    assert {f.name for f in dataclasses.fields(CapacityLimit) if f.type == "float"} == set(
        _CAPACITY_FLOATS)


@pytest.mark.parametrize(("field", "value"), sorted(_EXECUTION_FLOATS.items()))
def test_an_int_in_an_execution_float_field_is_stored_as_the_equal_float(field, value):
    contract = ExecutionContract(rebalance_frequency="1d", **{field: value})
    stored = getattr(contract, field)
    assert type(stored) is float and stored == value
    assert contract == ExecutionContract(rebalance_frequency="1d", **{field: float(value)})
    assert type(dataclasses.asdict(contract)[field]) is float
    assert strategy_config_hash(_config(contract)) == strategy_config_hash(
        _config(ExecutionContract(rebalance_frequency="1d", **{field: float(value)})))


@pytest.mark.parametrize(("field", "value"), sorted(_CAPACITY_FLOATS.items()))
def test_an_int_in_a_capacity_float_field_is_stored_as_the_equal_float(field, value):
    capacity = CapacityLimit(**{**_CAPACITY, field: value})
    stored = getattr(capacity, field)
    assert type(stored) is float and stored == value
    execution = ExecutionContract(rebalance_frequency="1d", capacity=capacity)
    assert type(dataclasses.asdict(execution)["capacity"][field]) is float
    assert strategy_config_hash(_config(execution)) == strategy_config_hash(_config(
        ExecutionContract(rebalance_frequency="1d",
                          capacity=CapacityLimit(**{**_CAPACITY, field: float(value)}))))


def test_the_recorded_config_and_the_hash_payload_agree_on_whole_numbers():
    """The root cause, pinned: ``asdict`` and ``model_dump(mode="json")`` now both hold 1.0."""
    execution = ExecutionContract(
        rebalance_frequency="1d", max_gross_exposure=2, fees=0,
        capacity=CapacityLimit(reference_aum=1_000_000, max_participation_rate=1,
                               adv_window_bars=20))
    dumped = _config(execution).model_dump(mode="json")["execution"]
    assert json.dumps(dataclasses.asdict(execution), sort_keys=True) == json.dumps(
        dumped, sort_keys=True)


def test_int_fields_are_untouched():
    contract = ExecutionContract(rebalance_frequency="1d", decision_lag_bars=2, warmup_bars=3)
    assert type(contract.decision_lag_bars) is int and type(contract.warmup_bars) is int
    assert type(CapacityLimit(**_CAPACITY).adv_window_bars) is int


@pytest.mark.parametrize("kwargs", [
    {"fees": True}, {"slippage": True}, {"target_gross_utilization": True},
])
def test_a_bool_is_never_normalised_and_its_rail_still_refuses_it(kwargs):
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1d", **kwargs)


@pytest.mark.parametrize("field", ["reference_aum", "max_participation_rate"])
def test_a_bool_capacity_value_is_still_refused(field):
    with pytest.raises(ValueError):
        CapacityLimit(**{**_CAPACITY, field: True})


@pytest.mark.parametrize("kwargs", [
    {"fees": -1}, {"slippage": -1}, {"max_weight_per_symbol": 0},
    {"target_gross_utilization": 0}, {"target_gross_utilization": 2},
])
def test_existing_rails_still_refuse_int_values(kwargs):
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1d", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"reference_aum": 0}, {"reference_aum": -5}, {"max_participation_rate": 0},
    {"max_participation_rate": 2},
])
def test_existing_capacity_rails_still_refuse_int_values(kwargs):
    with pytest.raises(ValueError):
        CapacityLimit(**{**_CAPACITY, **kwargs})


def test_an_int_too_large_for_a_float_is_refused_as_a_value_error():
    with pytest.raises(ValueError, match="max_gross_exposure"):
        ExecutionContract(rebalance_frequency="1d", max_gross_exposure=10**400)
    with pytest.raises(ValueError, match="reference_aum"):
        CapacityLimit(**{**_CAPACITY, "reference_aum": 10**400})
