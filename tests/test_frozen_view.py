"""Story 1.3c §2: the strict decoder that turns a frozen descriptor's RECORDED config into the
supervisor's view, without importing the tenant's strategy module.

The recorded config is ``model_dump(mode="json")``, so ``execution`` is always a mapping, which
``StrategyConfig`` refuses by design (#344). The decoder must rebuild ``CapacityLimit`` and
``ExecutionContract`` through their constructors, reproduce the recorded object exactly and hash to
the descriptor's ``config_hash``; anything else is ``frozen_content_unsupported``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys

import pytest

from algua.contracts.canonical import canonical_json
from algua.contracts.types import CapacityLimit, ExecutionContract
from algua.portfolio.overlays import OverlaySpec
from algua.registry.frozen_tenant_errors import FrozenTenantUnsupported
from algua.registry.frozen_view import (
    CAPACITY_FIELDS,
    CONFIG_FIELDS,
    EXECUTION_FIELDS,
    FrozenStrategyView,
    decode_frozen_config,
    decode_frozen_view,
)
from algua.strategies.base import StrategyConfig, config_hash, strategy_config_hash
from algua.strategies.loader import (
    _index,
    _loaded_for_test,
    list_strategies,
    load_tradable_strategy,
)
from tests._deployment_helpers import frozen_manifest

_STOP = {"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 2}


def _config(**overrides) -> StrategyConfig:
    fields = {
        "name": "s", "universe": ["AAPL", "MSFT", "NVDA"],
        "execution": ExecutionContract(
            rebalance_frequency="1d", warmup_bars=5, max_weight_per_symbol=0.5,
            capacity=CapacityLimit(
                reference_aum=1_000_000.0, max_participation_rate=0.1, adv_window_bars=20),
        ),
        "params": {"lookback": 20, "alpha": 0.5, "flag": True, "tags": ["a", "b"]},
        "construction": "top_k_equal_weight", "construction_params": {"top_k": 2},
        "overlays": [OverlaySpec(policy="trailing_stop", params=_STOP)],
        "feature_lookback": 30,
    }
    fields.update(overrides)
    return StrategyConfig(**fields)


def _recorded(config: StrategyConfig) -> dict:
    """Exactly what Story 1.3b records: the canonical JSON form of ``model_dump(mode="json")``."""
    return json.loads(canonical_json(config.model_dump(mode="json")))


def _manifest(recorded: dict, *, digest: str | None = None, universe_name: str | None = None):
    return frozen_manifest(
        code_hash="b" * 32, config_hash=digest if digest is not None else "c" * 32,
        dependency_hash="d" * 64, resolved_config=recorded, universe_name=universe_name,
    )


def _valid_manifest(config: StrategyConfig | None = None):
    config = config if config is not None else _config()
    return _manifest(_recorded(config), digest=config_hash(_loaded_for_test(config)))


def _mutated(mutate) -> object:
    """A manifest whose recorded config is ``mutate``d while keeping the ORIGINAL config hash."""
    config = _config()
    recorded = copy.deepcopy(_recorded(config))
    mutate(recorded)
    return _manifest(recorded, digest=config_hash(_loaded_for_test(config)))


# --- the recorded config cannot be validated naively (readiness B1) ----------------------------


def test_a_naive_model_validate_of_the_recorded_config_is_refused():
    """Why the decoder exists: the #344 guard refuses ``execution`` as a raw mapping."""
    with pytest.raises(ValueError, match="raw mapping"):
        StrategyConfig.model_validate(_recorded(_config()))


# --- round trips -------------------------------------------------------------------------------


def test_decoder_reproduces_a_config_with_capacity_and_overlays_exactly():
    config = _config()
    manifest = _valid_manifest(config)

    decoded = decode_frozen_config(manifest)

    assert decoded.model_dump(mode="json") == config.model_dump(mode="json")
    assert isinstance(decoded.execution, ExecutionContract)
    assert decoded.execution == config.execution
    assert isinstance(decoded.execution.capacity, CapacityLimit)
    assert decoded.overlays == config.overlays
    assert strategy_config_hash(decoded) == manifest.config_hash


def _repo_strategy_names() -> list[str]:
    return list_strategies()


@pytest.mark.parametrize("name", _repo_strategy_names())
def test_decoder_round_trips_every_loadable_repo_strategy(name):
    """The real CONFIG of every tradable strategy in the repo decodes, dumps back to exactly its
    recorded form, and hashes (config-only, no module) to the loaded strategy's ``config_hash``."""
    try:
        loaded = load_tradable_strategy(name)
    except (LookupError, ValueError) as exc:  # non-bar lanes / unresolvable models: not tradable
        pytest.skip(f"not a tradable strategy: {type(exc).__name__}")
    recorded = _recorded(loaded.config)
    manifest = _manifest(recorded, digest=config_hash(loaded))

    view = decode_frozen_view(manifest, loaded.universe)

    assert canonical_json(view.config.model_dump(mode="json")) == canonical_json(recorded)
    assert strategy_config_hash(view.config) == config_hash(loaded)
    assert view.name == name
    assert view.universe == tuple(loaded.universe)
    assert view.execution == loaded.execution
    assert view.config.feature_lookback == loaded.config.feature_lookback


@pytest.mark.parametrize("execution", [
    dict(max_gross_exposure=1),
    dict(fees=0, slippage=0),
    dict(max_weight_per_symbol=1, target_gross_utilization=1),
    dict(capacity=dict(reference_aum=1_000_000, max_participation_rate=0.05, adv_window_bars=20)),
    dict(capacity=dict(reference_aum=1e6, max_participation_rate=1, adv_window_bars=20)),
], ids=["gross", "zero-costs", "weight-and-utilization", "int-aum", "int-rate"])
def test_decoder_round_trips_a_config_authored_with_whole_numbers(execution):
    """A whole number written into a float field (``max_gross_exposure=1``) is an ordinary way to
    author a strategy. The descriptor's hash and its recorded config must describe the same value,
    so such a candidate decodes instead of being refused at every intake."""
    execution = dict(execution)  # never mutate the shared parameter
    capacity = execution.pop("capacity", None)
    config = _config(execution=ExecutionContract(
        rebalance_frequency="1d", **execution,
        capacity=CapacityLimit(**capacity) if capacity is not None else None))
    manifest = _manifest(_recorded(config), digest=strategy_config_hash(config))

    decoded = decode_frozen_config(manifest)

    assert decoded.execution == config.execution
    assert canonical_json(decoded.model_dump(mode="json")) == canonical_json(
        manifest.resolved_config)
    assert strategy_config_hash(decoded) == manifest.config_hash


def test_at_least_the_readiness_cohort_of_repo_strategies_is_tradable():
    """Guards the parametrized proof above against silently skipping everything."""
    tradable = []
    for name in _repo_strategy_names():
        try:
            load_tradable_strategy(name)
        except (LookupError, ValueError):
            continue
        tradable.append(name)
    assert len(tradable) >= 21


def test_decoding_never_imports_the_strategy_module():
    name = "momentum_regime_stop"
    loaded = load_tradable_strategy(name)
    manifest = _manifest(_recorded(loaded.config), digest=config_hash(loaded))
    dotted = _index()[name]
    sys.modules.pop(dotted, None)

    decode_frozen_view(manifest, loaded.universe)

    assert dotted not in sys.modules


# --- the gate-universe overlay ------------------------------------------------------------------


def test_view_overlays_the_gate_universe_and_leaves_the_record_untouched():
    manifest = _valid_manifest()
    recorded_before = canonical_json(manifest.resolved_config)

    view = decode_frozen_view(manifest, ["MSFT", "GOOGL"])

    assert isinstance(view, FrozenStrategyView)
    assert view.universe == ("MSFT", "GOOGL")
    assert view.config.universe == ["MSFT", "GOOGL"]
    assert view.name == "s"
    assert view.execution is view.config.execution
    assert view.config.feature_lookback == 30
    # Everything but the universe is the recorded config.
    dumped = view.config.model_dump(mode="json")
    assert {k: v for k, v in dumped.items() if k != "universe"} == {
        k: v for k, v in manifest.resolved_config.items() if k != "universe"}
    # The descriptor keeps its template universe: the planner re-hashes against it.
    assert canonical_json(manifest.resolved_config) == recorded_before
    assert manifest.resolved_config["universe"] == ["AAPL", "MSFT", "NVDA"]


def test_view_exposes_exactly_what_the_supervisor_reads():
    """``run_tick`` reads name/universe, ``build_cycle_plan`` reads config.feature_lookback and
    execution.{capacity,warmup_bars}, ``_run_paper_strategy_tick`` reads universe; nothing else."""
    view = decode_frozen_view(_valid_manifest(), ["AAPL"])
    public = {attr for attr in dir(view) if not attr.startswith("_")}
    assert public == {"name", "universe", "config", "execution"}
    assert [field.name for field in dataclasses.fields(view)] == ["config"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.config = view.config
    assert view.execution.warmup_bars == 5
    assert view.execution.capacity is not None
    assert view.execution.capacity.adv_window_bars == 20


# --- the decoder's schema tracks the config schema ----------------------------------------------


def test_decoder_field_tables_match_the_config_schema():
    """A new config/contract field must be taught to the decoder, not silently defaulted."""
    assert set(CONFIG_FIELDS) == set(StrategyConfig.model_fields)
    assert set(EXECUTION_FIELDS) | {"capacity"} == {
        field.name for field in dataclasses.fields(ExecutionContract)}
    assert set(CAPACITY_FIELDS) == {field.name for field in dataclasses.fields(CapacityLimit)}


# --- refusals -----------------------------------------------------------------------------------


def _set(path: tuple[str, ...], value):
    def mutate(recorded: dict) -> None:
        target = recorded
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _drop(path: tuple[str, ...]):
    def mutate(recorded: dict) -> None:
        target = recorded
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return mutate


REFUSALS = {
    # exact JSON types: no boolean as a number, no float where an int is required
    "bool as float (execution)": _set(("execution", "max_gross_exposure"), True),
    "bool as int (execution)": _set(("execution", "warmup_bars"), True),
    "bool as int (capacity)": _set(("execution", "capacity", "adv_window_bars"), True),
    "bool as float (capacity)": _set(("execution", "capacity", "reference_aum"), True),
    "bool as int (feature_lookback)": _set(("feature_lookback",), True),
    "float as int": _set(("execution", "warmup_bars"), 5.0),
    "string as float": _set(("execution", "fees"), "0.0005"),
    "int as bool": _set(("execution", "allow_short"), 0),
    "null as str": _set(("execution", "fill_price"), None),
    "universe not a list": _set(("universe",), "AAPL"),
    "universe member not a str": _set(("universe",), ["AAPL", 1]),
    "params not an object": _set(("params",), ["lookback", 20]),
    "construction_params not an object": _set(("construction_params",), None),
    "overlays not a list": _set(("overlays",), {"policy": "trailing_stop"}),
    "overlay policy not a str": _set(("overlays",), [{"policy": 1, "params": {}}]),
    # a raw execution mapping is rebuilt through the constructors, so their rails still run
    "execution not an object": _set(("execution",), "1d"),
    "execution rail (t->t+1)": _set(("execution", "decision_lag_bars"), 0),
    "execution rail (fill price)": _set(("execution", "fill_price"), "vwap"),
    "capacity rail (participation)": _set(("execution", "capacity", "max_participation_rate"), 2.0),
    "capacity not an object": _set(("execution", "capacity"), [1.0, 0.1, 20]),
    "config rail (negative lookback)": _set(("feature_lookback",), -1),
    # exact key sets at every level
    "extra top-level key": _set(("surprise",), 1),
    "missing top-level key": _drop(("construction_params",)),
    "extra execution key": _set(("execution", "leverage"), 2.0),
    "missing execution key": _drop(("execution", "slippage")),
    "extra capacity key": _set(("execution", "capacity", "extra"), 1),
    "missing capacity key": _drop(("execution", "capacity", "adv_window_bars")),
    "extra overlay key": _set(
        ("overlays",), [{"policy": "trailing_stop", "params": _STOP, "x": 1}]),
    # lanes a bars-only frozen paper tenant cannot run
    "fundamentals lane": _set(("needs_fundamentals",), True),
    "news lane": _set(("needs_news",), True),
    "model lane": _set(("needs_model",), True),
    "model ref": _set(("model_ref",), {"name": "m", "version": 1}),
}


@pytest.mark.parametrize("mutate", list(REFUSALS.values()), ids=list(REFUSALS))
def test_decoder_refuses_every_malformed_recorded_config(mutate):
    with pytest.raises(FrozenTenantUnsupported) as info:
        decode_frozen_view(_mutated(mutate), ["AAPL"])
    assert info.value.code == "frozen_content_unsupported"


def test_decoder_refuses_a_recorded_config_that_is_not_an_object():
    manifest = _valid_manifest()
    object.__setattr__(manifest, "resolved_config", ["not", "an", "object"])
    with pytest.raises(FrozenTenantUnsupported):
        decode_frozen_config(manifest)


def test_decoder_refuses_a_config_hash_mismatch():
    manifest = _manifest(_recorded(_config()), digest="0" * 32)
    with pytest.raises(FrozenTenantUnsupported, match="config hash"):
        decode_frozen_config(manifest)


def test_decoder_refuses_a_record_that_does_not_dump_back_exactly():
    """An int where the schema holds a float is a JSON number, so it decodes and constructs (the
    constructor stores it as ``0.0``, so it even hashes to the descriptor's hash). Only the exact
    dump-back sees that the recorded bytes are not what ``model_dump(mode="json")`` produces
    (``0`` vs ``0.0``)."""
    config = _config(execution=ExecutionContract(rebalance_frequency="1d", fees=0))
    recorded = _recorded(config)
    recorded["execution"]["fees"] = 0
    manifest = _manifest(recorded, digest=config_hash(_loaded_for_test(config)))
    assert manifest.resolved_config["execution"]["fees"] == 0
    assert type(manifest.resolved_config["execution"]["fees"]) is int

    with pytest.raises(FrozenTenantUnsupported, match="dump back"):
        decode_frozen_config(manifest)


def test_refusal_carries_the_stable_code_and_deployment_binding():
    error = FrozenTenantUnsupported("recorded config is malformed", deployment_id=7)
    assert error.code == "frozen_content_unsupported"
    assert error.deployment_id == 7
    assert "recorded config is malformed" in str(error)
