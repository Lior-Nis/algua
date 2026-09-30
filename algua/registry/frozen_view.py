"""The supervisor's view of a frozen tenant, decoded strictly from its recorded config.

Story 1.3c §2. The paper supervisor never imports a frozen tenant's strategy module; everything it
reads (name, gate universe, feature lookback, execution contract) comes from the descriptor's
RECORDED ``resolved_config`` — ``StrategyConfig.model_dump(mode="json")`` as Story 1.3b stored it.

That record cannot be validated naively: ``execution`` is a mapping there, and ``StrategyConfig``
refuses a raw mapping by design (#344), because pydantic would coerce nested values before the
``ExecutionContract``/``CapacityLimit`` ``__post_init__`` rails run. So the decoder checks exact
JSON types first (never a boolean as a number, never a float where an integer is required), builds
``CapacityLimit`` and ``ExecutionContract`` through their constructors (the rails run), then builds
``StrategyConfig`` strictly. The result must dump back to exactly the recorded bytes and hash, from
the config alone, to the descriptor's ``config_hash``. Any failure is
``frozen_content_unsupported``. The gate universe is overlaid afterwards, as
``prepare_paper_runtime`` does for a loaded strategy.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from algua.contracts.canonical import canonical_json
from algua.contracts.types import CapacityLimit, ExecutionContract
from algua.portfolio.overlays import OverlaySpec
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_tenant_errors import FrozenTenantUnsupported
from algua.strategies.base import StrategyConfig, strategy_config_hash

Check = Callable[[Any], bool]


def _is_str(value: Any) -> bool:
    return type(value) is str


def _is_int(value: Any) -> bool:
    return type(value) is int  # exact: a JSON ``true`` (bool) or ``5.0`` (float) is refused


def _is_number(value: Any) -> bool:
    # Any JSON number, never a boolean. Whether an integer may stand for a float is decided by the
    # exact dump-back below: ``model_dump(mode="json")`` always writes a float field as a float.
    return type(value) in (int, float)


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _is_object(value: Any) -> bool:
    return type(value) is dict


def _is_symbols(value: Any) -> bool:
    return type(value) is list and all(type(item) is str for item in value)


def _is_optional_int(value: Any) -> bool:
    return value is None or type(value) is int


def _is_false(value: Any) -> bool:
    # Frozen paper runs bars-only planners: a fundamentals, news or model lane is unsupported.
    return value is False


def _is_null(value: Any) -> bool:
    return value is None  # model assets are unsupported in frozen content (Story 1.3b)


EXECUTION_FIELDS: dict[str, Check] = {
    "rebalance_frequency": _is_str, "decision_lag_bars": _is_int, "allow_fractional": _is_bool,
    "max_gross_exposure": _is_number, "max_weight_per_symbol": _is_number,
    "allow_short": _is_bool, "warmup_bars": _is_int, "fees": _is_number,
    "slippage": _is_number, "fill_price": _is_str, "target_gross_utilization": _is_number,
}
CAPACITY_FIELDS: dict[str, Check] = {
    "reference_aum": _is_number, "max_participation_rate": _is_number,
    "adv_window_bars": _is_int,
}
CONFIG_FIELDS: dict[str, Check] = {
    "name": _is_str, "universe": _is_symbols, "execution": _is_object, "params": _is_object,
    "construction": _is_str, "construction_params": _is_object,
    "overlays": lambda value: type(value) is list, "needs_fundamentals": _is_false,
    "needs_news": _is_false, "needs_model": _is_false, "model_ref": _is_null,
    "feature_lookback": _is_optional_int,
}
_OVERLAY_FIELDS: dict[str, Check] = {"policy": _is_str, "params": _is_object}


def _fields(value: Any, schema: dict[str, Check], label: str) -> dict[str, Any]:
    """Exactly ``schema``'s keys, each of exactly its JSON type."""
    if type(value) is not dict or set(value) != set(schema):
        raise FrozenTenantUnsupported(f"{label} has unknown or missing fields")
    for key, check in schema.items():
        if not check(value[key]):
            raise FrozenTenantUnsupported(f"{label}.{key} is malformed or unsupported")
    return dict(value)


def _execution(value: Any) -> ExecutionContract:
    record = _fields(value, {**EXECUTION_FIELDS, "capacity": lambda _value: True}, "execution")
    raw_capacity = record.pop("capacity")
    capacity = None
    if raw_capacity is not None:
        capacity = CapacityLimit(**_fields(raw_capacity, CAPACITY_FIELDS, "execution.capacity"))
    return ExecutionContract(**record, capacity=capacity)


def _config(value: Any) -> StrategyConfig:
    record = _fields(value, CONFIG_FIELDS, "config")
    record["execution"] = _execution(record["execution"])
    record["overlays"] = [
        OverlaySpec.model_validate(_fields(item, _OVERLAY_FIELDS, "overlay"), strict=True)
        for item in record["overlays"]
    ]
    return StrategyConfig.model_validate(record, strict=True)


def decode_frozen_config(manifest: FrozenManifest) -> StrategyConfig:
    """The descriptor's recorded config as a ``StrategyConfig``, byte-exact and hash-bound."""
    recorded = manifest.resolved_config
    try:
        # Decode a private copy so the view can never alias (and mutate) the descriptor's record.
        config = _config(json.loads(canonical_json(recorded)))
        dumped = canonical_json(config.model_dump(mode="json"))
        digest = strategy_config_hash(config)
        expected = canonical_json(recorded)
    except FrozenTenantUnsupported:
        raise
    except (TypeError, ValueError) as exc:  # a constructor rail or strict validation refused it
        raise FrozenTenantUnsupported("recorded config violates a contract rail") from exc
    if dumped != expected:
        raise FrozenTenantUnsupported("decoded config does not dump back to the recorded config")
    if digest != manifest.config_hash:
        raise FrozenTenantUnsupported("decoded config does not match the descriptor config hash")
    return config


@dataclass(frozen=True)
class FrozenStrategyView:
    """Exactly what the paper supervisor reads from a tenant, and nothing else: the tick loop
    reads ``name``/``universe``, cycle planning ``config.feature_lookback`` and
    ``execution.{warmup_bars,capacity}``. ``universe`` is the GATE universe."""

    config: StrategyConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> tuple[str, ...]:
        return tuple(self.config.universe)

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution


def overlay_gate_universe(
    config: StrategyConfig, gate_universe: Sequence[str],
) -> FrozenStrategyView:
    """Overlay the gate-bound universe exactly as ``prepare_paper_runtime`` does."""
    universe = list(gate_universe)
    if universe != config.universe:
        config = config.model_copy(update={"universe": universe})
    return FrozenStrategyView(config)


def decode_frozen_view(
    manifest: FrozenManifest, gate_universe: Sequence[str],
) -> FrozenStrategyView:
    return overlay_gate_universe(decode_frozen_config(manifest), gate_universe)
