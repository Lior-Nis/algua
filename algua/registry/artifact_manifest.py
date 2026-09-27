"""Strict parser for canonical frozen artifact manifests."""
from __future__ import annotations

import json
from typing import Any

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.registry.artifact_contract import (
    DESCRIPTOR_VERSION,
    FROZEN_WIRE,
    FROZEN_WIRE_NAME,
    FROZEN_WIRE_VERSION,
    MAX_MANIFEST_BYTES,
    PLANNER_BOUNDARY_VERSION,
    BundleDescriptor,
    _require_keys,
    canonical_json,
)
from algua.registry.environment_contract import (
    EnvironmentDescriptor,
    EnvironmentKey,
    InterpreterIdentity,
)
from algua.registry.frozen_manifest_contract import FrozenManifest


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_interpreter(value: Any) -> InterpreterIdentity:
    raw = _require_keys(value, {
        "implementation", "version", "cache_tag", "soabi", "platform_tag", "os_name", "machine",
    }, "interpreter identity")
    return InterpreterIdentity(**raw)


def _parse_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("installer argv must be a list")
    return tuple(value)


def _require_version(value: Any, expected: int, label: str) -> None:
    """Exact integer equality: JSON ``true`` and ``1.0`` compare equal to ``1`` in Python."""
    if type(value) is not int or value != expected:
        raise ValueError(f"unsupported frozen manifest {label}")


def _load_canonical(raw: Any) -> Any:
    if not isinstance(raw, str):
        raise ValueError("frozen manifest must be text")
    if len(raw.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("frozen manifest exceeds the canonical size bound")
    try:
        payload = json.loads(raw, object_pairs_hook=_no_duplicates)
        canonical = canonical_json(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("frozen manifest is not valid JSON") from exc
    except RecursionError as exc:
        raise ValueError("frozen manifest nesting is too deep") from exc
    if canonical != raw:
        raise ValueError("frozen manifest bytes are not canonical")
    return payload


def parse_frozen_manifest(raw: str) -> FrozenManifest:
    root = _require_keys(_load_canonical(raw), {
        "descriptor_version", "source_kind", "source_ref", "identity", "resolved_config",
        "universe_name", "bundle", "environment", "assets", "planner_protocol_version",
        "planner_boundary_version", "frozen_wire",
    }, "frozen manifest")
    _require_version(root["descriptor_version"], DESCRIPTOR_VERSION, "descriptor version")
    _require_version(
        root["planner_protocol_version"], PLANNER_PROTOCOL_VERSION, "planner protocol version")
    _require_version(
        root["planner_boundary_version"], PLANNER_BOUNDARY_VERSION, "planner boundary version")
    wire = _require_keys(root["frozen_wire"], set(FROZEN_WIRE), "frozen wire identity")
    _require_version(wire["version"], FROZEN_WIRE_VERSION, "frozen wire version")
    if (root["source_kind"] != "frozen" or root["assets"] != []
            or wire["name"] != FROZEN_WIRE_NAME):
        raise ValueError("unsupported frozen manifest version or lane")
    identity = _require_keys(root["identity"], {"code_hash", "config_hash", "dependency_hash"},
                             "artifact identity")
    bundle_raw = _require_keys(root["bundle"], {
        "digest", "locator", "inventory_digest", "file_count", "total_bytes",
    }, "bundle descriptor")
    environment_raw = _require_keys(root["environment"], {
        "key", "key_digest", "inventory_digest", "interpreter", "digest", "locator",
    }, "environment descriptor")
    key_raw = _require_keys(environment_raw["key"], {
        "build_inputs_digest", "dependency_hash", "interpreter", "uv_version", "create_argv",
        "sync_argv",
    }, "environment key")
    interpreter = _parse_interpreter(environment_raw["interpreter"])
    key = EnvironmentKey(
        build_inputs_digest=key_raw["build_inputs_digest"],
        dependency_hash=key_raw["dependency_hash"],
        interpreter=_parse_interpreter(key_raw["interpreter"]),
        uv_version=key_raw["uv_version"],
        create_argv=_parse_argv(key_raw["create_argv"]),
        sync_argv=_parse_argv(key_raw["sync_argv"]),
    )
    if environment_raw["key_digest"] != key.digest:
        raise ValueError("environment key digest disagrees")
    environment = EnvironmentDescriptor(
        key=key, inventory_digest=environment_raw["inventory_digest"], interpreter=interpreter)
    environment_disagrees = (
        environment_raw["digest"] != environment.digest
        or environment_raw["locator"] != environment.locator
    )
    if environment_disagrees:
        raise ValueError("environment descriptor disagrees")
    return FrozenManifest(
        source_ref=root["source_ref"], code_hash=identity["code_hash"],
        config_hash=identity["config_hash"], dependency_hash=identity["dependency_hash"],
        resolved_config=root["resolved_config"], universe_name=root["universe_name"],
        bundle=BundleDescriptor(**bundle_raw), environment=environment,
    )
