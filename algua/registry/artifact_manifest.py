"""Strict parser for canonical frozen artifact manifests."""
from __future__ import annotations

import json
from typing import Any

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.registry.artifact_contract import (
    DESCRIPTOR_VERSION,
    FROZEN_WIRE,
    MAX_MANIFEST_BYTES,
    PLANNER_BOUNDARY_VERSION,
    BundleDescriptor,
    EnvironmentDescriptor,
    EnvironmentKey,
    InterpreterIdentity,
    _require_keys,
    _require_str,
    canonical_json,
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
    return InterpreterIdentity(**{key: _require_str(item, key) for key, item in raw.items()})


def _parse_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("installer argv must be a non-empty string list")
    return tuple(value)


def parse_frozen_manifest(raw: str) -> FrozenManifest:
    if len(raw.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("frozen manifest exceeds the canonical size bound")
    try:
        payload = json.loads(raw, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError("frozen manifest is not valid JSON") from exc
    if canonical_json(payload) != raw:
        raise ValueError("frozen manifest bytes are not canonical")
    root = _require_keys(payload, {
        "descriptor_version", "source_kind", "source_ref", "identity", "resolved_config",
        "universe_name", "bundle", "environment", "assets", "planner_protocol_version",
        "planner_boundary_version", "frozen_wire",
    }, "frozen manifest")
    if (root["descriptor_version"] != DESCRIPTOR_VERSION or root["source_kind"] != "frozen"
            or root["assets"] != [] or root["planner_protocol_version"] != PLANNER_PROTOCOL_VERSION
            or root["planner_boundary_version"] != PLANNER_BOUNDARY_VERSION
            or root["frozen_wire"] != FROZEN_WIRE):
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
        uv_version=_require_str(key_raw["uv_version"], "uv version"),
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
    resolved_config = root["resolved_config"]
    if not isinstance(resolved_config, dict):
        raise ValueError("resolved config must be an object")
    universe_name = root["universe_name"]
    if universe_name is not None and not isinstance(universe_name, str):
        raise ValueError("universe name must be a string or null")
    return FrozenManifest(
        source_ref=root["source_ref"], code_hash=identity["code_hash"],
        config_hash=identity["config_hash"], dependency_hash=identity["dependency_hash"],
        resolved_config=resolved_config, universe_name=universe_name,
        bundle=BundleDescriptor(**bundle_raw), environment=environment,
    )
