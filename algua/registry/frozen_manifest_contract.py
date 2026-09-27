"""Pure canonical outer descriptor for a recoverable frozen planner."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.registry.artifact_contract import (
    _HEX32,
    _OID,
    DESCRIPTOR_VERSION,
    FROZEN_WIRE,
    MAX_MANIFEST_BYTES,
    PLANNER_BOUNDARY_VERSION,
    BundleDescriptor,
    _digest,
    _require_digest,
    canonical_json,
)
from algua.registry.environment_contract import EnvironmentDescriptor


@dataclass(frozen=True)
class FrozenManifest:
    source_ref: str
    code_hash: str
    config_hash: str
    dependency_hash: str
    resolved_config: dict[str, Any]
    universe_name: str | None
    bundle: BundleDescriptor
    environment: EnvironmentDescriptor

    def __post_init__(self) -> None:
        _require_digest(self.source_ref, "source ref", _OID)
        _require_digest(self.code_hash, "code hash", _HEX32)
        _require_digest(self.config_hash, "config hash", _HEX32)
        _require_digest(self.dependency_hash, "dependency hash")
        if not isinstance(self.resolved_config, dict):
            raise ValueError("resolved config must be an object")
        if self.universe_name is not None:
            if not isinstance(self.universe_name, str):
                raise ValueError("universe name must be a string or null")
            # Retain the canonical-JSON form so the ledger projection cannot disagree with it.
            object.__setattr__(
                self, "universe_name", unicodedata.normalize("NFC", self.universe_name))
        if not isinstance(self.bundle, BundleDescriptor) or not isinstance(
            self.environment, EnvironmentDescriptor,
        ):
            raise ValueError("frozen manifest bundle and environment must be typed descriptors")
        if self.environment.key.dependency_hash != self.dependency_hash:
            raise ValueError("environment dependency hash disagrees")
        normalized = json.loads(canonical_json(self.resolved_config))
        object.__setattr__(self, "resolved_config", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_version": DESCRIPTOR_VERSION,
            "source_kind": "frozen",
            "source_ref": self.source_ref,
            "identity": {
                "code_hash": self.code_hash,
                "config_hash": self.config_hash,
                "dependency_hash": self.dependency_hash,
            },
            "resolved_config": self.resolved_config,
            "universe_name": self.universe_name,
            "bundle": self.bundle.to_dict(),
            "environment": self.environment.to_dict(),
            "assets": [],
            "planner_protocol_version": PLANNER_PROTOCOL_VERSION,
            "planner_boundary_version": PLANNER_BOUNDARY_VERSION,
            "frozen_wire": FROZEN_WIRE,
        }

    @property
    def json(self) -> str:
        raw = canonical_json(self.to_dict())
        if len(raw.encode()) > MAX_MANIFEST_BYTES:
            raise ValueError("frozen manifest exceeds the canonical size bound")
        return raw

    @property
    def digest(self) -> str:
        _ = self.json
        return _digest("algua.frozen-manifest", self.to_dict())
