"""Pure canonical values for recoverable frozen planner artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

DESCRIPTOR_VERSION = 1
PLANNER_BOUNDARY_VERSION = 1
FROZEN_WIRE = {"name": "frozen-planner", "version": 1}
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in result:
                raise ValueError("canonical JSON keys collide after normalization")
            result[normalized] = _normalized(item)
        return result
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalized(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _digest(domain: str, payload: Any) -> str:
    preimage = {"domain": domain, "version": "1", "payload": payload}
    return hashlib.sha256(canonical_json(preimage).encode()).hexdigest()


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unknown or missing fields")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_digest(value: Any, label: str, pattern: re.Pattern[str] = _HEX64) -> str:
    text = _require_str(value, label)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{label} is not canonical hexadecimal")
    return text


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    mode: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or unicodedata.normalize("NFC", self.path) != self.path:
            raise ValueError("artifact path must be non-empty NFC")
        if self.mode not in {"100644", "100755"}:
            raise ValueError("artifact mode is unsupported")
        if isinstance(self.size, bool) or self.size < 0:
            raise ValueError("artifact size must be non-negative")
        _require_digest(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "mode": self.mode, "size": self.size, "sha256": self.sha256}


def _inventory_payload(files: tuple[ArtifactFile, ...]) -> list[dict[str, Any]]:
    paths = [entry.path.encode() for entry in files]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("artifact inventory must be uniquely sorted by UTF-8 path")
    return [entry.to_dict() for entry in files]


@dataclass(frozen=True)
class BuildInputs:
    files: tuple[ArtifactFile, ...]

    @property
    def digest(self) -> str:
        return _digest("algua.frozen-build-inputs", _inventory_payload(self.files))


@dataclass(frozen=True)
class BundleDescriptor:
    digest: str
    locator: str
    inventory_digest: str
    file_count: int
    total_bytes: int

    @classmethod
    def from_files(cls, files: tuple[ArtifactFile, ...]) -> BundleDescriptor:
        payload = _inventory_payload(files)
        if any(item.size > MAX_FILE_BYTES for item in files):
            raise ValueError("bundle file exceeds the per-file size bound")
        if sum(item.size for item in files) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds the aggregate size bound")
        digest = _digest("algua.frozen-bundle", payload)
        return cls(
            digest=digest,
            locator=f"frozen/bundles/sha256/{digest[:2]}/{digest}",
            inventory_digest=digest,
            file_count=len(files),
            total_bytes=sum(item.size for item in files),
        )

    def __post_init__(self) -> None:
        _require_digest(self.digest, "bundle digest")
        _require_digest(self.inventory_digest, "bundle inventory digest")
        expected = f"frozen/bundles/sha256/{self.digest[:2]}/{self.digest}"
        if self.locator != expected or self.inventory_digest != self.digest:
            raise ValueError("bundle locator or inventory digest disagrees")
        counts = (self.file_count, self.total_bytes)
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("bundle counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest, "locator": self.locator,
            "inventory_digest": self.inventory_digest, "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class InterpreterIdentity:
    implementation: str
    version: str
    cache_tag: str
    soabi: str
    platform_tag: str
    os_name: str
    machine: str

    def to_dict(self) -> dict[str, str]:
        return {
            "implementation": self.implementation, "version": self.version,
            "cache_tag": self.cache_tag, "soabi": self.soabi,
            "platform_tag": self.platform_tag, "os_name": self.os_name,
            "machine": self.machine,
        }


@dataclass(frozen=True)
class EnvironmentKey:
    build_inputs_digest: str
    dependency_hash: str
    interpreter: InterpreterIdentity
    uv_version: str
    create_argv: tuple[str, ...]
    sync_argv: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest(self.build_inputs_digest, "build inputs digest")
        _require_digest(self.dependency_hash, "dependency hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_inputs_digest": self.build_inputs_digest,
            "dependency_hash": self.dependency_hash,
            "interpreter": self.interpreter.to_dict(), "uv_version": self.uv_version,
            "create_argv": list(self.create_argv), "sync_argv": list(self.sync_argv),
        }

    @property
    def digest(self) -> str:
        return _digest("algua.frozen-environment-key", self.to_dict())


@dataclass(frozen=True)
class InstalledDistribution:
    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class InterpreterLink:
    path: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "target": self.target}


@dataclass(frozen=True)
class InstalledInventory:
    distributions: tuple[InstalledDistribution, ...]
    files: tuple[ArtifactFile, ...]
    interpreter_links: tuple[InterpreterLink, ...] = ()

    @property
    def digest(self) -> str:
        distributions = [item.to_dict() for item in self.distributions]
        if distributions != sorted(distributions, key=lambda item: (item["name"], item["version"])):
            raise ValueError("installed distributions must be sorted")
        links = [item.to_dict() for item in self.interpreter_links]
        if links != sorted(links, key=lambda item: item["path"]):
            raise ValueError("interpreter links must be sorted")
        return _digest(
            "algua.frozen-installed-inventory",
            {"distributions": distributions, "files": _inventory_payload(self.files),
             "interpreter_links": links},
        )


@dataclass(frozen=True)
class EnvironmentDescriptor:
    key: EnvironmentKey
    inventory_digest: str
    interpreter: InterpreterIdentity

    def __post_init__(self) -> None:
        _require_digest(self.inventory_digest, "installed inventory digest")
        if self.key.interpreter != self.interpreter:
            raise ValueError("verified interpreter disagrees with environment key")

    @property
    def digest(self) -> str:
        return _digest("algua.frozen-environment", self.identity_payload())

    @property
    def locator(self) -> str:
        return f"frozen/environments/sha256/{self.digest[:2]}/{self.digest}"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(), "key_digest": self.key.digest,
            "inventory_digest": self.inventory_digest,
            "interpreter": self.interpreter.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "digest": self.digest, "locator": self.locator}
