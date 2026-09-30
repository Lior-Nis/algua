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
FROZEN_WIRE_NAME = "frozen-planner"
FROZEN_WIRE_VERSION = 1
FROZEN_WIRE = {"name": FROZEN_WIRE_NAME, "version": FROZEN_WIRE_VERSION}
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_FILES = 10_000
MAX_BUNDLE_FILES = MAX_SOURCE_FILES + 2  # exported source plus the two generated files
# Traversal bound on implied bundle directories (the repository's source tree has 36): a
# verifier counts every directory before retaining it, so empty-directory fanout cannot grow
# memory without ever reaching the file bound.
MAX_BUNDLE_DIRECTORIES = 10_000
MAX_PATH_BYTES = 1_024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_FILE_MODES = frozenset({"100644", "100755"})


def _utf8_nfc(value: str, label: str) -> str:
    """NFC form of ``value``; a lone surrogate is not UTF-8 text and fails as a plain ValueError."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} must be valid UTF-8 text") from None
    return unicodedata.normalize("NFC", value)


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return _utf8_nfc(value, "canonical JSON text")
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
            normalized = _utf8_nfc(key, "canonical JSON text")
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


def canonical_relative_path(value: Any, label: str = "artifact path") -> str:
    """Return ``value`` only if it is already a canonical portable relative POSIX path."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if len(encoded) > MAX_PATH_BYTES or "\0" in value or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{label} is not NFC-normalized")
    segments = value.split("/")
    if (
        _DRIVE.match(segments[0])
        or any(segment in {"", ".", ".."} for segment in segments)
        or any(segment.endswith((".", " ")) for segment in segments)
    ):
        raise ValueError(f"{label} is unsafe")
    return value


def _require_count(value: Any, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an exact integer count within its bound")
    return value


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    mode: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        canonical_relative_path(self.path)
        if not isinstance(self.mode, str) or self.mode not in _FILE_MODES:
            raise ValueError("artifact mode is unsupported")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("artifact size must be an exact non-negative integer")
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
        # Every bound precedes building or hashing the inventory payload.
        _require_count(len(files), "bundle file count", MAX_BUNDLE_FILES)
        if any(item.size > MAX_FILE_BYTES for item in files):
            raise ValueError("bundle file exceeds the per-file size bound")
        if sum(item.size for item in files) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds the aggregate size bound")
        digest = _digest("algua.frozen-bundle", _inventory_payload(files))
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
        _require_count(self.file_count, "bundle file count", MAX_BUNDLE_FILES)
        _require_count(self.total_bytes, "bundle byte count", MAX_BUNDLE_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest, "locator": self.locator,
            "inventory_digest": self.inventory_digest, "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }
