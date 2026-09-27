"""Pure canonical identity values for frozen planner environments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from algua.registry.artifact_contract import (
    ArtifactFile,
    _digest,
    _inventory_payload,
    _require_digest,
    _utf8_nfc,
    canonical_relative_path,
)

BASE_INTERPRETER = "base-interpreter"
MAX_IDENTITY_CHARS = 128
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_PYTHON_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*")
# PEP 503 canonical form: lowercase, every run of "-", "_" or "." collapsed to one hyphen.
_DISTRIBUTION_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_DISTRIBUTION_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*")


def _identity_text(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if (
        not isinstance(value, str) or len(value) > MAX_IDENTITY_CHARS
        or pattern.fullmatch(value) is None
    ):
        raise ValueError(f"{label} is empty or malformed")
    return value


def _typed_tuple(value: Any, kind: type, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple or any(not isinstance(item, kind) for item in value):
        raise ValueError(f"{label} must be a tuple of typed identities")
    return value


@dataclass(frozen=True)
class InterpreterIdentity:
    implementation: str
    version: str
    cache_tag: str
    soabi: str
    platform_tag: str
    os_name: str
    machine: str

    def __post_init__(self) -> None:
        for field, value in self.to_dict().items():
            pattern = _PYTHON_VERSION if field == "version" else _TOKEN
            _identity_text(value, pattern, f"interpreter {field}")

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
        if not isinstance(self.interpreter, InterpreterIdentity):
            raise ValueError("environment key interpreter identity is not typed")
        installer = self.uv_version
        if (
            not isinstance(installer, str) or not installer or len(installer) > MAX_IDENTITY_CHARS
            or not installer.isprintable() or installer != installer.strip()
        ):
            raise ValueError("installer identity must be a non-empty bounded single line")
        for label, argv in (("create", self.create_argv), ("sync", self.sync_argv)):
            if (
                type(argv) is not tuple or not argv
                or any(type(item) is not str or not item for item in argv)
            ):
                raise ValueError(
                    f"installer {label} argv must be a non-empty tuple of non-empty strings")
        # Canonical JSON NFC-normalizes strings: a non-NFC value would share its key digest with a
        # distinct retained value, so it is refused rather than silently aliased. A lone surrogate
        # (not UTF-8 text) fails here too, not later when the key digest is first encoded.
        if any(
            _utf8_nfc(text, "installer identity and argv strings") != text
            for text in (installer, *self.create_argv, *self.sync_argv)
        ):
            raise ValueError("installer identity and argv strings must be NFC-normalized")

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

    def __post_init__(self) -> None:
        _identity_text(self.name, _DISTRIBUTION_NAME, "installed distribution name")
        _identity_text(self.version, _DISTRIBUTION_VERSION, "installed distribution version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class InterpreterLink:
    path: str
    target: str

    def __post_init__(self) -> None:
        canonical_relative_path(self.path, "interpreter link path")
        if self.target != BASE_INTERPRETER:
            canonical_relative_path(self.target, "interpreter link target")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "target": self.target}


@dataclass(frozen=True)
class InstalledInventory:
    distributions: tuple[InstalledDistribution, ...]
    files: tuple[ArtifactFile, ...]
    interpreter_links: tuple[InterpreterLink, ...] = ()

    def __post_init__(self) -> None:
        distributions = _typed_tuple(
            self.distributions, InstalledDistribution, "installed distributions")
        names = [item.name for item in distributions]
        if any(left >= right for left, right in zip(names, names[1:], strict=False)):
            raise ValueError("installed distributions must be uniquely sorted by name")
        links = _typed_tuple(self.interpreter_links, InterpreterLink, "interpreter links")
        paths = [item.path.encode() for item in links]
        if any(left >= right for left, right in zip(paths, paths[1:], strict=False)):
            raise ValueError("interpreter links must be uniquely sorted by UTF-8 path")
        _inventory_payload(_typed_tuple(self.files, ArtifactFile, "installed files"))

    @property
    def digest(self) -> str:
        return _digest(
            "algua.frozen-installed-inventory",
            {"distributions": [item.to_dict() for item in self.distributions],
             "files": _inventory_payload(self.files),
             "interpreter_links": [item.to_dict() for item in self.interpreter_links]},
        )


@dataclass(frozen=True)
class EnvironmentDescriptor:
    key: EnvironmentKey
    inventory_digest: str
    interpreter: InterpreterIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.key, EnvironmentKey):
            raise ValueError("environment descriptor key is not typed")
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
