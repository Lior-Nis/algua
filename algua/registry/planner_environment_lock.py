"""Committed-lock policy for frozen planner environments.

The committed `uv.lock` is authoritative. Every locked registry package must carry a canonical
name and version, and every locked wheel exactly one canonical HTTPS URL identity, so duplicate
wheel ownership and outage evidence are keyed by a single spelling of each wheel.
"""
from __future__ import annotations

import re
import tomllib
from typing import Any

from algua.registry.environment_contract import InstalledDistribution
from algua.registry.planner_environment_errors import EnvironmentIncompatible

_HOST_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_WHEEL_URL = re.compile(
    rf"https://(?P<host>{_HOST_LABEL}(?:\.{_HOST_LABEL})*)(?::(?P<port>[1-9][0-9]{{0,4}}))?"
    r"(?P<path>/.*)"
)
# Unreserved characters appear raw and every other byte as an uppercase escape, so each path
# character has exactly one spelling (RFC 3986 section 6.2.2); `?` and `#` can never appear, so
# no query or fragment can either, and every segment is non-empty.
_PATH_SEGMENT = re.compile(r"(?:[A-Za-z0-9\-._~]|%[0-9A-F]{2})+")
_UNRESERVED = re.compile(r"[A-Za-z0-9\-._~]")
_ESCAPE = re.compile(r"%([0-9A-F]{2})")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_HOST_CHARS = 253
_MAX_PORT = 65535
_HTTPS_DEFAULT_PORT = 443


def canonical_wheel_url(url: str) -> str:
    """Return ``url`` only if it is exactly one canonical HTTPS wheel URL identity.

    Canonical: lowercase `https` scheme and lowercase DNS hostname, no credentials, query or
    fragment, an explicit port only if it is valid and not the default 443, and a path of
    non-empty, non-dot segments whose unreserved characters are raw and every other byte an
    uppercase percent escape, ending in `.whl`. Whitespace, controls and non-ASCII text cannot
    appear, and two accepted URLs that differ are distinct identities.
    """
    match = _WHEEL_URL.fullmatch(url)
    if match is None:
        raise EnvironmentIncompatible("locked wheel URL is not a canonical HTTPS URL")
    host, port, path = match.group("host", "port", "path")
    if len(host) > _MAX_HOST_CHARS:
        raise EnvironmentIncompatible("locked wheel URL hostname is too long")
    if port is not None and (int(port) > _MAX_PORT or int(port) == _HTTPS_DEFAULT_PORT):
        raise EnvironmentIncompatible("locked wheel URL port is invalid or the default")
    segments = path[1:].split("/")
    if any(
        segment in {".", ".."} or _PATH_SEGMENT.fullmatch(segment) is None
        for segment in segments
    ):
        raise EnvironmentIncompatible("locked wheel URL path is not canonical")
    if any(_UNRESERVED.fullmatch(chr(int(code, 16))) for code in _ESCAPE.findall(path)):
        raise EnvironmentIncompatible("locked wheel URL escapes an unreserved character")
    if not segments[-1].endswith(".whl"):
        raise EnvironmentIncompatible("locked wheel URL does not name a wheel")
    return url


def locked_wheels(raw: bytes) -> dict[str, tuple[str, str]]:
    """Validate the committed lock and map every locked wheel URL to its canonical identity.

    Total over arbitrary bytes: any malformation is `EnvironmentIncompatible`. Each registry
    package must carry a PEP 503 canonical name and a bounded version, and no wheel URL may be
    locked for two packages.
    """
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EnvironmentIncompatible("committed uv lock is invalid") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise EnvironmentIncompatible("committed uv lock has no package inventory")
    result: dict[str, tuple[str, str]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise EnvironmentIncompatible("committed uv lock package is invalid")
        source = package.get("source")
        name: Any = package.get("name")
        if name == "algua" and source == {"editable": "."}:
            continue
        if not isinstance(source, dict) or set(source) != {"registry"}:
            raise EnvironmentIncompatible(
                "local, editable, URL and VCS dependencies are unsupported")
        version: Any = package.get("version")
        try:
            identity = InstalledDistribution(name, version)
        except ValueError as exc:
            raise EnvironmentIncompatible(
                "locked package name or version is not canonical") from exc
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or not wheels:
            raise EnvironmentIncompatible("every locked registry package requires a wheel")
        for wheel in wheels:
            if not isinstance(wheel, dict):
                raise EnvironmentIncompatible("locked wheel metadata is invalid")
            url = wheel.get("url")
            digest = wheel.get("hash")
            if not isinstance(url, str) or not isinstance(digest, str):
                raise EnvironmentIncompatible("locked wheel URL or hash is not canonical")
            wheel_url = canonical_wheel_url(url)
            if _HASH.fullmatch(digest) is None:
                raise EnvironmentIncompatible("locked wheel hash is not canonical")
            if wheel_url in result:
                raise EnvironmentIncompatible("a wheel URL is locked for more than one package")
            result[wheel_url] = (identity.name, identity.version)
    return result


def validate_lock(raw: bytes) -> None:
    locked_wheels(raw)
