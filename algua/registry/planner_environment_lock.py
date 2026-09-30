"""Committed-lock policy for frozen planner environments.

The committed `uv.lock` is authoritative. Every locked registry package must carry a canonical
name and version, and every locked wheel a valid HTTPS wheel URL. The exact locked spelling is
kept for uv and for outage evidence; a separate RFC 3986 canonical identity is derived only to
decide duplicate wheel ownership, so semantic aliases of one wheel cannot claim two owners.
"""
from __future__ import annotations

import ipaddress
import re
import string
import tomllib
from typing import Any

from algua.registry.environment_contract import InstalledDistribution
from algua.registry.planner_environment_errors import EnvironmentIncompatible

_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_URL_CHARACTERS = _UNRESERVED | frozenset(":/?#[]@!$&'()*+,;=%")
_URL = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://(?P<authority>[^/?#]*)(?P<path>[^?#]*)"
    r"(?:\?(?P<query>[^#]*))?"
)
_ESCAPE = re.compile(r"%(?P<hex>[0-9A-Fa-f]{2})")
_BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PCHAR = r"A-Za-z0-9\-._~!$&'()*+,;=:@%"
_PATH = re.compile(rf"[{_PCHAR}/]*")
_QUERY = re.compile(rf"[{_PCHAR}/?]*")
_IP_LITERAL = re.compile(r"\[(?P<address>[0-9A-Fa-f:.]+)\](?::(?P<port>[0-9]*))?")
_REG_NAME = re.compile(r"(?P<host>[^:\[\]]+)(?::(?P<port>[0-9]*))?")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
# uv's WHATWG URL parser reads a host whose last label is numeric as IPv4 (octal, hex or a
# single integer included), so such a host must be exactly one strict dotted quad.
_NUMERIC_LABEL = re.compile(r"[0-9]+|0[xX][0-9A-Fa-f]*")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_HOST_CHARS = 253
_MAX_PORT = 65535
_HTTPS_DEFAULT_PORT = 443
_WHEEL_SUFFIX = ".whl"


def _normalize_escapes(text: str) -> str:
    """Uppercase every escape and decode escaped unreserved characters; reserved stay escaped."""
    def normalized(match: re.Match[str]) -> str:
        character = chr(int(match["hex"], 16))
        return character if character in _UNRESERVED else f"%{match['hex'].upper()}"

    return _ESCAPE.sub(normalized, text)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 for an absolute path."""
    segments = path.split("/")
    output: list[str] = []
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment in {".", ".."}:
            if segment == ".." and len(output) > 1:
                output.pop()
            if last:
                output.append("")
            continue
        output.append(segment)
    return "/".join(output)


def _hostname(raw: str) -> str:
    host = raw.lower()
    labels = host.split(".")
    if _NUMERIC_LABEL.fullmatch(labels[-1]):
        try:
            return str(ipaddress.IPv4Address(host))
        except ValueError as exc:
            raise EnvironmentIncompatible("locked wheel URL has an ambiguous numeric host") from exc
    if len(host) > _MAX_HOST_CHARS or any(_DNS_LABEL.fullmatch(label) is None for label in labels):
        raise EnvironmentIncompatible("locked wheel URL hostname is invalid")
    return host


def _port(raw: str | None) -> int | None:
    """The explicit non-default port, or None; an empty port is the default.

    ``raw`` is any string of ASCII digits. Leading zeros are insignificant, and more significant
    digits than the largest port has cannot be valid, so the integer conversion never sees an
    arbitrarily long digit string (Python refuses one over 4,300 digits with `ValueError`).
    """
    if not raw:
        return None
    significant = raw.lstrip("0")
    if len(significant) > len(str(_MAX_PORT)):
        raise EnvironmentIncompatible("locked wheel URL port is invalid")
    number = int(significant or "0")
    if not 1 <= number <= _MAX_PORT:
        raise EnvironmentIncompatible("locked wheel URL port is invalid")
    return None if number == _HTTPS_DEFAULT_PORT else number


def _authority(authority: str) -> str:
    literal = _IP_LITERAL.fullmatch(authority)
    if literal is not None:
        try:
            host = f"[{ipaddress.IPv6Address(literal['address']).compressed}]"
        except ValueError as exc:
            raise EnvironmentIncompatible("locked wheel URL IPv6 address is invalid") from exc
        port = _port(literal["port"])
    else:
        name = _REG_NAME.fullmatch(authority)
        if name is None:
            raise EnvironmentIncompatible("locked wheel URL authority is invalid")
        host, port = _hostname(name["host"]), _port(name["port"])
    return host if port is None else f"{host}:{port}"


def wheel_url_identity(url: str) -> str:
    """Validate a locked wheel URL and return its RFC 3986 canonical ownership identity.

    Accepted: any `https` URL with a DNS, strict IPv4 or IPv6 host, an optional valid port, a
    path of RFC 3986 path characters (sub-delimiters included) naming a `.whl` file, and an
    optional query. Refused: credentials, any fragment, raw whitespace, controls or non-ASCII,
    malformed percent escapes, an invalid or ambiguous authority and a non-wheel path (the
    grammars cannot consume `@` in an authority or `#` anywhere, so credentials and fragments
    fail structurally). The
    identity lowercases scheme and host, spells IP addresses canonically, omits the default port,
    uppercases escapes and decodes escaped unreserved characters (never reserved ones), and
    removes dot segments. It is used only for ownership; uv receives the exact locked URL.
    """
    if not url or any(character not in _URL_CHARACTERS for character in url):
        # Checked before any case folding: a non-ASCII host such as KELVIN SIGN lowercases to
        # ASCII and would otherwise alias an ASCII hostname.
        raise EnvironmentIncompatible("locked wheel URL contains a character a URL cannot carry")
    if _BAD_ESCAPE.search(url):
        raise EnvironmentIncompatible("locked wheel URL has a malformed percent escape")
    match = _URL.fullmatch(url)
    if match is None or match["scheme"].lower() != "https":
        raise EnvironmentIncompatible("locked wheel URL is not an HTTPS URL")
    authority = _authority(match["authority"])
    query = match["query"]
    if not _PATH.fullmatch(match["path"]) or (query is not None and not _QUERY.fullmatch(query)):
        raise EnvironmentIncompatible("locked wheel URL path or query is invalid")
    path = _remove_dot_segments(_normalize_escapes(match["path"]))
    name = path.rsplit("/", 1)[-1]
    if len(name) <= len(_WHEEL_SUFFIX) or not name.endswith(_WHEEL_SUFFIX):
        raise EnvironmentIncompatible("locked wheel URL does not name a wheel")
    identity = f"https://{authority}{path}"
    return identity if query is None else f"{identity}?{_normalize_escapes(query)}"


def locked_wheels(raw: bytes) -> dict[str, tuple[str, str]]:
    """Validate the committed lock and map every exact locked wheel URL to its distribution.

    Total over arbitrary bytes: any malformation is `EnvironmentIncompatible`. Each registry
    package must carry a PEP 503 canonical name and a bounded version, and no wheel URL may be
    locked for two packages under any spelling of its canonical ownership identity.
    """
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EnvironmentIncompatible("committed uv lock is invalid") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise EnvironmentIncompatible("committed uv lock has no package inventory")
    result: dict[str, tuple[str, str]] = {}
    owners: set[str] = set()
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
            owner = wheel_url_identity(url)
            if _HASH.fullmatch(digest) is None:
                raise EnvironmentIncompatible("locked wheel hash is not canonical")
            if owner in owners:
                raise EnvironmentIncompatible("a wheel URL is locked for more than one package")
            owners.add(owner)
            result[url] = (identity.name, identity.version)
    return result


def validate_lock(raw: bytes) -> None:
    locked_wheels(raw)


def locked_wheel_owners(raw: bytes) -> dict[str, tuple[str, str]]:
    """Map each locked wheel's canonical ownership identity to its locked (name, version).

    Outage evidence names a wheel by the URL uv reports, which may be uv's normalized spelling
    of the exact locked URL, so it is matched by identity; `locked_wheels` already guarantees
    that no two locked URLs share one.
    """
    return {wheel_url_identity(url): owner for url, owner in locked_wheels(raw).items()}
