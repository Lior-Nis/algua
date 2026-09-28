"""Positive evidence that a failed locked `uv sync` was a temporary wheel-acquisition outage.

uv 0.9.26 offers no structured failure signal for `sync`, so retryability is decided by a narrow
exact-pattern recognizer over its rendered error report, tied to a wheel URL the committed lock
already selected. It returns True only when both bounded streams decode as UTF-8 and consist of
optional `Resolved ...` and `help: ...` lines plus exactly one report, within one stream,
shaped like::

    × Failed to download `NAME==VERSION`
    ├─▶ Request failed after N retries
    ├─▶ Failed to fetch: `LOCKED-HTTPS-WHEEL-URL`
    ├─▶ error sending request for url (SAME-URL)
    ├─▶ client error (Connect)
    ├─▶ tcp connect error
    ╰─▶ Connection refused (os error 111)

where the reported URL is a valid HTTPS wheel URL whose canonical ownership identity is one of
NAME==VERSION's locked wheels (uv receives the exact locked spelling but may report its own
normalized form, so identity rather than spelling decides; a malformed URL is never evidence),
every cause is one of the exact forms below repeating that reported spelling, and the final
cause is a temporary transport failure (connection refused/reset, network unreachable) or an
HTTP 429/502/503/504 status for that URL. `│` lines continue the previous
element (uv wraps long URLs). Anything else -- a timeout, DNS, TLS/certificate, checksum,
extraction or filesystem failure, 404/500, an unknown line or cause, or a token that merely
resembles a network error -- is not evidence, and the caller classifies it as non-retryable.
"""
from __future__ import annotations

import re

from algua.registry.planner_environment_errors import EnvironmentIncompatible
from algua.registry.planner_environment_lock import wheel_url_identity

_HEAD = "× "
_CAUSES = ("├─▶ ", "╰─▶ ")
_CONTINUATION = "│"
_RESOLVED = re.compile(r"Resolved [0-9]+ packages? in [0-9.]+(?:ms|s)")
_DOWNLOAD = re.compile(r"Failed to download `([a-z0-9][a-z0-9._-]*)==([^`\s]+)`")
_FETCH = re.compile(r"Failed to fetch: `([^`\s]+)`")
_RETRIES = re.compile(r"Request failed after [0-9]{1,2} retries")
_TRANSPORT = frozenset({"client error (Connect)", "tcp connect error",
                        "tunnel error: failed to create underlying connection"})
_TEMPORARY_OS_ERRORS = frozenset({
    "Connection refused (os error 111)",
    "Connection reset by peer (os error 104)",
    "Network is unreachable (os error 101)",
})
_TEMPORARY_STATUSES = (
    "HTTP status client error (429 Too Many Requests)",
    "HTTP status server error (502 Bad Gateway)",
    "HTTP status server error (503 Service Unavailable)",
    "HTTP status server error (504 Gateway Timeout)",
)


def _reports(text: str) -> list[list[str]] | None:
    """Split one stream into reports, or None if any line is outside the known shape."""
    reports: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _RESOLVED.fullmatch(line) or line.startswith("help: "):
            continue
        if line.startswith(_HEAD):
            reports.append([line[len(_HEAD):].strip()])
        elif line.startswith(_CAUSES) and reports:
            reports[-1].append(line[len(_CAUSES[0]):].strip())
        elif line.startswith(_CONTINUATION) and reports:
            reports[-1][-1] = f"{reports[-1][-1]} {line[len(_CONTINUATION):].strip()}"
        else:
            return None
    return reports


def is_locked_wheel_outage(
    stdout: bytes, stderr: bytes, owners: dict[str, tuple[str, str]],
) -> bool:
    """Each stream is parsed on its own; together they must hold exactly one report.

    ``owners`` maps each locked wheel's canonical ownership identity (`locked_wheel_owners`) to
    its locked (name, version).
    """
    reports: list[list[str]] = []
    for stream in (stdout, stderr):
        try:
            parsed = _reports(stream.decode("utf-8"))
        except UnicodeDecodeError:
            return False
        if parsed is None:
            return False
        reports.extend(parsed)
    if len(reports) != 1:
        return False
    elements = reports[0]
    download = _DOWNLOAD.fullmatch(elements[0])
    causes = elements[1:]
    fetches = [match for cause in causes if (match := _FETCH.fullmatch(cause))]
    if download is None or len(fetches) != 1:
        return False
    url = fetches[0].group(1)
    try:
        identity = wheel_url_identity(url)
    except EnvironmentIncompatible:
        return False
    if owners.get(identity) != (download.group(1), download.group(2)):
        return False
    statuses = {f"{status} for url ({url})" for status in _TEMPORARY_STATUSES}
    allowed = {f"Failed to fetch: `{url}`", f"error sending request for url ({url})",
               *_TRANSPORT, *_TEMPORARY_OS_ERRORS, *statuses}
    if any(cause not in allowed and _RETRIES.fullmatch(cause) is None for cause in causes):
        return False
    return causes[-1] in _TEMPORARY_OS_ERRORS or causes[-1] in statuses
