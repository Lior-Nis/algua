"""Retry classification for a failed locked `uv sync`.

`RECORDED_CONNECTION_REFUSED` is verbatim uv 0.9.26 stderr, captured once while developing
Story 1.3b chunk 2 by running the normative sync against a one-wheel lock with the connection
forced to fail. Every other fixture is a documented, conservative variant of that rendering
(same miette layout, same reqwest cause wording). Anything the recognizer does not match exactly
is classified as non-retryable incompatibility, so an inaccurate positive fixture can only make
classification stricter, never broader.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import (
    EnvironmentIncompatible,
    validate_lock,
)
from algua.registry.planner_environment_errors import EnvironmentUnavailable
from algua.registry.planner_environment_lock import locked_wheel_owners, locked_wheels
from algua.registry.planner_environment_outage import is_locked_wheel_outage
from tests.test_planner_environment import _inputs
from tests.test_planner_environment_uv import _provision

WHEEL = (
    "https://files.pythonhosted.org/packages/b7/ce/"
    "149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/"
    "six-1.17.0-py2.py3-none-any.whl"
)
OTHER_WHEEL = "https://files.pythonhosted.org/packages/aa/bb/other-1.0-py3-none-any.whl"
HASH = "sha256:4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274"
LOCK = (
    "version = 1\n"
    "[[package]]\nname = \"six\"\nversion = \"1.17.0\"\n"
    "source = { registry = \"https://pypi.org/simple\" }\n"
    f"wheels = [{{ url = \"{WHEEL}\", hash = \"{HASH}\" }}]\n"
    "[[package]]\nname = \"other\"\nversion = \"1.0\"\n"
    "source = { registry = \"https://pypi.org/simple\" }\n"
    f"wheels = [{{ url = \"{OTHER_WHEEL}\", hash = \"sha256:{'b' * 64}\" }}]\n"
).encode()
BRANCH = "├─▶"
LAST = "╰─▶"
RULE = "│"
RECORDED_CONNECTION_REFUSED = (
    "Resolved 2 packages in 0.56ms\n"
    "  × Failed to download `six==1.17.0`\n"
    f"  {BRANCH} Request failed after 3 retries\n"
    f"  {BRANCH} Failed to fetch:\n"
    f"  {RULE}   `{WHEEL}`\n"
    f"  {BRANCH} error sending request for url\n"
    f"  {RULE}   ({WHEEL})\n"
    f"  {BRANCH} client error (Connect)\n"
    f"  {BRANCH} tunnel error: failed to create underlying connection\n"
    f"  {BRANCH} tcp connect error\n"
    f"  {LAST} Connection refused (os error 111)\n"
    "  help: `six` (v1.17.0) was included because `probe` (v0) depends on `six`\n"
)


def _report(*causes: str, package: str = "six==1.17.0", url: str = WHEEL) -> str:
    lines = ["Resolved 2 packages in 0.56ms", f"  × Failed to download `{package}`",
             f"  {BRANCH} Request failed after 3 retries", f"  {BRANCH} Failed to fetch:",
             f"  {RULE}   `{url}`"]
    for index, cause in enumerate(causes):
        arrow = LAST if index == len(causes) - 1 else BRANCH
        lines.append(f"  {arrow} {cause}")
    return "\n".join(lines) + "\n"


def _direct(terminal: str) -> str:
    return _report(f"error sending request for url ({WHEEL})", "client error (Connect)",
                   "tcp connect error", terminal)


def _status(status: str) -> str:
    return _report(f"{status} for url ({WHEEL})")


RETRYABLE = {
    "recorded-connection-refused": RECORDED_CONNECTION_REFUSED,
    "direct-connection-refused": _direct("Connection refused (os error 111)"),
    "connection-reset": _direct("Connection reset by peer (os error 104)"),
    "network-unreachable": _direct("Network is unreachable (os error 101)"),
    "bad-gateway": _status("HTTP status server error (502 Bad Gateway)"),
    "service-unavailable": _status("HTTP status server error (503 Service Unavailable)"),
    "gateway-timeout": _status("HTTP status server error (504 Gateway Timeout)"),
    "rate-limited": _status("HTTP status client error (429 Too Many Requests)"),
}

NOT_RETRYABLE = {
    "empty": "",
    "arbitrary-tokens": "error: download failed: network timeout while reading connection\n",
    "unlocked-url": RECORDED_CONNECTION_REFUSED.replace(WHEEL, WHEEL + "x"),
    "other-package-url": _report(
        f"error sending request for url ({OTHER_WHEEL})", "Connection refused (os error 111)",
        url=OTHER_WHEEL),
    "wrong-version": RECORDED_CONNECTION_REFUSED.replace("six==1.17.0", "six==1.16.0"),
    "http-url": RECORDED_CONNECTION_REFUSED.replace("https://", "http://"),
    "mismatched-request-url": RECORDED_CONNECTION_REFUSED.replace(
        f"({WHEEL})", f"({OTHER_WHEEL})"),
    "timeout": _direct("operation timed out"),
    "dns": _direct("dns error: failed to lookup address information: "
                   "Temporary failure in name resolution"),
    "tls-certificate": _report(f"error sending request for url ({WHEEL})",
                               "client error (Connect)",
                               "invalid peer certificate: UnknownIssuer"),
    "not-found": _status("HTTP status client error (404 Not Found)"),
    "server-error-500": _status("HTTP status server error (500 Internal Server Error)"),
    "hash-mismatch": (
        "  × Failed to download `six==1.17.0`\n"
        f"  {LAST} Hash mismatch for `six==1.17.0`\n"
    ),
    "extraction": _report("Failed to extract archive: six-1.17.0-py2.py3-none-any.whl",
                          "No space left on device (os error 28)"),
    "permission": _direct("Permission denied (os error 13)"),
    "extra-cause": _report(f"error sending request for url ({WHEEL})", "proxy misconfigured",
                           "Connection refused (os error 111)"),
    "unknown-line": RECORDED_CONNECTION_REFUSED + "warning: something else happened\n",
    "two-reports": RECORDED_CONNECTION_REFUSED + RECORDED_CONNECTION_REFUSED,
    "no-fetch": (
        "  × Failed to download `six==1.17.0`\n"
        f"  {LAST} Connection refused (os error 111)\n"
    ),
    "two-fetches": _report(f"Failed to fetch: `{WHEEL}`", "Connection refused (os error 111)"),
    "status-for-other-url": _status(
        "HTTP status server error (503 Service Unavailable)").replace(
            f"for url ({WHEEL})", f"for url ({OTHER_WHEEL})"),
    "transport-not-terminal": _report(
        f"error sending request for url ({WHEEL})", "Connection refused (os error 111)",
        "client error (Connect)"),
}


@pytest.mark.parametrize("stderr", RETRYABLE.values(), ids=RETRYABLE.keys())
def test_positive_locked_wheel_outage_evidence_is_retryable(stderr: str) -> None:
    assert is_locked_wheel_outage(b"", stderr.encode(), locked_wheel_owners(LOCK)) is True


def test_outage_evidence_on_stdout_is_inspected() -> None:
    assert is_locked_wheel_outage(
        RECORDED_CONNECTION_REFUSED.encode(), b"", locked_wheel_owners(LOCK)) is True


@pytest.mark.parametrize("stderr", NOT_RETRYABLE.values(), ids=NOT_RETRYABLE.keys())
def test_anything_else_is_not_retryable(stderr: str) -> None:
    assert is_locked_wheel_outage(b"", stderr.encode(), locked_wheel_owners(LOCK)) is False


def test_evidence_split_across_streams_or_undecodable_is_not_retryable() -> None:
    wheels = locked_wheel_owners(LOCK)
    head, tail = RECORDED_CONNECTION_REFUSED.split(f"  {BRANCH} client error", 1)
    assert is_locked_wheel_outage(
        head.encode(), (f"  {BRANCH} client error" + tail).encode(), wheels) is False
    assert is_locked_wheel_outage(
        b"\xff", RECORDED_CONNECTION_REFUSED.encode(), wheels) is False
    assert is_locked_wheel_outage(
        b"help: \xff\n", RECORDED_CONNECTION_REFUSED.encode(), wheels) is False


def test_only_https_wheel_urls_can_be_evidence() -> None:
    insecure = WHEEL.replace("https://", "http://")
    report = RECORDED_CONNECTION_REFUSED.replace(WHEEL, insecure)

    assert is_locked_wheel_outage(b"", report.encode(), {insecure: ("six", "1.17.0")}) is False


def test_locked_wheels_maps_every_locked_wheel_url_to_its_distribution() -> None:
    assert locked_wheels(LOCK) == {WHEEL: ("six", "1.17.0"), OTHER_WHEEL: ("other", "1.0")}


def _lock_inputs() -> tuple[FrozenFile, ...]:
    inputs = list(_inputs())
    inputs[2] = FrozenFile("uv.lock", "100644", LOCK)
    return tuple(inputs)


def test_sync_failure_with_positive_evidence_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EnvironmentUnavailable):
        _provision(tmp_path, monkeypatch, inputs=_lock_inputs(), sync=BoundedCompletion(
            2, b"", RECORDED_CONNECTION_REFUSED.encode()))


def _single_wheel_lock(url: str) -> bytes:
    return (
        "version = 1\n"
        "[[package]]\nname = \"six\"\nversion = \"1.17.0\"\n"
        "source = { registry = \"https://pypi.org/simple\" }\n"
        f"wheels = [{{ url = \"{url}\", hash = \"{HASH}\" }}]\n"
    ).encode()


ESCAPED = WHEEL.replace("six-1.17.0-py2", "six-1.17.0%2Blocal-py2")
# (raw URL in the committed lock, uv's normalized spelling of it in the failure report)
NORMALIZED_REPORTS = {
    "uppercase-host": (WHEEL.replace("files.", "FILES."), WHEEL),
    "uppercase-scheme": ("HTTPS" + WHEEL[len("https"):], WHEEL),
    "default-port": (WHEEL.replace(".org/", ".org:443/"), WHEEL),
    "escaped-unreserved": (WHEEL.replace("six-1.17.0-", "six%2D1.17.0-"), WHEEL),
    "lowercase-reserved-escape": (ESCAPED.replace("%2B", "%2b"), ESCAPED),
    "dot-segment": (WHEEL.replace("/packages/b7/", "/packages/./b7/"), WHEEL),
    "dot-dot-segment": (WHEEL.replace("/packages/b7/", "/packages/x/../b7/"), WHEEL),
}


@pytest.mark.parametrize("pair", NORMALIZED_REPORTS.values(), ids=NORMALIZED_REPORTS.keys())
def test_uvs_normalized_spelling_of_a_locked_alias_is_retryable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pair: tuple[str, str],
) -> None:
    raw, reported = pair
    inputs = list(_inputs())
    inputs[2] = FrozenFile("uv.lock", "100644", _single_wheel_lock(raw))
    report = RECORDED_CONNECTION_REFUSED.replace(WHEEL, reported)

    with pytest.raises(EnvironmentUnavailable):
        _provision(tmp_path, monkeypatch, inputs=tuple(inputs),
                   sync=BoundedCompletion(2, b"", report.encode()))


@pytest.mark.parametrize("pair", NORMALIZED_REPORTS.values(), ids=NORMALIZED_REPORTS.keys())
def test_outage_evidence_matches_the_reported_urls_canonical_wheel_identity(
    pair: tuple[str, str],
) -> None:
    from algua.registry import planner_environment_lock

    raw, reported = pair
    owners = planner_environment_lock.locked_wheel_owners(_single_wheel_lock(raw))
    report = RECORDED_CONNECTION_REFUSED.replace(WHEEL, reported)

    assert is_locked_wheel_outage(b"", report.encode(), owners) is True
    # the reverse spelling (a canonical lock, a non-canonical report) is the same wheel too
    canonical = planner_environment_lock.locked_wheel_owners(_single_wheel_lock(reported))
    reverse = RECORDED_CONNECTION_REFUSED.replace(WHEEL, raw)
    assert is_locked_wheel_outage(b"", reverse.encode(), canonical) is True


def test_locked_wheel_owners_are_keyed_by_canonical_identity() -> None:
    from algua.registry import planner_environment_lock

    aliased = WHEEL.replace("files.", "FILES.").replace(".org/", ".org:443/")
    assert planner_environment_lock.locked_wheel_owners(_single_wheel_lock(aliased)) == {
        WHEEL: ("six", "1.17.0")}


MALFORMED_REPORTED_URLS = {
    "malformed-escape": WHEEL.replace("/b7/", "/%zz/"),
    "credentials": WHEEL.replace("https://", "https://user@"),
    "invalid-port": WHEEL.replace(".org/", ".org:0/"),
    "oversized-port": WHEEL.replace(".org/", ".org:" + "9" * 5000 + "/"),
    "ambiguous-numeric-host": WHEEL.replace("files.pythonhosted.org", "01.2.3.4"),
    "not-a-wheel": WHEEL.replace(".whl", ".tar.gz"),
    "http": WHEEL.replace("https://", "http://"),
    "fragment": WHEEL + "#sha256=x",
    "unlocked-wheel": WHEEL.replace("six-1.17.0", "six-1.18.0"),
}


@pytest.mark.parametrize(
    "reported", MALFORMED_REPORTED_URLS.values(), ids=MALFORMED_REPORTED_URLS.keys())
def test_a_malformed_or_unlocked_reported_url_is_not_evidence(reported: str) -> None:
    from algua.registry import planner_environment_lock

    owners = planner_environment_lock.locked_wheel_owners(LOCK)
    report = RECORDED_CONNECTION_REFUSED.replace(WHEEL, reported)

    assert is_locked_wheel_outage(b"", report.encode(), owners) is False


def test_an_aliased_report_for_another_distribution_is_not_evidence() -> None:
    from algua.registry import planner_environment_lock

    owners = planner_environment_lock.locked_wheel_owners(LOCK)
    report = RECORDED_CONNECTION_REFUSED.replace(
        WHEEL, WHEEL.replace("files.", "FILES.")).replace("six==1.17.0", "other==1.0")

    assert is_locked_wheel_outage(b"", report.encode(), owners) is False


def test_every_cause_must_repeat_the_reported_url_spelling_exactly() -> None:
    from algua.registry import planner_environment_lock

    owners = planner_environment_lock.locked_wheel_owners(LOCK)
    alias = WHEEL.replace("files.", "FILES.")
    report = RECORDED_CONNECTION_REFUSED.replace(f"({WHEEL})", f"({alias})")

    assert is_locked_wheel_outage(b"", report.encode(), owners) is False


@pytest.mark.parametrize(
    "sync",
    [BoundedCompletion(2, b"", NOT_RETRYABLE["arbitrary-tokens"].encode()),
     BoundedCompletion(2, b"", NOT_RETRYABLE["hash-mismatch"].encode()),
     BoundedCompletion(2, b"", NOT_RETRYABLE["timeout"].encode()),
     subprocess.TimeoutExpired(["uv", "sync"], 900)],
    ids=["tokens", "hash", "uv-timeout", "subprocess-timeout"],
)
def test_sync_failure_without_positive_evidence_is_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync: object,
) -> None:
    with pytest.raises(EnvironmentIncompatible) as caught:
        _provision(tmp_path, monkeypatch, inputs=_lock_inputs(), sync=sync)

    assert not isinstance(caught.value, EnvironmentUnavailable)


@pytest.mark.parametrize(
    "url", ["https://[::1/x.whl", "https://[not-an-ip]/x.whl", "https://:443/x.whl"],
)
def test_malformed_locked_wheel_url_is_incompatible(url: str) -> None:
    lock = (
        "version=1\n[[package]]\nname='bad'\nversion='1'\n"
        "source={registry='https://pypi.org/simple'}\n"
        f"wheels=[{{url='{url}',hash='sha256:{'a' * 64}'}}]\n"
    ).encode()
    with pytest.raises(EnvironmentIncompatible):
        validate_lock(lock)
