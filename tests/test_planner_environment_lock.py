"""Committed-lock identity policy: every locked registry package is canonically named and
versioned, and extracting its locked wheels is total -- any malformation stays inside the
non-retryable `EnvironmentIncompatible` boundary, never a raw `KeyError` or `TypeError`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion
from algua.registry import planner_environment, planner_environment_lock
from algua.registry.artifact_contract import BuildInputs
from algua.registry.environment_contract import EnvironmentKey
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import (
    CREATE_FLAGS,
    SYNC_FLAGS,
    EnvironmentIncompatible,
    current_interpreter_identity,
    provision_environment,
    validate_lock,
)
from tests._venv_fixture import uv_like_venv

URL = "https://files.pythonhosted.org/packages/aa/bb/six-1.17.0-py2.py3-none-any.whl"
HASH = "sha256:" + "a" * 64
REPO = Path(__file__).resolve().parents[1]


def _lock(*, name: str | None = "'six'", version: str | None = "'1.17.0'",
          wheels: str | None = None, extra: str = "") -> bytes:
    fields = ["[[package]]"]
    if name is not None:
        fields.append(f"name = {name}")
    if version is not None:
        fields.append(f"version = {version}")
    fields.append("source = { registry = 'https://pypi.org/simple' }")
    default = f"wheels = [{{ url = '{URL}', hash = '{HASH}' }}]"
    fields.append(wheels if wheels is not None else default)
    return ("version = 1\n" + "\n".join(fields) + "\n" + extra).encode()


MALFORMED = {
    "name-missing": _lock(name=None),
    "name-empty": _lock(name="''"),
    "name-integer": _lock(name="1"),
    "name-uppercase": _lock(name="'Six'"),
    "name-underscore": _lock(name="'six_x'"),
    "name-dot": _lock(name="'six.x'"),
    "name-whitespace": _lock(name="' six'"),
    "name-overlong": _lock(name=f"'{'s' * 129}'"),
    "version-missing": _lock(version=None),
    "version-empty": _lock(version="''"),
    "version-integer": _lock(version="1"),
    "version-whitespace": _lock(version="' 1.17.0'"),
    "version-inner-space": _lock(version="'1 .17'"),
    "version-overlong": _lock(version=f"'{'1' * 129}'"),
    "wheels-not-list": _lock(wheels=f"wheels = {{ url = '{URL}', hash = '{HASH}' }}"),
    "wheel-url-missing": _lock(wheels=f"wheels = [{{ hash = '{HASH}' }}]"),
    "wheel-url-integer": _lock(wheels=f"wheels = [{{ url = 1, hash = '{HASH}' }}]"),
    "duplicate-wheel-url": _lock(extra=(
        "[[package]]\nname = 'other'\nversion = '1.0'\n"
        "source = { registry = 'https://pypi.org/simple' }\n"
        f"wheels = [{{ url = '{URL}', hash = '{HASH}' }}]\n")),
    "not-utf8": b"\xff" + _lock(),
    "not-toml": b"[[package]\n",
    "package-table-not-list": b"version = 1\npackage = 1\n",
    "package-not-table": b"version = 1\npackage = [1]\n",
}


@pytest.mark.parametrize("raw", MALFORMED.values(), ids=MALFORMED.keys())
def test_validate_lock_refuses_malformed_locked_identities(raw: bytes) -> None:
    with pytest.raises(EnvironmentIncompatible):
        validate_lock(raw)


@pytest.mark.parametrize("raw", MALFORMED.values(), ids=MALFORMED.keys())
def test_locked_wheels_is_total_and_fail_closed(raw: bytes) -> None:
    with pytest.raises(EnvironmentIncompatible):
        planner_environment_lock.locked_wheels(raw)


def test_locked_wheels_maps_each_wheel_to_its_canonical_identity() -> None:
    assert planner_environment_lock.locked_wheels(_lock()) == {URL: ("six", "1.17.0")}


def _wheel(toml_url: str) -> str:
    """A wheels array whose URL is a TOML basic string, so escapes reach the parsed value."""
    return f'wheels = [{{ url = "{toml_url}", hash = "{HASH}" }}]'


REJECTED_WHEEL_URLS = {
    # characters a URL can never carry raw
    "leading-space": " https://files.pythonhosted.org/x.whl",
    "inner-space": "https://files.pythonhosted.org/x y.whl",
    "trailing-space": "https://files.pythonhosted.org/x.whl ",
    "tab": "https://files.pythonhosted.org/x\\t.whl",
    "newline": "https://files.pythonhosted.org/x\\n.whl",
    "carriage-return": "https://files.pythonhosted.org/x\\r.whl",
    "bell": "https://files.pythonhosted.org/x\\u0007.whl",
    "delete": "https://files.pythonhosted.org/x\\u007f.whl",
    "zero-width-space": "https://files.pythonhosted.org/x\\u200b.whl",
    "line-separator": "https://files.pythonhosted.org/x\\u2028.whl",
    "non-ascii": "https://files.pythonhosted.org/café.whl",
    # KELVIN SIGN lowercases to ASCII "k", so it must be refused before the host is folded
    "non-ascii-host-folding": "https://\\u212aeras.org/x.whl",
    "backslash": "https://files.pythonhosted.org/a\\\\x.whl",
    "angle-bracket": "https://files.pythonhosted.org/<x>.whl",
    "bracket-in-path": "https://files.pythonhosted.org/[x].whl",
    "bracket-in-query": "https://files.pythonhosted.org/x.whl?v=[1]",
    # credentials and fragments
    "credentials": "https://user@files.pythonhosted.org/x.whl",
    "password": "https://user:secret@files.pythonhosted.org/x.whl",
    "empty-userinfo": "https://@files.pythonhosted.org/x.whl",
    "fragment": "https://files.pythonhosted.org/x.whl#sha256=abc",
    "empty-fragment": "https://files.pythonhosted.org/x.whl#",
    # malformed percent escapes
    "short-escape": "https://files.pythonhosted.org/x%2.whl",
    "non-hex-escape": "https://files.pythonhosted.org/x%zz.whl",
    "query-bad-escape": "https://files.pythonhosted.org/x.whl?v=%g1",
    # scheme and authority
    "http": "http://files.pythonhosted.org/x.whl",
    "no-authority": "https:files.pythonhosted.org/x.whl",
    "empty-host": "https:///x.whl",
    "underscore-host": "https://files_python.org/x.whl",
    "trailing-dot-host": "https://files.pythonhosted.org./x.whl",
    "empty-label": "https://files..pythonhosted.org/x.whl",
    "hyphen-edge-label": "https://-files.pythonhosted.org/x.whl",
    "overlong-hostname": "https://" + ".".join(["a" * 63] * 4) + "/x.whl",
    "escaped-host": "https://files%2Epythonhosted.org/x.whl",
    "ipv6-zone": "https://[fe80::1%25eth0]/x.whl",
    "ipv6-invalid": "https://[::g]/x.whl",
    "ipv-future": "https://[v1.x]/x.whl",
    "ipv6-unbracketed": "https://::1/x.whl",
    "numeric-host-leading-zero": "https://01.2.3.4/x.whl",
    "numeric-host-short": "https://1.2.3/x.whl",
    "numeric-host-hex": "https://0x7f.0.0.1/x.whl",
    "numeric-host-integer": "https://16909060/x.whl",
    "zero-port": "https://files.pythonhosted.org:0/x.whl",
    "port-out-of-range": "https://files.pythonhosted.org:65536/x.whl",
    "huge-port": "https://files.pythonhosted.org:99999999999/x.whl",
    "non-numeric-port": "https://files.pythonhosted.org:https/x.whl",
    # not a wheel
    "no-path": "https://files.pythonhosted.org",
    "not-a-wheel": "https://files.pythonhosted.org/x.tar.gz",
    "bare-extension": "https://files.pythonhosted.org/.whl",
    "wheel-only-in-query": "https://files.pythonhosted.org/x?file=x.whl",
    "escaped-slash-suffix": "https://files.pythonhosted.org/x.whl%2F",
}


@pytest.mark.parametrize("toml_url", REJECTED_WHEEL_URLS.values(),
                         ids=REJECTED_WHEEL_URLS.keys())
def test_invalid_locked_wheel_urls_are_refused(toml_url: str) -> None:
    with pytest.raises(EnvironmentIncompatible):
        planner_environment_lock.locked_wheels(_lock(wheels=_wheel(toml_url)))


HOST = "https://files.pythonhosted.org"
WHEEL_URL_IDENTITIES = {
    "repository-shape": (URL, URL),
    "scheme-and-host-case": ("HTTPS://Files.PythonHosted.ORG/x.whl", f"{HOST}/x.whl"),
    "default-port": ("https://files.pythonhosted.org:443/x.whl", f"{HOST}/x.whl"),
    "empty-port": ("https://files.pythonhosted.org:/x.whl", f"{HOST}/x.whl"),
    "zero-padded-default-port": ("https://files.pythonhosted.org:0443/x.whl", f"{HOST}/x.whl"),
    "zero-padded-port": ("https://files.pythonhosted.org:08443/x.whl",
                         "https://files.pythonhosted.org:8443/x.whl"),
    "ipv4": ("https://192.0.2.10/x.whl", "https://192.0.2.10/x.whl"),
    "ipv6": ("https://[2001:DB8:0:0:0:0:0:1]:8443/x.whl", "https://[2001:db8::1]:8443/x.whl"),
    "ipv6-mapped-ipv4": ("https://[::FFFF:192.0.2.10]/x.whl", "https://[::ffff:c000:20a]/x.whl"),
    "query": (f"{HOST}/x.whl?token=abc&v=1", f"{HOST}/x.whl?token=abc&v=1"),
    "empty-query": (f"{HOST}/x.whl?", f"{HOST}/x.whl?"),
    "query-escapes": (f"{HOST}/x.whl?q=%7e%2f", f"{HOST}/x.whl?q=~%2F"),
    "path-sub-delimiters": (f"{HOST}/a!$&'()*+,;=:@b/x.whl", f"{HOST}/a!$&'()*+,;=:@b/x.whl"),
    "reserved-escape-kept": (f"{HOST}/torch-2.4.0%2bcpu.whl", f"{HOST}/torch-2.4.0%2Bcpu.whl"),
    "unreserved-escape-decoded": (f"{HOST}/%78.whl", f"{HOST}/x.whl"),
    "escaped-extension": (f"{HOST}/x%2ewhl", f"{HOST}/x.whl"),
    "dot-segments": (f"{HOST}/a/./b/../x.whl", f"{HOST}/a/x.whl"),
    "escaped-dot-segments": (f"{HOST}/a/%2E%2E/x.whl", f"{HOST}/x.whl"),
    "leading-dot-dot": (f"{HOST}/../x.whl", f"{HOST}/x.whl"),
    "empty-segment-kept": (f"{HOST}//x.whl", f"{HOST}//x.whl"),
}


@pytest.mark.parametrize("pair", WHEEL_URL_IDENTITIES.values(), ids=WHEEL_URL_IDENTITIES.keys())
def test_valid_wheel_urls_are_accepted_under_their_exact_raw_spelling(
    pair: tuple[str, str],
) -> None:
    raw, _identity = pair
    assert planner_environment_lock.locked_wheels(_lock(wheels=_wheel(raw))) == {
        raw: ("six", "1.17.0")}


@pytest.mark.parametrize("pair", WHEEL_URL_IDENTITIES.values(), ids=WHEEL_URL_IDENTITIES.keys())
def test_wheel_url_ownership_identity_is_rfc_canonical(pair: tuple[str, str]) -> None:
    raw, identity = pair
    assert planner_environment_lock.wheel_url_identity(raw) == identity


SEMANTIC_ALIASES = {
    "host-case": (f"{HOST}/x.whl", "https://FILES.pythonhosted.org/x.whl"),
    "scheme-case": (f"{HOST}/x.whl", "HTTPS://files.pythonhosted.org/x.whl"),
    "default-port": (f"{HOST}/x.whl", "https://files.pythonhosted.org:443/x.whl"),
    "empty-port": (f"{HOST}/x.whl", "https://files.pythonhosted.org:/x.whl"),
    "zero-padded-default-port": (f"{HOST}/x.whl", "https://files.pythonhosted.org:0443/x.whl"),
    "path-escape-case": (f"{HOST}/x%2B1.whl", f"{HOST}/x%2b1.whl"),
    "query-escape-case": (f"{HOST}/x.whl?a=%2F", f"{HOST}/x.whl?a=%2f"),
    "escaped-unreserved": (f"{HOST}/A.whl", f"{HOST}/%41.whl"),
    "escaped-extension": (f"{HOST}/x.whl", f"{HOST}/x%2Ewhl"),
    "dot-segment": (f"{HOST}/x.whl", f"{HOST}/./x.whl"),
    "dot-dot-segment": (f"{HOST}/x.whl", f"{HOST}/a/../x.whl"),
    "ipv6-spelling": ("https://[2001:db8::1]/x.whl", "https://[2001:DB8:0:0:0:0:0:1]/x.whl"),
}


@pytest.mark.parametrize("pair", SEMANTIC_ALIASES.values(), ids=SEMANTIC_ALIASES.keys())
def test_semantic_aliases_cannot_claim_a_second_wheel_owner(pair: tuple[str, str]) -> None:
    canonical, alias = pair
    lock = _lock(wheels=_wheel(canonical), extra=(
        "[[package]]\nname = 'other'\nversion = '1.0'\n"
        "source = { registry = 'https://pypi.org/simple' }\n" + _wheel(alias) + "\n"))

    with pytest.raises(EnvironmentIncompatible, match="more than one package"):
        planner_environment_lock.locked_wheels(lock)


@pytest.mark.parametrize(
    "pair",
    [(f"{HOST}/x%2Bcpu.whl", f"{HOST}/x+cpu.whl"), (f"{HOST}/x.whl", f"{HOST}/x.whl?")],
    ids=["reserved-escape-versus-raw", "empty-query-versus-none"],
)
def test_rfc_distinct_urls_remain_distinct_owners(pair: tuple[str, str]) -> None:
    first, second = pair
    lock = _lock(wheels=_wheel(first), extra=(
        "[[package]]\nname = 'other'\nversion = '1.0'\n"
        "source = { registry = 'https://pypi.org/simple' }\n" + _wheel(second) + "\n"))

    assert planner_environment_lock.locked_wheels(lock) == {
        first: ("six", "1.17.0"), second: ("other", "1.0")}


def test_repository_lock_has_canonical_identities() -> None:
    import tomllib

    raw = (REPO / "uv.lock").read_bytes()
    wheels = planner_environment_lock.locked_wheels(raw)
    locked = [
        wheel["url"] for package in tomllib.loads(raw.decode())["package"]
        for wheel in package.get("wheels", [])
    ]

    assert len(locked) == 1785
    assert sorted(wheels) == sorted(locked)  # keyed by each exact raw URL
    assert all(planner_environment_lock.wheel_url_identity(url) == url for url in locked)


OVERSIZED_PORT_LOCK = _lock(wheels=_wheel(f"https://files.pythonhosted.org:{'9' * 5000}/x.whl"))


@pytest.mark.parametrize(
    "raw", ["9" * 5000, "1" + "0" * 4400, "0" * 5000, "65536", "0"],
    ids=["5000-nines", "4401-digit", "5000-zeros", "just-over", "zero"])
def test_port_digit_strings_of_any_length_stay_environment_incompatible(raw: str) -> None:
    with pytest.raises(EnvironmentIncompatible):
        planner_environment_lock._port(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0" * 5000 + "8443", 8443), ("0" * 5000 + "443", None), ("00065535", 65535), ("1", 1)],
    ids=["padded-8443", "padded-default", "padded-max", "one"])
def test_zero_padded_ports_of_any_length_resolve_to_their_number(
    raw: str, expected: int | None,
) -> None:
    assert planner_environment_lock._port(raw) == expected


def test_an_oversized_locked_port_is_environment_incompatible() -> None:
    with pytest.raises(EnvironmentIncompatible):
        planner_environment_lock.locked_wheels(OVERSIZED_PORT_LOCK)


@pytest.mark.parametrize(
    "lock", [_lock(version=None), OVERSIZED_PORT_LOCK], ids=["missing-version", "oversized-port"])
def test_provision_refuses_a_malformed_lock_before_running_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock: bytes,
) -> None:
    """A key recorded for a malformed lock must not reach uv, nor escape as a raw KeyError."""
    inputs = (
        FrozenFile(".python-version", "100644",
                   f"{sys.version_info.major}.{sys.version_info.minor}\n".encode()),
        FrozenFile("pyproject.toml", "100644", b"[project]\nname='algua'\nversion='0'\n"),
        FrozenFile("uv.lock", "100644", lock),
    )
    key = EnvironmentKey(
        build_inputs_digest=BuildInputs(tuple(item.contract_entry for item in inputs)).digest,
        dependency_hash="a" * 64, interpreter=current_interpreter_identity(),
        uv_version="uv 0.9.26", create_argv=CREATE_FLAGS, sync_argv=SYNC_FLAGS,
    )
    monkeypatch.setattr(planner_environment, "installer_version", lambda: "uv 0.9.26")
    calls: list[list[str]] = []
    environment = tmp_path / "environment"

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        if argv[1] == "venv":
            uv_like_venv(environment)
            return BoundedCompletion(0, b"", b"")
        return BoundedCompletion(2, b"", b"error: sync failed\n")

    with pytest.raises(EnvironmentIncompatible):
        provision_environment(tmp_path / "inputs", environment, inputs, key, runner=runner)

    assert calls == []
