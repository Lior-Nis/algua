"""Committed-lock identity policy: every locked registry package is canonically named and
versioned, and extracting its locked wheels is total -- any malformation stays inside the
non-retryable `EnvironmentIncompatible` boundary, never a raw `KeyError` or `TypeError`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion
from algua.registry import planner_environment
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
        planner_environment.locked_wheels(raw)


def test_locked_wheels_maps_each_wheel_to_its_canonical_identity() -> None:
    assert planner_environment.locked_wheels(_lock()) == {URL: ("six", "1.17.0")}


def _wheel(toml_url: str) -> str:
    """A wheels array whose URL is a TOML basic string, so escapes reach the parsed value."""
    return f'wheels = [{{ url = "{toml_url}", hash = "{HASH}" }}]'


NON_CANONICAL_URLS = {
    "scheme-uppercase": "HTTPS://files.pythonhosted.org/x.whl",
    "empty-query": "https://files.pythonhosted.org/x.whl?",
    "empty-fragment": "https://files.pythonhosted.org/x.whl#",
    "leading-space": " https://files.pythonhosted.org/x.whl",
    "tab": "https://files.pythonhosted.org/x\\t.whl",
    "newline": "https://files.pythonhosted.org/x\\n.whl",
    "carriage-return": "https://files.pythonhosted.org/x\\r.whl",
    "bell": "https://files.pythonhosted.org/x\\u0007.whl",
    "delete": "https://files.pythonhosted.org/x\\u007f.whl",
    "zero-width-space": "https://files.pythonhosted.org/x\\u200b.whl",
    "line-separator": "https://files.pythonhosted.org/x\\u2028.whl",
}


@pytest.mark.parametrize("toml_url", NON_CANONICAL_URLS.values(), ids=NON_CANONICAL_URLS.keys())
def test_locked_wheel_urls_must_be_printable_and_round_trip_exactly(toml_url: str) -> None:
    with pytest.raises(EnvironmentIncompatible):
        planner_environment.locked_wheels(_lock(wheels=_wheel(toml_url)))


@pytest.mark.parametrize("alias", ["https://files.pythonhosted.org/x.whl?",
                                   "https://files.pythonhosted.org/x\\t.whl"])
def test_url_aliases_cannot_evade_duplicate_detection(alias: str) -> None:
    canonical = "https://files.pythonhosted.org/x.whl"
    lock = _lock(wheels=_wheel(canonical), extra=(
        "[[package]]\nname = 'other'\nversion = '1.0'\n"
        "source = { registry = 'https://pypi.org/simple' }\n" + _wheel(alias) + "\n"))

    with pytest.raises(EnvironmentIncompatible):
        planner_environment.locked_wheels(lock)


def test_repository_lock_has_canonical_identities() -> None:
    wheels = planner_environment.locked_wheels((REPO / "uv.lock").read_bytes())

    assert wheels
    assert all(url.startswith("https://") for url in wheels)


def test_provision_refuses_a_malformed_lock_before_running_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key recorded for a malformed lock must not reach uv, nor escape as a raw KeyError."""
    inputs = (
        FrozenFile(".python-version", "100644",
                   f"{sys.version_info.major}.{sys.version_info.minor}\n".encode()),
        FrozenFile("pyproject.toml", "100644", b"[project]\nname='algua'\nversion='0'\n"),
        FrozenFile("uv.lock", "100644", _lock(version=None)),
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
