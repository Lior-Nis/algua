from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion, OutputLimitExceeded
from algua.registry import planner_environment
from algua.registry.planner_environment import (
    EnvironmentIncompatible,
    build_environment_key,
    installer_version,
    provision_environment,
)
from algua.registry.planner_environment_errors import EnvironmentUnavailable
from tests._venv_fixture import uv_like_venv
from tests.test_planner_environment import _inputs

UV_VERSION = "uv 0.9.26"
SECRET = "https://user:token@files.example/private/secret.whl"


def _launch_failures() -> list[BaseException]:
    return [
        FileNotFoundError(2, "No such file or directory", "/private/bin/uv"),
        PermissionError(13, "Permission denied", "/private/bin/uv"),
        OSError(7, "Argument list too long"),
        OutputLimitExceeded("stdout"),
        OutputLimitExceeded("stderr"),
    ]


def test_installer_version_runs_through_the_bounded_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return BoundedCompletion(0, f"{UV_VERSION}\n".encode(), b"")

    monkeypatch.setattr(planner_environment, "run_bounded", fake, raising=False)

    assert installer_version() == UV_VERSION
    [(argv, kwargs)] = calls
    assert argv[1:] == ["--version"]
    assert 0 < kwargs["max_stdout"] <= 4096 and 0 < kwargs["max_stderr"] <= 4096
    assert 0 < kwargs["timeout"] <= 60


@pytest.mark.parametrize(
    "outcome",
    [
        *_launch_failures(),
        subprocess.TimeoutExpired(["uv"], 30),
        BoundedCompletion(1, f"{UV_VERSION}\n".encode(), b""),
        BoundedCompletion(0, b"\xff\n", b""),
        BoundedCompletion(0, b"", b""),
        BoundedCompletion(0, f"{UV_VERSION}\nextra\n".encode(), b""),
    ],
)
def test_installer_version_failures_are_incompatible(
    monkeypatch: pytest.MonkeyPatch, outcome: object,
) -> None:
    def fake(*_args, **_kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(planner_environment, "run_bounded", fake, raising=False)

    with pytest.raises(EnvironmentIncompatible):
        installer_version()


def _provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, create: object = None,
    sync: object = None, inputs=None, calls: list[tuple[list[str], dict]] | None = None,
) -> tuple[list[tuple[list[str], dict]], object]:
    monkeypatch.setattr(planner_environment, "installer_version", lambda: UV_VERSION)
    inputs = inputs or _inputs()
    key = build_environment_key(inputs, "a" * 64, uv_version=UV_VERSION)
    environment = tmp_path / "environment"
    calls = [] if calls is None else calls

    def runner(argv, **kwargs):
        calls.append((list(argv), kwargs))
        outcome = create if argv[1] == "venv" else sync
        if isinstance(outcome, BaseException):
            raise outcome
        if argv[1] == "venv" and outcome is None:
            uv_like_venv(environment)
        return outcome or BoundedCompletion(0, b"", b"")

    result = provision_environment(
        tmp_path / "inputs", environment, inputs, key, runner=runner)
    return calls, result


def test_both_uv_commands_run_with_bounded_output_and_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, inventory = _provision(tmp_path, monkeypatch)

    assert [argv[1] for argv, _kwargs in calls] == ["venv", "sync"]
    for _argv, kwargs in calls:
        assert 0 < kwargs["max_stdout"] <= 1024 * 1024
        assert 0 < kwargs["max_stderr"] <= 1024 * 1024
        assert 0 < kwargs["timeout"] <= 3600
    assert inventory is not None


@pytest.mark.parametrize(
    "outcome",
    [*_launch_failures(), subprocess.TimeoutExpired(["uv"], 900),
     BoundedCompletion(1, b"", SECRET.encode())],
)
def test_create_failures_are_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: object,
) -> None:
    calls: list[tuple[list[str], dict]] = []
    with pytest.raises(EnvironmentIncompatible) as caught:
        _provision(tmp_path, monkeypatch, create=outcome, calls=calls)

    assert SECRET not in str(caught.value)
    assert [argv[1] for argv, _kwargs in calls] == ["venv"]


@pytest.mark.parametrize("outcome", _launch_failures())
def test_sync_launch_and_overflow_failures_are_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: BaseException,
) -> None:
    with pytest.raises(EnvironmentIncompatible) as caught:
        _provision(tmp_path, monkeypatch, sync=outcome)

    assert not isinstance(caught.value, EnvironmentUnavailable)


def test_sync_failure_diagnostics_never_carry_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises((EnvironmentIncompatible, EnvironmentUnavailable)) as caught:
        _provision(
            tmp_path, monkeypatch,
            sync=BoundedCompletion(1, SECRET.encode(), f"error: {SECRET}".encode()))

    chain: list[BaseException] = []
    current: BaseException | None = caught.value
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    assert all(SECRET not in str(item) for item in chain)
