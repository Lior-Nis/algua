"""Tests for `algua operator lock-run` (factory slice 3, #536 follow-on).

This is a NEW, purely additive command in `algua/cli/operator_cmd.py` — merge-back needs the SAME
`operator.lock` `operator run --job paper` takes, but fires many times a day (once per drain
cycle), so it cannot go through the session-gated `operator run` wrapper (see the command's own
docstring and `algua/operator/jobs.py`'s comment for why). These tests exercise `lock-run` in
isolation; `tests/test_cli_operator.py` is the separate, UNMODIFIED regression suite proving
`operator run`'s session-gate behavior is untouched by this addition.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.cli.operator_cmd as operator_cmd
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(operator_cmd, "_resolve_git_dir", lambda: tmp_path)
    return tmp_path


def _invoke(cmd: list[str]):
    return runner.invoke(app, ["operator", "lock-run", "--", *cmd])


# --- transparent passthrough on a run -------------------------------------------------------------


def test_ran_command_passthrough_stdout_and_exit_code(lock_dir, monkeypatch):
    def _spy(command: list[str]) -> subprocess.CompletedProcess:
        assert command == ["algua", "paper", "merge-back", "--branch", "b"]
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(operator_cmd, "_run_locked_command", _spy)
    result = _invoke(["algua", "paper", "merge-back", "--branch", "b"])
    assert result.exit_code == 0, result.output
    # Transparent passthrough: lock-run prints NOTHING of its own on the ran path (the wrapped
    # command's stdout — inherited directly, not captured — would appear here in a real
    # subprocess.run(); the stub above emits nothing, so CliRunner's captured output is empty).
    assert result.output == ""


def test_ran_command_propagates_nonzero_exit_code(lock_dir, monkeypatch):
    def _spy(command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, returncode=7, stdout="", stderr="")

    monkeypatch.setattr(operator_cmd, "_run_locked_command", _spy)
    result = _invoke(["algua", "paper", "merge-back"])
    assert result.exit_code == 7, result.output


def test_lock_is_actually_acquired_and_released_around_the_command(lock_dir, monkeypatch):
    calls: list[str] = []

    def _spy(command: list[str]) -> subprocess.CompletedProcess:
        # While the wrapped command is "running", the lock file must be HELD (a second acquire
        # attempt from a separate fd must fail).
        lock_path = lock_dir / "operator.lock"
        handle = open(lock_path, "a+")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            handle.close()
        calls.append("ran")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(operator_cmd, "_run_locked_command", _spy)
    result = _invoke(["algua", "paper", "merge-back"])
    assert result.exit_code == 0, result.output
    assert calls == ["ran"]

    # AFTER the command returns, the lock must be released — a fresh acquire now succeeds.
    lock_path = lock_dir / "operator.lock"
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must NOT raise
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


# --- lock contention: benign no-op, wrapped command NEVER runs -----------------------------------


def _hold_lock(lock_dir: Path):
    lock = lock_dir / "operator.lock"
    handle = open(lock, "a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def test_contention_is_benign_noop_and_never_runs_the_command(lock_dir, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        operator_cmd, "_run_locked_command",
        lambda c: calls.append(c) or subprocess.CompletedProcess(c, 0),
    )
    handle = _hold_lock(lock_dir)
    try:
        result = _invoke(["algua", "paper", "merge-back"])
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"ok": True, "ran": False, "reason": "locked"}
    assert calls == []  # the wrapped command must NEVER have run


# --- setup failures: git dir / lock file unavailable ------------------------------------------


def test_git_dir_unresolvable_fails_closed(tmp_path, monkeypatch):
    def _boom() -> Path:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "--absolute-git-dir"])

    monkeypatch.setattr(operator_cmd, "_resolve_git_dir", _boom)
    calls: list = []
    monkeypatch.setattr(
        operator_cmd, "_run_locked_command",
        lambda c: calls.append(c) or subprocess.CompletedProcess(c, 0),
    )

    result = _invoke(["algua", "paper", "merge-back"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["ran"] is False
    assert payload["reason"] == "git_dir_unresolved"
    assert calls == []


def test_lock_file_unopenable_fails_closed(tmp_path, monkeypatch):
    # `operator_run_lock` (algua.primitives.flock, #8) auto-creates any missing lock-dir via
    # `file_lock`'s `path.parent.mkdir`, so a bare "missing dir" no longer reproduces an unopenable
    # lock file. Force the primitive's `os.open` itself to fail instead (permission denied /
    # read-only fs / disk full, in spirit).
    monkeypatch.setattr(operator_cmd, "_resolve_git_dir", lambda: tmp_path)

    def _open_boom(*_a, **_k):
        raise PermissionError("EACCES")

    monkeypatch.setattr("algua.primitives.flock.os.open", _open_boom)
    calls: list = []
    monkeypatch.setattr(
        operator_cmd, "_run_locked_command",
        lambda c: calls.append(c) or subprocess.CompletedProcess(c, 0),
    )

    result = _invoke(["algua", "paper", "merge-back"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "lock_unavailable"
    assert calls == []


# --- empty command -> error envelope --------------------------------------------------------------


def test_empty_command_errors(lock_dir):
    result = runner.invoke(app, ["operator", "lock-run", "--"])
    assert result.exit_code == 1
    assert json.loads(result.output)["ok"] is False


# --- no session gate, no completion marker: proven by absence, not by touching schedule.py -------


def test_lock_run_writes_no_session_marker(lock_dir, monkeypatch):
    monkeypatch.setattr(
        operator_cmd, "_run_locked_command",
        lambda c: subprocess.CompletedProcess(c, 0, stdout="", stderr=""),
    )
    result = _invoke(["algua", "paper", "merge-back"])
    assert result.exit_code == 0, result.output
    assert not (lock_dir / "operator_sessions.json").exists()
