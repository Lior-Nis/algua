"""Hermetic merge-gate runner (#485 factory hardening): the gate runs in a throwaway worktree of
the STAGED tree, so the live checkout's held ``merge_back.lock``, real ``.env``, untracked files,
and operator environment can never leak into the suite. Commands are injected (the production
default is the real 4-command gate) so these tests prove the hermetic properties without running
an 8-minute pytest inside pytest."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from algua.operator.gate_runner import run_hermetic_quality_gate


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("initial\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def _worktree_count(repo: Path) -> int:
    return _git(repo, "worktree", "list", "--porcelain").count("worktree ")


def test_gates_staged_tree_not_ambient_files(repo: Path) -> None:
    # Staged-but-uncommitted content (the saga's merge preview) IS in the gated tree; an untracked
    # ambient file (a live .env, scratch junk) is NOT.
    (repo / "staged_marker.txt").write_text("m\n")
    _git(repo, "add", "staged_marker.txt")
    (repo / ".env").write_text("ALPACA_API_KEY=real-secret\n")  # untracked, never staged
    ok = run_hermetic_quality_gate(
        repo,
        commands=[["test", "-f", "staged_marker.txt"], ["test", "!", "-f", ".env"]],
    )
    assert ok is True
    # The live checkout's staged index is untouched by the gate run.
    assert "staged_marker.txt" in _git(repo, "diff", "--cached", "--name-only")


def test_operator_env_is_scrubbed(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", "/live/operator/algua.db")
    monkeypatch.setenv("ALPACA_API_KEY", "real-secret")
    monkeypatch.setenv("VIRTUAL_ENV", "/live/checkout/.venv")
    ok = run_hermetic_quality_gate(
        repo,
        commands=[[
            "sh", "-c",
            'test -z "$ALGUA_DB_PATH" && test -z "$ALPACA_API_KEY" && test -z "$VIRTUAL_ENV"',
        ]],
    )
    assert ok is True


def test_gate_worktree_has_own_git_dir(repo: Path) -> None:
    # The lock-collision fix: merge_back.lock anchors at `git rev-parse --absolute-git-dir`, which
    # is PER-WORKTREE — a lock taken inside the gate worktree can never collide with the live
    # cycle's held lock.
    main_git_dir = _git(repo, "rev-parse", "--absolute-git-dir")
    ok = run_hermetic_quality_gate(
        repo,
        commands=[["sh", "-c", f'[ "$(git rev-parse --absolute-git-dir)" != "{main_git_dir}" ]']],
    )
    assert ok is True


def test_red_command_fails_closed_and_cleans_up(repo: Path) -> None:
    baseline = _worktree_count(repo)
    assert run_hermetic_quality_gate(repo, commands=[["false"]]) is False
    assert _worktree_count(repo) == baseline


def test_short_circuits_on_first_failure(repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "second_ran"
    ok = run_hermetic_quality_gate(
        repo, commands=[["false"], ["sh", "-c", f"touch {marker}"]])
    assert ok is False
    assert not marker.exists()


def test_green_cleans_up_worktree(repo: Path) -> None:
    baseline = _worktree_count(repo)
    assert run_hermetic_quality_gate(repo, commands=[["true"]]) is True
    assert _worktree_count(repo) == baseline


def test_plumbing_failure_is_red_not_raise(tmp_path: Path) -> None:
    # Not a git repo at all: the gate must return False, never raise into the saga.
    not_repo = tmp_path / "not_a_repo"
    not_repo.mkdir()
    assert run_hermetic_quality_gate(not_repo, commands=[["true"]]) is False
