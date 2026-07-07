"""Real-git smoke tests for :class:`RealGitOps` + the repo-global lock (#485, Task 3).

Each test builds a throwaway repo in a temp dir with a **bare "origin"** remote (no network), so the
remote-authoritative paths (fetch/remote_tip/push_cas/blob_at) exercise real git plumbing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from algua.operator.gitops import RealGitOps, RemoteMovedError, merge_back_lock


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """``main`` (with an initial commit, pushed to a bare origin) + a ``feature`` branch one commit
    ahead adding a strategy artifact. Checkout left on ``main`` with ``origin`` tracking."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "--bare")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("initial\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "initial")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin", "main")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "algua").mkdir()
    strat = repo / "algua" / "strat.py"
    strat.write_text("SIGNAL = 1\n")
    _git(repo, "add", "algua/strat.py")
    _git(repo, "commit", "-q", "-m", "feat: add strategy")
    _git(repo, "checkout", "-q", "main")
    return repo


def test_basic_probes(repo: Path) -> None:
    git = RealGitOps(repo)
    assert git.current_branch() == "main"
    assert git.working_tree_clean() is True
    assert git.remote_tip("main") == git.resolve("main")


def test_changed_entries_parses_add(repo: Path) -> None:
    git = RealGitOps(repo)
    base = git.merge_base("main", "feature")
    tip = git.resolve("feature")
    entries = git.changed_entries(base, tip)
    assert len(entries) == 1
    e = entries[0]
    assert e.change_type.startswith("A")
    assert e.new_path == "algua/strat.py"
    assert e.mode == "100644"


def test_changed_entries_parses_rename(repo: Path) -> None:
    git = RealGitOps(repo)
    _git(repo, "checkout", "-q", "-b", "rename-branch")
    _git(repo, "mv", "README.md", "renamed.md")
    _git(repo, "commit", "-q", "-am", "rename")
    tip = git.resolve("rename-branch")
    base = git.merge_base("main", "rename-branch")
    entries = git.changed_entries(base, tip)
    rename = [e for e in entries if e.change_type.startswith("R")]
    assert rename, entries
    assert rename[0].old_path == "README.md"
    assert rename[0].new_path == "renamed.md"


def test_changed_entries_parses_symlink_mode(repo: Path) -> None:
    git = RealGitOps(repo)
    _git(repo, "checkout", "-q", "-b", "link-branch")
    (repo / "algua").mkdir(exist_ok=True)
    import os
    os.symlink("../README.md", repo / "algua" / "link.py")
    _git(repo, "add", "algua/link.py")
    _git(repo, "commit", "-q", "-m", "add symlink")
    tip = git.resolve("link-branch")
    base = git.merge_base("main", "link-branch")
    entries = git.changed_entries(base, tip)
    link = [e for e in entries if e.new_path == "algua/link.py"]
    assert link and link[0].mode == "120000"


def test_merge_push_cas_and_content_check(repo: Path) -> None:
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    base = git.remote_tip("main")
    git.begin_merge(tip)
    git.commit_merge()
    merge_sha = git.merge_commit_of(tip)
    assert git.commit_second_parent(merge_sha) == tip

    git.push_cas(merge_sha, base)                       # real remote CAS
    assert git.remote_tip("main") == merge_sha
    assert git.is_ancestor(merge_sha, "refs/remotes/origin/main")

    # Effective-presence content check against origin/main.
    captured = git.tree_blobs(merge_sha, ["algua/strat.py"])
    assert git.blob_at("main", "algua/strat.py") == captured["algua/strat.py"]


def test_push_cas_stale_base_rejects(repo: Path) -> None:
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    git.commit_merge()
    merge_sha = git.merge_commit_of(tip)
    # A wrong expected_base (origin/main has not moved to it) fails the pre-push CAS.
    with pytest.raises(RemoteMovedError):
        git.push_cas(merge_sha, "0" * 40)


def test_revert_merge_returns_sha_and_undoes_code(repo: Path) -> None:
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    git.commit_merge()
    merge_sha = git.merge_commit_of(tip)
    assert (repo / "algua" / "strat.py").exists()
    revert_sha = git.revert_merge(merge_sha)
    assert revert_sha == git.resolve("HEAD")
    assert not (repo / "algua" / "strat.py").exists()


def test_blob_at_absent_path_is_none(repo: Path) -> None:
    git = RealGitOps(repo)
    assert git.blob_at("main", "does/not/exist.py") is None


def test_merge_back_lock_is_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "merge_back.git.lock"
    with merge_back_lock(lock_path):
        with pytest.raises(RuntimeError, match="another merge-back cycle is in progress"):
            with merge_back_lock(lock_path):
                pass  # pragma: no cover
    with merge_back_lock(lock_path):  # released on exit — re-acquirable
        pass
