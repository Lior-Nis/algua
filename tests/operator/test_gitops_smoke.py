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


def _commit_staged_merge(git: RealGitOps, repo: Path) -> None:
    """Commit the staged merge preview bound to its own current tree (the gate-green happy path)."""
    git.commit_merge(expected_tree=_git(repo, "write-tree").strip())


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


def test_changed_entries_detects_unmodified_copy(repo: Path) -> None:
    """Finding #4: ``--find-copies-harder`` must catch an UNMODIFIED file copied to a new path — the
    dangerous case (copy a denylisted file byte-for-byte to an allowlisted dest, source untouched).
    This exercises the REAL git plumbing, not the pure ``evaluate_diff`` unit, proving the copy is
    reported as a ``C`` entry with ``old_path`` set so the dual-path guard can reject it.
    """
    git = RealGitOps(repo)
    _git(repo, "checkout", "-q", "-b", "copy-branch")
    # Copy README.md byte-for-byte to a new path WITHOUT modifying the original.
    (repo / "copy.md").write_text((repo / "README.md").read_text())
    _git(repo, "add", "copy.md")
    _git(repo, "commit", "-q", "-m", "copy readme unmodified")
    base = git.merge_base("main", "copy-branch")
    tip = git.resolve("copy-branch")
    entries = git.changed_entries(base, tip)
    copies = [e for e in entries if e.change_type.startswith("C")]
    assert copies, entries
    assert copies[0].old_path == "README.md"
    assert copies[0].new_path == "copy.md"


def test_changed_entries_demotes_empty_copy_to_add(repo: Path) -> None:
    """A zero-byte file "copies" ANY other empty file under ``--find-copies-harder``, so a fresh
    family ``__init__.py`` was reported as ``C100`` of some pre-existing empty file (e.g.
    ``algua/audit/__init__.py``) and the dual-path guard vetoed a perfectly allowlisted add — the
    false positive that terminal-failed every new-family research branch. An empty DESTINATION
    blob carries no launderable content, so the parser demotes it to a plain ``A``.
    """
    git = RealGitOps(repo)
    # Pre-existing empty file on main — the copy-source git will pair the new empty file with.
    (repo / "existing_empty.py").write_text("")
    _git(repo, "add", "existing_empty.py")
    _git(repo, "commit", "-q", "-m", "empty file on main")
    _git(repo, "checkout", "-q", "-b", "newfam-branch")
    (repo / "algua" / "strategies" / "newfam").mkdir(parents=True)
    init = repo / "algua" / "strategies" / "newfam" / "__init__.py"
    init.write_text("")
    _git(repo, "add", "algua/strategies/newfam/__init__.py")
    _git(repo, "commit", "-q", "-m", "new family init")
    base = git.merge_base("main", "newfam-branch")
    tip = git.resolve("newfam-branch")
    entries = git.changed_entries(base, tip)
    (entry,) = [e for e in entries if e.new_path == "algua/strategies/newfam/__init__.py"]
    assert entry.change_type == "A", entries
    assert entry.old_path is None
    # And the policy verdict flips: the demoted add is allowlisted.
    from algua.operator.diff_policy import evaluate_diff

    result = evaluate_diff([entry], codeowners_text="/algua/registry/store.py @owner\n")
    assert result.ok, result


def test_changed_entries_keeps_nonempty_copy_and_empty_rename(repo: Path) -> None:
    """The demotion is surgical: a NON-empty copy keeps its ``C`` + source path (the R5 guard's
    real target), and a RENAME of an empty file keeps its ``R`` + source path (a rename deletes
    its source — a denied-path deletion must stay visible)."""
    git = RealGitOps(repo)
    (repo / "empty_on_main.py").write_text("")
    _git(repo, "add", "empty_on_main.py")
    _git(repo, "commit", "-q", "-m", "empty on main")
    _git(repo, "checkout", "-q", "-b", "mixed-branch")
    (repo / "copy.md").write_text((repo / "README.md").read_text())  # non-empty copy
    _git(repo, "add", "copy.md")
    _git(repo, "mv", "empty_on_main.py", "renamed_empty.py")  # empty rename
    _git(repo, "commit", "-q", "-m", "non-empty copy + empty rename")
    base = git.merge_base("main", "mixed-branch")
    tip = git.resolve("mixed-branch")
    entries = git.changed_entries(base, tip)
    copies = [e for e in entries if e.change_type.startswith("C")]
    assert copies and copies[0].old_path == "README.md"
    renames = [e for e in entries if e.change_type.startswith("R")]
    assert renames and renames[0].old_path == "empty_on_main.py"


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
    _commit_staged_merge(git, repo)
    merge_sha = git.merge_commit_of(tip)
    assert git.commit_second_parent(merge_sha) == tip

    git.push_cas(merge_sha, base)                       # real remote CAS
    assert git.remote_tip("main") == merge_sha
    assert git.is_ancestor(merge_sha, "refs/remotes/origin/main")

    # Effective-presence content check against origin/main.
    captured = git.tree_blobs(merge_sha, ["algua/strat.py"])
    assert git.blob_at("main", "algua/strat.py") == captured["algua/strat.py"]


def test_commit_merge_drifted_index_raises_and_does_not_commit(repo: Path) -> None:
    """Gated-tree binding (PR #552 review MEDIUM): the gate blesses a ``write-tree`` snapshot;
    ``commit_merge`` must refuse to commit an index that no longer writes that tree (an external
    ``git add`` in the gate→commit window) — raise, no commit, staged merge left intact."""
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    gated_tree = _git(repo, "write-tree").strip()
    head_before = git.resolve("HEAD")
    # External index mutation AFTER the gate snapshot: stage an extra (ungated) file.
    (repo / "sneaky.py").write_text("EVIL = 1\n")
    _git(repo, "add", "sneaky.py")
    with pytest.raises(RuntimeError, match="gate-blessed"):
        git.commit_merge(expected_tree=gated_tree)
    # Fail closed, no half-commit: HEAD unchanged and the merge is still staged (MERGE_HEAD live,
    # the preview's content still in the index) for the next cycle's abort_merge to clean up.
    assert git.resolve("HEAD") == head_before
    assert git.merge_in_progress() is True
    staged = _git(repo, "diff", "--cached", "--name-only")
    assert "algua/strat.py" in staged and "sneaky.py" in staged


def test_push_cas_stale_base_rejects(repo: Path) -> None:
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    _commit_staged_merge(git, repo)
    merge_sha = git.merge_commit_of(tip)
    # A wrong expected_base (origin/main has not moved to it) fails the pre-push CAS.
    with pytest.raises(RemoteMovedError):
        git.push_cas(merge_sha, "0" * 40)


def test_is_merge_of_true_for_merge_false_otherwise(repo: Path) -> None:
    # MEDIUM-1 adoption seam: is_merge_of is True only for a merge commit whose 2nd parent matches,
    # and SAFE (False, not a raise) for a non-merge commit with no ^2.
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    _commit_staged_merge(git, repo)
    merge_sha = git.merge_commit_of(tip)
    assert git.is_merge_of(merge_sha, tip) is True
    assert git.is_merge_of(merge_sha, "0" * 40) is False   # wrong second parent
    # A plain (non-merge) commit has no ^2: safe False, never a CalledProcessError.
    assert git.is_merge_of(git.resolve("HEAD~1"), tip) is False


def test_revert_merge_returns_sha_and_undoes_code(repo: Path) -> None:
    git = RealGitOps(repo)
    tip = git.resolve("feature")
    git.begin_merge(tip)
    _commit_staged_merge(git, repo)
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
