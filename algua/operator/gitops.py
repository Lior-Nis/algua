"""The git seam the merge-back orchestrator drives, plus the repo-global flock (#485, Task 3).

``GitOps`` is a Protocol so :func:`algua.operator.mergeback.run_merge_back` is a pure state machine
testable with a ``FakeGit``; ``RealGitOps`` is the subprocess-backed implementation. Branch-tip-SHA
identity, second-parent merge verification, the freshly-fetched ``origin/main`` content check
(C4/R1), and the remote compare-and-swap push (R2) all live here. Imports nothing from ``algua``.
"""

from __future__ import annotations

import contextlib
import fcntl
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from algua.operator.diff_policy import DiffEntry


class GitOps(Protocol):
    """Git operations over the one shared working tree, injected for testability."""

    def current_branch(self) -> str: ...
    def working_tree_clean(self) -> bool: ...
    def merge_in_progress(self) -> bool: ...
    def abort_merge(self) -> None: ...
    def resolve(self, ref: str) -> str:
        """SHA of ``ref`` (a branch name, tag, or revision)."""
        ...

    def fetch_remote(self, ref: str) -> None:
        """``git fetch origin <ref>`` — refresh ``refs/remotes/origin/<ref>`` from origin (R1)."""
        ...

    def remote_tip(self, ref: str) -> str:
        """SHA of ``refs/remotes/origin/<ref>`` (call after :meth:`fetch_remote`) (R1)."""
        ...

    def merge_base(self, a: str, b: str) -> str: ...
    def changed_entries(self, base: str, tip: str) -> list[DiffEntry]:
        """``git diff --raw -M -C --find-copies-harder base tip`` parsed into :class:`DiffEntry`
        tuples (R5). ``--find-copies-harder`` makes git treat UNMODIFIED files as copy sources, so
        a branch that copies a denylisted file byte-for-byte to an allowlisted path is reported as a
        ``C`` entry (with ``old_path`` set) and caught by the dual-path guard."""
        ...

    def begin_merge(self, tip: str) -> None:
        """``git merge --no-ff --no-commit <tip>`` — stage a merge preview."""
        ...

    def commit_merge(self) -> None: ...
    def merge_commit_of(self, tip: str) -> str:
        """SHA of the ``--no-ff`` merge commit whose 2nd parent equals ``tip``."""
        ...

    def commit_second_parent(self, sha: str) -> str:
        """SHA of ``<sha>^2`` (the merged branch tip)."""
        ...

    def is_merge_of(self, sha: str, expected_second_parent: str) -> bool:
        """True iff ``sha`` is a merge commit whose 2nd parent equals ``expected_second_parent``.

        SAFE where :meth:`commit_second_parent` is not: a non-merge commit (no ``^2``) returns
        False instead of raising. Used to ADOPT an unjournaled local merge on resume (a crash
        between ``commit_merge`` and journaling ``merge_sha`` — MEDIUM-1) by verifying the drifted
        local ``main`` HEAD really is THIS attempt's merge of the branch tip, not external drift."""
        ...

    def is_ancestor(self, sha: str, ref: str) -> bool:
        """True iff ``sha`` is an ancestor of ``ref`` (``git merge-base --is-ancestor``)."""
        ...

    def push_cas(self, merge_sha: str, expected_base: str) -> None:
        """Remote compare-and-swap push (R2): fetch ``origin/main``, assert it still equals
        ``expected_base``, ``git push origin <merge_sha>:refs/heads/main``, re-fetch and assert the
        pushed SHA is now live. Raises :class:`RemoteMovedError` on any CAS failure."""
        ...

    def revert_merge(self, merge_sha: str) -> str:
        """``git revert -m 1 <merge_sha> --no-edit`` and return the revert commit SHA."""
        ...

    def push_revert(self, revert_sha: str, expected_merge_sha: str) -> None:
        """Remote compare-and-swap push of a revert commit (finding #3), mirroring
        :meth:`push_cas`: fetch ``origin/main``, assert it still equals ``expected_merge_sha`` (the
        reverted merge is still the live remote tip), ``git push origin <revert_sha>:refs/heads/
        main``, re-fetch and assert the revert is now live. Raises :class:`RemoteMovedError` on any
        CAS failure, so a concurrent remote advance during the revert push is detected (a resume can
        then distinguish 'stale, needs recompute' from a transient retry) instead of surfacing an
        opaque non-fast-forward ``CalledProcessError``."""
        ...

    def tree_blobs(self, sha: str, paths: list[str]) -> dict[str, str]:
        """For each path present in ``<sha>``'s tree, its blob object id (missing paths omitted)."""
        ...

    def blob_at(self, ref: str, path: str) -> str | None:
        """Blob object id of ``<ref>:<path>`` (``refs/remotes/origin/main`` for R1), or None if
        absent."""
        ...


class RemoteMovedError(RuntimeError):
    """Raised when a remote compare-and-swap detects ``origin/main`` moved under the driver (R2)."""


class RealGitOps:
    """Subprocess-backed :class:`GitOps`, shelling ``git -C <repo_root> ...``.

    Every mutating call uses ``check=True`` so a git failure fails closed (raises
    ``subprocess.CalledProcessError``) rather than being silently swallowed. ``origin`` is the
    authoritative remote; ``fetch_remote``/``remote_tip`` read ``refs/remotes/origin/<ref>``.
    """

    def __init__(self, repo_root: Path, *, remote: str = "origin") -> None:
        self.repo_root = repo_root
        self.remote = remote

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repo_root), *args],
            check=check, capture_output=True, text=True,
        )

    def current_branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def working_tree_clean(self) -> bool:
        return self._run(["status", "--porcelain"]).stdout.strip() == ""

    def merge_in_progress(self) -> bool:
        if (self.repo_root / ".git" / "MERGE_HEAD").exists():
            return True
        return self._run(["rev-parse", "-q", "--verify", "MERGE_HEAD"], check=False).returncode == 0

    def abort_merge(self) -> None:
        self._run(["merge", "--abort"])

    def resolve(self, ref: str) -> str:
        return self._run(["rev-parse", ref]).stdout.strip()

    def fetch_remote(self, ref: str) -> None:
        self._run(["fetch", self.remote, ref])

    def remote_tip(self, ref: str) -> str:
        return self._run(["rev-parse", f"refs/remotes/{self.remote}/{ref}"]).stdout.strip()

    def merge_base(self, a: str, b: str) -> str:
        return self._run(["merge-base", a, b]).stdout.strip()

    def changed_entries(self, base: str, tip: str) -> list[DiffEntry]:
        # `git diff --raw -M -C` lines look like:
        #   :100644 100644 <src> <dst> M\tpath
        #   :100644 100644 <src> <dst> R100\told\tnew
        raw = self._run(
            ["diff", "--raw", "-M", "-C", "--find-copies-harder", "-z", base, tip]).stdout
        return _parse_raw_z(raw)

    def begin_merge(self, tip: str) -> None:
        self._run(["merge", "--no-ff", "--no-commit", tip])

    def commit_merge(self) -> None:
        self._run(["commit", "--no-edit"])

    def merge_commit_of(self, tip: str) -> str:
        merges = self._run(["rev-list", "--merges", "--max-count=200", "HEAD"]).stdout.split()
        for merge in merges:
            if self._run(["rev-parse", f"{merge}^2"]).stdout.strip() == tip:
                return merge
        raise RuntimeError(f"no merge commit of tip {tip} in recent HEAD history")

    def commit_second_parent(self, sha: str) -> str:
        return self._run(["rev-parse", f"{sha}^2"]).stdout.strip()

    def is_merge_of(self, sha: str, expected_second_parent: str) -> bool:
        # `rev-parse -q --verify <sha>^2` exits non-zero (no output) for a non-merge commit, so a
        # commit with no second parent is False rather than a raised CalledProcessError.
        r = self._run(["rev-parse", "-q", "--verify", f"{sha}^2"], check=False)
        return r.returncode == 0 and r.stdout.strip() == expected_second_parent

    def is_ancestor(self, sha: str, ref: str) -> bool:
        return self._run(["merge-base", "--is-ancestor", sha, ref], check=False).returncode == 0

    def push_cas(self, merge_sha: str, expected_base: str) -> None:
        self.fetch_remote("main")
        if self.remote_tip("main") != expected_base:
            raise RemoteMovedError(
                f"origin/main moved before push (expected base {expected_base}, "
                f"found {self.remote_tip('main')}); merge is stale")
        self._run(["push", self.remote, f"{merge_sha}:refs/heads/main"])
        self.fetch_remote("main")
        if self.remote_tip("main") != merge_sha:
            raise RemoteMovedError(
                f"origin/main is {self.remote_tip('main')} after push, not the pushed "
                f"{merge_sha}; remote moved between push and re-verify")

    def revert_merge(self, merge_sha: str) -> str:
        self._run(["revert", "-m", "1", merge_sha, "--no-edit"])
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def push_revert(self, revert_sha: str, expected_merge_sha: str) -> None:
        self.fetch_remote("main")
        if self.remote_tip("main") != expected_merge_sha:
            raise RemoteMovedError(
                f"origin/main moved before revert push (expected the reverted merge "
                f"{expected_merge_sha}, found {self.remote_tip('main')}); the revert is stale, "
                f"recompute rather than force it")
        self._run(["push", self.remote, f"{revert_sha}:refs/heads/main"])
        self.fetch_remote("main")
        if self.remote_tip("main") != revert_sha:
            raise RemoteMovedError(
                f"origin/main is {self.remote_tip('main')} after revert push, not the pushed "
                f"{revert_sha}; remote moved between push and re-verify")

    def tree_blobs(self, sha: str, paths: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for path in paths:
            r = self._run(["rev-parse", f"{sha}:{path}"], check=False)
            if r.returncode == 0:
                out[path] = r.stdout.strip()
        return out

    def blob_at(self, ref: str, path: str) -> str | None:
        target = f"refs/remotes/{self.remote}/{ref}" if "/" not in ref else ref
        r = self._run(["rev-parse", f"{target}:{path}"], check=False)
        return r.stdout.strip() if r.returncode == 0 else None


def _parse_raw_z(raw: str) -> list[DiffEntry]:
    """Parse ``git diff --raw -M -C -z`` output. The ``-z`` form separates every field (including
    the meta field and each path) with NUL. A meta field looks like ``:<srcmode> <dstmode> <srcsha>
    <dstsha> <status>``; A/M/D/T carry one path, R/C carry two (old then new)."""
    fields = raw.split("\0")
    entries: list[DiffEntry] = []
    i = 0
    while i < len(fields):
        meta = fields[i]
        if not meta:
            i += 1
            continue
        # meta = ":100644 100644 <src> <dst> M100"
        parts = meta.lstrip(":").split()
        dst_mode = parts[1]
        status = parts[4]
        if status.startswith(("R", "C")):
            old_path = fields[i + 1]
            new_path = fields[i + 2]
            entries.append(DiffEntry(dst_mode, status, old_path, new_path))
            i += 3
        else:
            path = fields[i + 1]
            entries.append(DiffEntry(dst_mode, status, path if status.startswith("D") else None,
                                     path))
            i += 2
    return entries


@contextlib.contextmanager
def merge_back_lock(lock_path: Path) -> Iterator[None]:
    """Repo-global exclusive ``flock`` for the whole merge-back saga.

    Mirrors the staging-lease flock discipline in ``algua/data/staging.py``: a non-blocking
    ``LOCK_EX`` on a dedicated marker file. The kernel releases the lock on the holder's death (even
    a hard kill), so a crashed cycle never wedges the next one; a *live* concurrent cycle makes the
    acquire fail, and we **fail closed** rather than mutating the shared checkout under a second
    driver. The lock is always released and the fd closed in a finally.
    """
    handle = open(lock_path, "a")  # noqa: SIM115 — released in the finally below
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError("another merge-back cycle is in progress") from exc
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
