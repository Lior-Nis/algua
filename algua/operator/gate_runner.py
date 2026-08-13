"""Hermetic quality-gate runner for the merge-back saga (#485, factory hardening).

The first end-to-end factory drains (2026-08-13) proved that running the quality gate **inside the
live operational checkout** cannot work: the surrounding saga holds ``merge_back.lock`` (anchored at
the live repo's git dir), so every test that exercises the merge-back CLI fails with "another
merge-back cycle is in progress"; the live ``.env`` (real paper credentials) breaks the doctor
tests' clean-environment assumptions; and the staged-but-uncommitted merge makes the tree dirty,
breaking code-hash determinism assertions. Patching individual tests would leave the class open —
any future ambient-state-sensitive test would silently re-break the factory's gate.

Instead the gate runs in a **throwaway worktree of the staged merge**:

1. ``git write-tree`` snapshots the staged index (the ``--no-ff --no-commit`` merge preview).
2. ``git commit-tree`` wraps it in an UNREFERENCED temp commit (plain garbage if the gate is red —
   nothing to clean up; the saga's real ``commit_merge`` still creates the durable merge commit on
   green, preserving "commit only on green").
3. ``git worktree add --detach`` checks it out in a temp dir: clean tree, no ``.env``, no ``data/``,
   and a PER-WORKTREE git dir — so ``merge_back.lock``-taking tests lock their own anchor, never
   colliding with the live cycle's held lock.
4. ``uv sync --frozen`` provisions the venv (fast — hardlinked from uv's cache; same lockfile), then
   the gate commands run there with a SCRUBBED environment (no ``ALGUA_*``/``ALPACA_*``/
   ``VIRTUAL_ENV``/``PYTHONPATH`` leakage from the operator service) and their stdout routed to the
   parent's STDERR — gate noise stays visible in journald without polluting the parent CLI's
   stdout-JSON contract.
5. The worktree is removed in a ``finally`` — red or green, and on any exception. A hard kill
   (SIGKILL) mid-gate can still leak: ``git worktree prune`` on the next run reaps only the stale
   ADMIN entry once the directory is already gone; the leaked ``/tmp`` directory itself is left for
   the OS tmp reaper.

The gate is BINDING on the exact tree it ran: a green run returns the gated ``write-tree`` sha so
the saga can pin the merge commit to it (``commit_merge(expected_tree=...)``) — an index mutated
between gate and commit can never land ungated.

Fail-closed: ANY failure — plumbing, sync, or a gate command — is a red gate (``None``), never an
exception the saga would misread.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

# The full quality gate, short-circuiting on the first failure. ``--frozen`` everywhere: the gate
# must run the LOCKED dependency set (the root uv.lock is dependency_hash identity) — a gate that
# silently resolved fresh deps would test a different world than the one that ships.
_GATE_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "--frozen", "pytest", "-q"),
    ("uv", "run", "--frozen", "ruff", "check", "."),
    ("uv", "run", "--frozen", "mypy", "algua"),
    ("uv", "run", "--frozen", "lint-imports"),
)

# Environment keys that leak live-operator state into what must be a hermetic test run. ALGUA_*
# redirects DB/data/kb paths; ALPACA_* is real broker credentials (the doctor tests assert their
# ABSENCE in a clean env); VIRTUAL_ENV would point uv at the LIVE checkout's venv instead of the
# gate worktree's own; PYTHONPATH would shadow the gate worktree's modules with the live
# checkout's; PYTEST_ADDOPTS would silently reshape the gate's pytest invocation.
_SCRUB_PREFIXES = ("ALGUA_", "ALPACA_")
_SCRUB_EXACT = frozenset(
    {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "PYTEST_ADDOPTS"})

# Gate children write their stdout to THIS file descriptor — the parent's stderr — so build/test
# noise stays visible (journald under the operator service) without polluting the parent CLI's
# stdout-JSON contract. The literal fd 2 rather than ``sys.stderr.fileno()``: test harnesses
# (pytest capture) replace ``sys.stderr`` with objects whose ``fileno()`` raises, while fd 2 is
# always the process's real stderr stream.
_STDERR_FD = 2


def _scrubbed_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items()
        if not k.startswith(_SCRUB_PREFIXES) and k not in _SCRUB_EXACT
    }


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True)


def run_hermetic_quality_gate(
    repo_root: Path, *, commands: Sequence[Sequence[str]] | None = None,
) -> str | None:
    """Gate the STAGED tree of ``repo_root`` in a throwaway detached worktree; returns the gated
    TREE SHA (the ``git write-tree`` snapshot the gate ran on) iff all green, else None.

    The caller must bind the merge commit to the returned sha
    (``commit_merge(expected_tree=...)``): the gate blesses a SNAPSHOT of the index, so committing
    a live index that no longer writes that tree would land ungated content.

    Must be called while the merge preview is staged (the saga's ``run_gate`` moment). ``commands``
    is a test seam — production always uses the full ``_GATE_COMMANDS`` gate. Injected commands
    also SKIP the ``uv sync`` provisioning step (they run bare in the worktree; only the real gate
    needs the locked venv), keeping the hermetic-property tests fast and uv-free — and return the
    gated tree sha on success exactly like the real gate.
    """
    gate_commands = _GATE_COMMANDS if commands is None else commands
    gate_dir: Path | None = None
    try:
        # Reap any leaked gate worktree from a previous crashed cycle before creating ours.
        _git(repo_root, "worktree", "prune")
        tree = _git(repo_root, "write-tree").stdout.strip()
        tmp_commit = _git(
            repo_root, "commit-tree", tree, "-p", "HEAD", "-m", "merge-back gate preview",
        ).stdout.strip()
        gate_dir = Path(tempfile.mkdtemp(prefix="algua-merge-gate-"))
        # mkdtemp creates the dir; `git worktree add` insists on creating the leaf itself.
        gate_dir.rmdir()
        _git(repo_root, "worktree", "add", "--detach", str(gate_dir), tmp_commit)
        env = _scrubbed_env()
        if commands is None:
            # Provision the gate venv from the LOCKED set before any gate command runs.
            sync = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["uv", "sync", "--frozen", "-q"], cwd=gate_dir, env=env, stdout=_STDERR_FD)
            if sync.returncode != 0:
                return None
        for cmd in gate_commands:
            proc = subprocess.run(  # noqa: S603 — fixed/injected argv, no shell
                list(cmd), cwd=gate_dir, env=env, stdout=_STDERR_FD)
            if proc.returncode != 0:
                return None
        return tree
    except (OSError, subprocess.CalledProcessError):
        return None  # fail closed: a gate that cannot run is a red gate, never a pass
    finally:
        if gate_dir is not None:
            subprocess.run(  # noqa: S603 — best-effort cleanup; prune on next run is the backstop
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(gate_dir)],
                capture_output=True, text=True, check=False)
