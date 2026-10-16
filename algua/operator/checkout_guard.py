"""Clean-``main`` checkout guard for the unattended operator lanes (#687).

The systemd units run from the box's main checkout. In September an agent left that checkout on a
story branch, so production ran unmerged code and nothing noticed. ``paper run-all`` / ``live
run-all`` therefore REFUSE to start unless the checkout is on ``main`` with no tracked
modifications, and ``fleet health`` reports the same condition so an external watchdog sees it.

Policy (decided, documented, no bypass):
* Applies to BOTH ``--refresh`` and ``--snapshot`` replay: a replay still ticks the real lane
  against the broker, so unmerged code is exactly as dangerous there.
* Untracked files do not count; tracked modifications (staged or not) do. A detached HEAD, an
  unreadable repo, or a missing ``git`` all FAIL CLOSED.
* There is NO environment variable or flag to skip it (an agent could set one). The git probe is
  injected: tests replace :func:`default_probe` (``tests/conftest.py`` installs a clean-main fake
  for the whole suite, so worktree-based test runs are unaffected); production uses
  :class:`GitCheckoutProbe` on the checkout the code is imported from.
Imports nothing from ``algua``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Protocol

REQUIRED_BRANCH = "main"


class CheckoutNotCleanMain(RuntimeError):
    """The operator checkout is not a clean ``main`` (error code ``checkout_not_clean_main``)."""


class CheckoutProbe(Protocol):
    def branch(self) -> str: ...
    def tracked_dirty(self) -> bool: ...


class GitCheckoutProbe:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.repo_root), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout

    def branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def tracked_dirty(self) -> bool:
        return self._git("status", "--porcelain", "--untracked-files=no").strip() != ""


def default_probe() -> CheckoutProbe:
    """The probe for the checkout this code runs from. Tests monkeypatch THIS function."""
    return GitCheckoutProbe(Path(__file__).resolve().parents[2])


def checkout_status(probe: CheckoutProbe | None = None) -> dict[str, Any]:
    """``{ok, branch, tracked_dirty, reason}``; any probe failure is ``ok: False`` (fail closed)."""
    try:
        p = probe if probe is not None else default_probe()
        branch, dirty = p.branch(), p.tracked_dirty()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "branch": None, "tracked_dirty": None,
                "reason": f"cannot read git state: {exc}"}
    if branch != REQUIRED_BRANCH:
        reason: str | None = f"checkout is on {branch!r}, not {REQUIRED_BRANCH!r}"
    elif dirty:
        reason = "checkout has tracked modifications"
    else:
        reason = None
    return {"ok": reason is None, "branch": branch, "tracked_dirty": dirty, "reason": reason}


def require_clean_main(probe: CheckoutProbe | None = None) -> None:
    status = checkout_status(probe)
    if not status["ok"]:
        raise CheckoutNotCleanMain(
            f"operator checkout must be a clean {REQUIRED_BRANCH!r}: {status['reason']}")
