from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from algua.provenance import lockfile

_ROOT = Path(__file__).resolve().parents[2]


def runtime_stamps() -> dict[str, str | None]:
    return {
        "code_hash": _code_hash(),
        "dependency_hash": lockfile.dependency_hash(),
    }


def _code_hash() -> str | None:
    head = _git_text(["rev-parse", "HEAD"])
    if head is None:
        return None
    status = _git_bytes(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    if status is None:
        # git status was unavailable (timeout/error) — DON'T equate that with a clean
        # workspace, or a dirty tree would be silently stamped as the bare clean HEAD (#256).
        return None
    if not status:  # b"" => genuinely clean
        return head
    dirty_hash = _dirty_workspace_hash(status)
    if dirty_hash is None:
        # An input to the dirty-tree hash (diff / ls-files) was unavailable — the same
        # principle as above: don't emit a confident dirty stamp from incomplete evidence.
        return None
    return f"{head}-dirty-{dirty_hash[:16]}"


def _dirty_workspace_hash(status: bytes) -> str | None:
    # If any dirty-tree input is unavailable (timeout/error -> None), return None rather than
    # treating it as empty, which would hash incomplete evidence into a confident stamp (#256).
    diff = _git_bytes(["diff", "HEAD", "--binary"])
    if diff is None:
        return None
    untracked = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"])
    if untracked is None:
        return None
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(diff)
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        digest.update(raw_path)
        path = _ROOT / os.fsdecode(raw_path)
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_text(args: list[str]) -> str | None:
    output = _git_bytes(args)
    if output is None:
        return None
    value = output.decode().strip()
    return value or None


def _git_bytes(args: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout
