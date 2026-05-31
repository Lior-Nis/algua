from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def runtime_stamps() -> dict[str, str | None]:
    return {
        "code_hash": _code_hash(),
        "dependency_hash": _file_hash(_ROOT / "uv.lock"),
    }


def _code_hash() -> str | None:
    head = _git_text(["rev-parse", "HEAD"])
    if head is None:
        return None
    status = _git_bytes(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    if not status:
        return head
    dirty_hash = _dirty_workspace_hash(status)
    return f"{head}-dirty-{dirty_hash[:16]}"


def _dirty_workspace_hash(status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(_git_bytes(["diff", "HEAD", "--binary"]) or b"")
    untracked = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"]) or b""
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


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
