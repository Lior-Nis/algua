from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def runtime_stamps() -> dict[str, str | None]:
    return {
        "code_hash": _git_head(),
        "dependency_hash": _file_hash(_ROOT / "uv.lock"),
    }


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
