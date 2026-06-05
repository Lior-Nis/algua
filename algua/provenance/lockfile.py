from __future__ import annotations

import hashlib
from pathlib import Path

# Repo root: algua/provenance/dependency_hash.py -> parents[2] is the project root that holds
# uv.lock. Kept here (not in backtest) so the registry can pin the same locked-dependency set
# the backtest stamps record, without the registry importing the backtest engine.
_ROOT = Path(__file__).resolve().parents[2]


def dependency_hash() -> str | None:
    """Hash the locked dependency set (``uv.lock``).

    This is the SINGLE source of truth for the dependency identity: the backtest
    reproducibility stamps and the live-approval gate both call it, so a ``uv.lock`` bump that
    can change fill or numerical semantics shifts the hash for both at once. Returns ``None`` if
    the lockfile is absent (no deterministic identity to pin)."""
    return _file_hash(_ROOT / "uv.lock")


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
