"""Filesystem-backed model-artifact registry — the immutable system-of-record for trained model
versions (issue #376).

Layout under `root` (default `<data_dir>/models` via `default_root()`):

    <root>/<name>/manifest.jsonl        # one append-only row per version (authoritative)
    <root>/<name>/v<N>/artifact.bin     # the raw artifact bytes for version N
    <root>/<name>.lock                  # per-name flock lease serializing register()

Immutability + integrity (all FAIL CLOSED on read):
  - the manifest is authoritative for which versions are VALID; a `vN/` dir without a manifest
    row is a torn/aborted write and is never served;
  - an artifact whose recomputed sha256 != the row's `digest` is rejected (bytes rewritten);
  - a row whose recomputed `provenance_digest` != the stored one is rejected (metadata rewritten);
  - duplicate version rows, or a row with a missing/short artifact, are rejected.

Concurrency + torn-write recovery (under the per-name flock lease):
  `next = max(existing vN/ dir numbers, manifest versions) + 1` — computed off BOTH the directory
  listing AND the manifest, so a version left dangling by a crash between the artifact rename and
  the manifest append can never be re-selected (its number is already reserved by the dir). Gaps
  are allowed; recovery is passive (no scrubber). The artifact is written into a unique temp dir
  and atomically renamed into place, then the manifest row is appended and fsynced.

This module is a leaf: it imports only stdlib + `algua.contracts`. The fsync/flock helpers are
inlined (rather than importing `algua.data`) to keep the model layer a clean leaf under the
import-linter boundary. Threat model is a single local Linux filesystem (see `algua.data.files`).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algua.contracts.model_types import ModelVersion, compute_provenance_digest

_ARTIFACT = "artifact.bin"
_MANIFEST = "manifest.jsonl"


class ModelRegistryError(Exception):
    """Any model-registry failure — a torn/absent version, an integrity mismatch, or a
    registration conflict. Callers fail closed on it (never silently continue)."""


def default_root() -> Path:
    """The registry root under the configured data dir. Imported lazily so this leaf module does
    not pull `algua.config` at import time (keeps the import graph clean)."""
    from algua.config.settings import get_settings

    return Path(get_settings().data_dir) / "models"


# --------------------------------------------------------------------------- #
# durability primitives (inlined — see module docstring)
# --------------------------------------------------------------------------- #

def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _name_lease(name_dir: Path) -> Iterator[None]:
    """Exclusive per-name flock on a sibling `<name>.lock`, serializing register()."""
    name_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = name_dir.parent / f"{name_dir.name}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #

def _validate_name(name: str) -> None:
    # Names become a directory component; reject anything that could escape the root or collide
    # with the lock/temp naming. Fail closed rather than sanitize silently.
    if not name or name != name.strip() or "/" in name or "\\" in name or name.startswith("."):
        raise ModelRegistryError(f"invalid model name {name!r}")


_REQUIRED_ROW_KEYS = frozenset({"name", "version", "digest", "created_at", "training_as_of",
                                "provenance_digest"})


def _read_manifest_rows(name_dir: Path) -> list[dict[str, Any]]:
    manifest = name_dir / _MANIFEST
    if not manifest.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(manifest.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelRegistryError(
                f"model {name_dir.name!r}: corrupt manifest at line {lineno}: {exc}"
            ) from exc
        missing = _REQUIRED_ROW_KEYS - row.keys()
        if missing:
            raise ModelRegistryError(
                f"model {name_dir.name!r}: manifest row {lineno} missing keys {sorted(missing)}"
            )
        rows.append(row)
    return rows


def _existing_dir_versions(name_dir: Path) -> list[int]:
    if not name_dir.exists():
        return []
    out: list[int] = []
    for p in name_dir.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
            out.append(int(p.name[1:]))
    return out


def _row_to_version(name_dir: Path, row: dict[str, Any]) -> ModelVersion:
    return ModelVersion(
        name=row["name"],
        version=row["version"],
        digest=row["digest"],
        created_at=row["created_at"],
        training_snapshot_id=row.get("training_snapshot_id"),
        training_as_of=row["training_as_of"],
        code_hash=row.get("code_hash"),
        hyperparameters=row.get("hyperparameters", {}),
        seed=row.get("seed"),
        eval_report=row.get("eval_report", {}),
        artifact_path=str(name_dir / f"v{row['version']}" / _ARTIFACT),
        provenance_digest=row["provenance_digest"],
    )


def _verify_row(mv: ModelVersion, artifact_bytes: bytes) -> None:
    """Fail closed unless the artifact + row metadata match the recorded digests (tamper /
    history-rewrite detection)."""
    actual_digest = hashlib.sha256(artifact_bytes).hexdigest()[:16]
    if actual_digest != mv.digest:
        raise ModelRegistryError(
            f"model {mv.name!r} v{mv.version}: artifact digest {actual_digest} != recorded "
            f"{mv.digest} (artifact was modified)"
        )
    expected_prov = compute_provenance_digest(
        digest=mv.digest,
        training_snapshot_id=mv.training_snapshot_id,
        training_as_of=mv.training_as_of,
        code_hash=mv.code_hash,
        hyperparameters=mv.hyperparameters,
        seed=mv.seed,
        eval_report=mv.eval_report,
    )
    if expected_prov != mv.provenance_digest:
        raise ModelRegistryError(
            f"model {mv.name!r} v{mv.version}: provenance digest mismatch "
            f"(manifest metadata was modified)"
        )


def _resolve_row(name_dir: Path, version: int) -> dict[str, Any]:
    rows = [r for r in _read_manifest_rows(name_dir) if r.get("version") == version]
    if not rows:
        raise ModelRegistryError(
            f"model {name_dir.name!r} v{version} not found (no manifest row)"
        )
    if len(rows) > 1:
        raise ModelRegistryError(
            f"model {name_dir.name!r} v{version}: duplicate manifest rows (immutable history "
            f"violated)"
        )
    return rows[0]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def register(
    name: str,
    artifact_bytes: bytes,
    *,
    training_snapshot_id: str | None,
    training_as_of: str,
    code_hash: str | None,
    hyperparameters: dict[str, Any] | None = None,
    seed: int | None = None,
    eval_report: dict[str, Any] | None = None,
    root: Path | None = None,
) -> ModelVersion:
    """Register `artifact_bytes` as a new immutable version of model `name`. Assigns the next
    integer version (gaps allowed), stores the bytes content-addressed by digest, and appends a
    manifest row carrying the full training provenance. Returns the `ModelVersion`.

    `training_as_of` is the LATEST information the model may have seen — the point-in-time cutoff
    the backtest PIT guard enforces. Fails closed (`ModelRegistryError`) on an empty artifact or
    an invalid name; serialized per-name so concurrent registrations never collide on a version.
    """
    _validate_name(name)
    if not artifact_bytes:
        raise ModelRegistryError(f"model {name!r}: refusing to register an empty artifact")
    root = (root or default_root()).resolve()
    name_dir = root / name
    hyperparameters = dict(hyperparameters or {})
    eval_report = dict(eval_report or {})

    with _name_lease(name_dir):
        name_dir.mkdir(parents=True, exist_ok=True)
        # Reserve off BOTH the manifest and the dir listing so a dangling torn-write dir is never
        # reused (its number is already counted here).
        reserved = _existing_dir_versions(name_dir) + [
            r["version"] for r in _read_manifest_rows(name_dir)
        ]
        version = (max(reserved) + 1) if reserved else 1

        digest = hashlib.sha256(artifact_bytes).hexdigest()[:16]
        provenance_digest = compute_provenance_digest(
            digest=digest,
            training_snapshot_id=training_snapshot_id,
            training_as_of=training_as_of,
            code_hash=code_hash,
            hyperparameters=hyperparameters,
            seed=seed,
            eval_report=eval_report,
        )

        # Write the artifact into a unique temp dir, then atomically rename into vN/.
        staging = name_dir / f"_staging-{uuid.uuid4().hex}"
        staging.mkdir()
        artifact_tmp = staging / _ARTIFACT
        artifact_tmp.write_bytes(artifact_bytes)
        _fsync_file(artifact_tmp)
        _fsync_dir(staging)
        version_dir = name_dir / f"v{version}"
        os.rename(staging, version_dir)
        _fsync_dir(name_dir)

        created_at = datetime.now(tz=UTC).isoformat()
        row = {
            "name": name,
            "version": version,
            "digest": digest,
            "created_at": created_at,
            "training_snapshot_id": training_snapshot_id,
            "training_as_of": training_as_of,
            "code_hash": code_hash,
            "hyperparameters": hyperparameters,
            "seed": seed,
            "eval_report": eval_report,
            "provenance_digest": provenance_digest,
        }
        manifest = name_dir / _MANIFEST
        with manifest.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(name_dir)

        return _row_to_version(name_dir, row)


def _resolve_verified(name_dir: Path, version: int) -> tuple[ModelVersion, bytes]:
    """Resolve one version's row + artifact bytes in a SINGLE read and verify them together (no
    TOCTOU between the metadata read and the bytes read). Fail closed on a missing artifact,
    digest mismatch, or provenance mismatch."""
    row = _resolve_row(name_dir, version)
    mv = _row_to_version(name_dir, row)
    artifact = name_dir / f"v{version}" / _ARTIFACT
    if not artifact.is_file():
        raise ModelRegistryError(
            f"model {name_dir.name!r} v{version}: manifest row present but artifact missing at "
            f"{artifact}"
        )
    data = artifact.read_bytes()
    _verify_row(mv, data)
    return mv, data


def get_version(name: str, version: int, root: Path | None = None) -> ModelVersion:
    """Resolve one immutable version's manifest row + artifact path. Verifies the artifact bytes
    and the row metadata against their recorded digests (fail closed on any mismatch)."""
    _validate_name(name)
    root = (root or default_root()).resolve()
    mv, _ = _resolve_verified(root / name, version)
    return mv


def get_version_with_bytes(
    name: str, version: int, root: Path | None = None
) -> tuple[ModelVersion, bytes]:
    """Resolve a version's metadata AND artifact bytes atomically (one read), both verified. The
    strategy loader uses this so the returned bytes provably belong to the returned/validated
    version — no window in which metadata and bytes could diverge."""
    _validate_name(name)
    root = (root or default_root()).resolve()
    return _resolve_verified(root / name, version)


def list_versions(name: str, root: Path | None = None) -> list[ModelVersion]:
    """All VALID versions of `name`, ascending — for rollback identification. Each is fully
    verified (artifact present + digest + provenance); torn-write dirs (no manifest row) are
    excluded because the manifest is authoritative. Fails closed if any listed version is
    tampered/corrupt (the 'valid versions' contract)."""
    _validate_name(name)
    root = (root or default_root()).resolve()
    name_dir = root / name
    rows = _read_manifest_rows(name_dir)
    seen: set[int] = set()
    out: list[ModelVersion] = []
    for row in sorted(rows, key=lambda r: r["version"]):
        v = row["version"]
        if v in seen:
            raise ModelRegistryError(
                f"model {name!r}: duplicate manifest row for v{v} (immutable history violated)"
            )
        seen.add(v)
        mv, _ = _resolve_verified(name_dir, v)
        out.append(mv)
    return out


def load_artifact_bytes(name: str, version: int, root: Path | None = None) -> bytes:
    """The raw artifact bytes for a version, with the same fail-closed integrity checks as
    `get_version` (digest + provenance)."""
    _validate_name(name)
    root = (root or default_root()).resolve()
    _, data = _resolve_verified(root / name, version)
    return data
