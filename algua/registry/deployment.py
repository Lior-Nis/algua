"""Canonical working-tree deployment descriptors.

This slice records a reproducible descriptor but does not materialize an executable artifact.
The working tree is therefore re-verified before every deployment-aware paper tick.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from algua.contracts.planner import PLANNER_PROTOCOL_VERSION
from algua.registry.repository import ArtifactIdentity

_GENERATED_CACHE_DIR = "__pycache__"
_GENERATED_SUFFIXES = frozenset({".pyc", ".pyo"})
_CACHE_TAG_RE = re.compile(r"^(?P<module>.+?)\.(?:cpython|pypy)-[^.]+(?:\.opt-\d+)?$")


class DeploymentError(ValueError):
    """A deployment descriptor or epoch cannot be trusted."""


@dataclass(frozen=True)
class DeploymentManifest:
    manifest_digest: str
    manifest_json: str
    code_hash: str
    config_hash: str
    dependency_hash: str
    resolved_config_json: str
    universe_name: str | None
    environment_digest: str
    python_implementation: str
    python_version: str
    abi_tag: str
    platform_tag: str
    planner_protocol_version: int
    source_kind: str
    source_ref: str
    asset_digests_json: str


@dataclass(frozen=True)
class PreparedDeployment:
    manifest: DeploymentManifest
    research_gate_id: int


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeploymentError(f"working tree could not be verified: git {' '.join(args)}") from exc
    return proc.stdout.strip()


def _is_generated_from_tracked(path: Path, tracked: set[Path]) -> bool:
    """True only for bytecode mechanically derived from a tracked sibling source file."""
    if path.suffix not in _GENERATED_SUFFIXES or path.parent.name != _GENERATED_CACHE_DIR:
        return False
    match = _CACHE_TAG_RE.match(path.stem)
    module = match.group("module") if match is not None else path.stem
    return path.parent.parent / f"{module}.py" in tracked


def _assert_clean_working_tree(repo_root: Path) -> str:
    source_ref = _run_git(repo_root, "rev-parse", "HEAD")
    dirty = _run_git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise DeploymentError("working tree tracked files do not match recorded HEAD")

    tracked_raw = _run_git(repo_root, "ls-files", "-z", "--", "algua")
    tracked = {Path(p) for p in tracked_raw.split("\0") if p}
    source_root = repo_root / "algua"
    if not source_root.is_dir():
        raise DeploymentError("working tree has no algua source root")
    untracked = [
        path.relative_to(repo_root)
        for path in source_root.rglob("*")
        if path.is_file()
        and path.relative_to(repo_root) not in tracked
        and not _is_generated_from_tracked(path.relative_to(repo_root), tracked)
    ]
    if untracked:
        names = ", ".join(str(p) for p in sorted(untracked)[:3])
        raise DeploymentError(f"working tree contains untracked source/config files: {names}")
    return source_ref


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _asset_entries(asset_paths: tuple[Path, ...]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    canonical = {Path(os.path.abspath(path.expanduser())) for path in asset_paths}
    for raw in sorted(canonical, key=str):
        try:
            resolved = raw.resolve(strict=True)
        except OSError as exc:
            raise DeploymentError(f"deployment asset is missing or unreadable: {raw}") from exc
        if resolved != raw or not raw.is_file():
            raise DeploymentError(
                f"deployment asset must be a regular path without symlinks: {raw}")
        entries.append({"path": str(raw), "sha256": _sha256_bytes(raw.read_bytes())})
    return entries


def _environment(dependency_hash: str) -> dict[str, str]:
    values = {
        "dependency_digest": dependency_hash,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "abi_tag": sys.implementation.cache_tag or "unknown",
        "platform_tag": sysconfig.get_platform(),
    }
    values["environment_digest"] = _sha256_bytes(_canonical(values).encode())
    return values


def build_working_tree_manifest(
    *,
    identity: ArtifactIdentity,
    resolved_config: dict[str, Any],
    universe_name: str | None,
    repo_root: Path,
    asset_paths: tuple[Path, ...] = (),
) -> DeploymentManifest:
    """Capture the complete temporary working-tree deployment read set."""
    if identity.dependency_hash is None:
        raise DeploymentError("deployment requires a dependency digest")
    root = repo_root.resolve()
    source_ref = _assert_clean_working_tree(root)
    environment = _environment(identity.dependency_hash)
    assets = _asset_entries(asset_paths)
    payload: dict[str, Any] = {
        "identity": {
            "code_hash": identity.code_hash,
            "config_hash": identity.config_hash,
            "dependency_hash": identity.dependency_hash,
        },
        "resolved_config": resolved_config,
        "universe_name": universe_name,
        "environment": environment,
        "planner_protocol_version": PLANNER_PROTOCOL_VERSION,
        "source_kind": "working_tree",
        "source_ref": source_ref,
        "assets": assets,
    }
    manifest_json = _canonical(payload)
    return DeploymentManifest(
        manifest_digest=_sha256_bytes(manifest_json.encode()),
        manifest_json=manifest_json,
        code_hash=identity.code_hash,
        config_hash=identity.config_hash,
        dependency_hash=identity.dependency_hash,
        resolved_config_json=_canonical(resolved_config),
        universe_name=universe_name,
        environment_digest=environment["environment_digest"],
        python_implementation=environment["python_implementation"],
        python_version=environment["python_version"],
        abi_tag=environment["abi_tag"],
        platform_tag=environment["platform_tag"],
        planner_protocol_version=PLANNER_PROTOCOL_VERSION,
        source_kind="working_tree",
        source_ref=source_ref,
        asset_digests_json=_canonical(assets),
    )


def verify_working_tree_manifest(manifest: DeploymentManifest, *, repo_root: Path) -> None:
    """Fail closed if source, environment, assets, or the descriptor itself drifted."""
    try:
        payload = json.loads(manifest.manifest_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DeploymentError("deployment manifest is corrupt") from exc
    if not isinstance(payload, dict):
        raise DeploymentError("deployment manifest root is corrupt")
    if _sha256_bytes(manifest.manifest_json.encode()) != manifest.manifest_digest:
        raise DeploymentError("deployment manifest digest is corrupt")
    try:
        resolved_config = json.loads(manifest.resolved_config_json)
        recorded_assets = json.loads(manifest.asset_digests_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DeploymentError("deployment manifest descriptor fields are corrupt") from exc
    expected_payload = {
        "identity": {
            "code_hash": manifest.code_hash,
            "config_hash": manifest.config_hash,
            "dependency_hash": manifest.dependency_hash,
        },
        "resolved_config": resolved_config,
        "universe_name": manifest.universe_name,
        "environment": {
            "dependency_digest": manifest.dependency_hash,
            "python_implementation": manifest.python_implementation,
            "python_version": manifest.python_version,
            "abi_tag": manifest.abi_tag,
            "platform_tag": manifest.platform_tag,
            "environment_digest": manifest.environment_digest,
        },
        "planner_protocol_version": manifest.planner_protocol_version,
        "source_kind": manifest.source_kind,
        "source_ref": manifest.source_ref,
        "assets": recorded_assets,
    }
    if _canonical(payload) != _canonical(expected_payload):
        raise DeploymentError("deployment manifest descriptor fields disagree")
    source_ref = _assert_clean_working_tree(repo_root.resolve())
    environment = _environment(manifest.dependency_hash)
    try:
        assets = _asset_entries(tuple(Path(item["path"]) for item in recorded_assets))
    except (KeyError, TypeError) as exc:
        raise DeploymentError("deployment manifest asset entries are corrupt") from exc
    if (
        source_ref != manifest.source_ref
        or environment != expected_payload["environment"]
        or _canonical(assets) != manifest.asset_digests_json
        or manifest.planner_protocol_version != PLANNER_PROTOCOL_VERSION
        or manifest.source_kind != "working_tree"
    ):
        raise DeploymentError("deployment manifest drift detected")


def prepare_working_tree_deployment(
    conn,
    name: str,
    *,
    repo_root: Path | None = None,
) -> PreparedDeployment:
    """Resolve the exact qualifying gate and capture the current executable read set."""
    from algua.registry.approvals import compute_artifact_hashes
    from algua.strategies.loader import load_tradable_strategy

    row = conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()
    if row is None:
        raise DeploymentError(f"unknown strategy {name!r}")
    strategy_id = int(row["id"])
    identity = compute_artifact_hashes(name)
    gate = conn.execute(
        "SELECT id, actor, consumed, universe_name FROM gate_evaluations"
        " WHERE strategy_id=? AND passed=1 AND code_hash=? AND config_hash=?"
        " AND dependency_hash=? ORDER BY id DESC LIMIT 1",
        (strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash),
    ).fetchone()
    if gate is None:
        raise DeploymentError(
            "candidate has no qualifying research gate for the current artifact identity")
    eligible = (
        (gate["actor"] == "agent" and int(gate["consumed"]) == 1)
        or (gate["actor"] == "human" and int(gate["consumed"]) == 0)
    )
    if not eligible:
        raise DeploymentError("qualifying research gate has invalid actor/consumption state")
    if conn.execute(
        "SELECT 1 FROM strategy_deployments WHERE research_gate_id=?", (int(gate["id"]),)
    ).fetchone() is not None:
        raise DeploymentError("qualifying research gate already anchored a committed deployment")

    strategy = load_tradable_strategy(name)
    asset_paths: tuple[Path, ...] = ()
    if strategy.model_handle is not None:
        path = strategy.model_handle.version.artifact_path
        if path is None:
            raise DeploymentError("model-backed strategy has no resolved artifact path")
        asset_paths = (Path(path),)
    root = repo_root or Path(__file__).resolve().parents[2]
    manifest = build_working_tree_manifest(
        identity=identity,
        resolved_config=strategy.config.model_dump(mode="json"),
        universe_name=gate["universe_name"],
        repo_root=root,
        asset_paths=asset_paths,
    )
    return PreparedDeployment(manifest=manifest, research_gate_id=int(gate["id"]))
