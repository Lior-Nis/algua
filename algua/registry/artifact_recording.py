"""Typed bridge between frozen descriptors and the existing immutable deployment ledger."""
from __future__ import annotations

from algua.registry.artifact_contract import canonical_json
from algua.registry.artifact_manifest import parse_frozen_manifest
from algua.registry.deployment import DeploymentError, DeploymentManifest
from algua.registry.frozen_manifest_contract import FrozenManifest


def frozen_deployment_manifest(frozen: FrozenManifest) -> DeploymentManifest:
    """Project one canonical frozen descriptor onto the unchanged ledger schema."""
    interpreter = frozen.environment.interpreter
    return DeploymentManifest(
        manifest_digest=frozen.digest,
        manifest_json=frozen.json,
        code_hash=frozen.code_hash,
        config_hash=frozen.config_hash,
        dependency_hash=frozen.dependency_hash,
        resolved_config_json=canonical_json(frozen.resolved_config),
        universe_name=frozen.universe_name,
        environment_digest=frozen.environment.digest,
        python_implementation=interpreter.implementation,
        python_version=interpreter.version,
        abi_tag=interpreter.cache_tag,
        platform_tag=interpreter.platform_tag,
        planner_protocol_version=frozen.to_dict()["planner_protocol_version"],
        source_kind="frozen",
        source_ref=frozen.source_ref,
        asset_digests_json="[]",
    )


def parse_frozen_deployment_manifest(manifest: DeploymentManifest) -> FrozenManifest:
    """Parse and prove exact agreement between frozen JSON and every denormalized column."""
    if manifest.source_kind != "frozen":
        raise DeploymentError("deployment artifact is not a frozen descriptor")
    try:
        frozen = parse_frozen_manifest(manifest.manifest_json)
        expected = frozen_deployment_manifest(frozen)
    except (TypeError, ValueError) as exc:
        raise DeploymentError("frozen deployment descriptor is corrupt") from exc
    if manifest != expected:
        raise DeploymentError("frozen deployment descriptor fields disagree")
    return frozen
