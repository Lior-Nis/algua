from __future__ import annotations

import json

import pytest

from algua.registry.artifact_contract import (
    ArtifactFile,
    BuildInputs,
    BundleDescriptor,
    EnvironmentDescriptor,
    EnvironmentKey,
    InstalledDistribution,
    InstalledInventory,
    InterpreterIdentity,
    InterpreterLink,
    canonical_json,
)
from algua.registry.artifact_manifest import parse_frozen_manifest
from algua.registry.frozen_manifest_contract import FrozenManifest


def _identity() -> InterpreterIdentity:
    return InterpreterIdentity(
        implementation="CPython",
        version="3.12.11",
        cache_tag="cpython-312",
        soabi="cpython-312-x86_64-linux-gnu",
        platform_tag="linux-x86_64",
        os_name="linux",
        machine="x86_64",
    )


def _manifest() -> FrozenManifest:
    inputs = BuildInputs(
        files=(
            ArtifactFile(".python-version", "100644", 5, "1" * 64),
            ArtifactFile("pyproject.toml", "100644", 11, "2" * 64),
            ArtifactFile("uv.lock", "100644", 13, "3" * 64),
        )
    )
    key = EnvironmentKey(
        build_inputs_digest=inputs.digest,
        dependency_hash="4" * 64,
        interpreter=_identity(),
        uv_version="uv 0.9.26",
        create_argv=("uv", "venv", "--relocatable"),
        sync_argv=("uv", "sync", "--locked", "--link-mode", "copy"),
    )
    inventory = InstalledInventory(
        distributions=(InstalledDistribution("numpy", "2.3.3"),),
        files=(ArtifactFile("lib/python3.12/site-packages/numpy/__init__.py", "100644", 7,
                            "5" * 64),),
        interpreter_links=(
            InterpreterLink("bin/python", "base-interpreter"),
            InterpreterLink("bin/python3", "bin/python"),
            InterpreterLink("bin/python3.12", "bin/python"),
        ),
    )
    environment = EnvironmentDescriptor(
        key=key,
        inventory_digest=inventory.digest,
        interpreter=_identity(),
    )
    bundle = BundleDescriptor.from_files(
        (
            ArtifactFile("_algua/protocol.json", "100644", 17, "6" * 64),
            ArtifactFile("_algua/resolved-config.json", "100644", 2, "7" * 64),
            ArtifactFile("algua/__init__.py", "100644", 0, "e3b0c44298fc1c149afbf4c8996fb924"
                         "27ae41e4649b934ca495991b7852b855"),
        )
    )
    return FrozenManifest(
        source_ref="8" * 40,
        code_hash="9" * 32,
        config_hash="a" * 32,
        dependency_hash="4" * 64,
        resolved_config={"alpha": 1, "label": "caf\u00e9"},
        universe_name="liquid-us",
        bundle=bundle,
        environment=environment,
    )


def test_literal_digest_vectors_and_round_trip() -> None:
    manifest = _manifest()

    assert manifest.bundle.digest == (
        "9efd8307dd08f1508d5bdfdd8b4eeb0235d0f1fc36b809da543ab9674b3e28b4")
    assert manifest.environment.key.build_inputs_digest == (
        "b553df5a7248af7fd8d157396ea48c0c04dc1064b06908a3498d96632143f80e")
    assert manifest.environment.key.digest == (
        "f591017e696ca18a917b3aaa62df509720bdbdc396c240439f924f301a67ad06")
    assert manifest.environment.inventory_digest == (
        "868609493a8ffd308a32abcf7a5b9a53449c5b958379940874b3fb5179098d7f")
    assert manifest.environment.digest == (
        "be7ba718e3da41af3aa7d5a70d4762e07d61a2da48cd14bf3202a0a1a3cb227e")
    assert manifest.digest == "410ed32a3c39e2d978ee6760d6b6f4d1532bbfe413241514573ead74ea26bc23"
    assert parse_frozen_manifest(manifest.json) == manifest


def test_canonical_json_normalizes_unicode_and_rejects_non_finite() -> None:
    assert canonical_json({"z": "cafe\u0301", "a": 1}) == '{"a":1,"z":"caf\u00e9"}'
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"bad": float("nan")})


def test_manifest_retains_a_canonicalized_config_and_enforces_size_bound() -> None:
    manifest = _manifest()
    changed = FrozenManifest(
        source_ref=manifest.source_ref,
        code_hash=manifest.code_hash,
        config_hash=manifest.config_hash,
        dependency_hash=manifest.dependency_hash,
        resolved_config={"label": "cafe\u0301"},
        universe_name=manifest.universe_name,
        bundle=manifest.bundle,
        environment=manifest.environment,
    )
    assert changed.resolved_config == {"label": "caf\u00e9"}
    with pytest.raises(ValueError, match="size"):
        parse_frozen_manifest(" " * (1024 * 1024 + 1))


def test_bundle_descriptor_enforces_generated_and_aggregate_size_bounds(monkeypatch) -> None:
    monkeypatch.setattr("algua.registry.artifact_contract.MAX_FILE_BYTES", 4)
    monkeypatch.setattr("algua.registry.artifact_contract.MAX_BUNDLE_BYTES", 6)
    with pytest.raises(ValueError, match="per-file"):
        BundleDescriptor.from_files((ArtifactFile("x", "100644", 5, "1" * 64),))
    with pytest.raises(ValueError, match="aggregate"):
        BundleDescriptor.from_files((
            ArtifactFile("x", "100644", 4, "1" * 64),
            ArtifactFile("y", "100644", 3, "2" * 64),
        ))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update(extra=True),
        lambda p: p.pop("source_ref"),
        lambda p: p.update(descriptor_version=2),
        lambda p: p.update(source_kind="working_tree"),
        lambda p: p.update(assets=[{"path": "/tmp/model"}]),
    ],
)
def test_parser_rejects_unknown_missing_or_unsupported_fields(mutator) -> None:
    payload = json.loads(_manifest().json)
    mutator(payload)
    with pytest.raises(ValueError):
        parse_frozen_manifest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def test_parser_rejects_duplicate_keys_and_noncanonical_bytes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_frozen_manifest('{"descriptor_version":1,"descriptor_version":1}')
    with pytest.raises(ValueError, match="canonical"):
        parse_frozen_manifest(json.dumps(json.loads(_manifest().json), indent=2))


def test_identity_layers_change_independently() -> None:
    original = _manifest()
    changed_key = EnvironmentKey(
        build_inputs_digest=original.environment.key.build_inputs_digest,
        dependency_hash=original.environment.key.dependency_hash,
        interpreter=original.environment.key.interpreter,
        uv_version="uv 0.9.27",
        create_argv=original.environment.key.create_argv,
        sync_argv=original.environment.key.sync_argv,
    )
    changed_environment = EnvironmentDescriptor(
        key=changed_key,
        inventory_digest=original.environment.inventory_digest,
        interpreter=original.environment.interpreter,
    )
    changed = FrozenManifest(
        source_ref=original.source_ref,
        code_hash=original.code_hash,
        config_hash=original.config_hash,
        dependency_hash=original.dependency_hash,
        resolved_config=original.resolved_config,
        universe_name=original.universe_name,
        bundle=original.bundle,
        environment=changed_environment,
    )

    assert changed.bundle.digest == original.bundle.digest
    assert changed.environment.digest != original.environment.digest
    assert changed.digest != original.digest
