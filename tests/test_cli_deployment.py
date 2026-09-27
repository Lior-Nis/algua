from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from algua.cli.errors import error_code, is_retryable
from algua.cli.main import app
from algua.registry.artifact_errors import (
    ArtifactNotFound,
    FrozenAssetsUnsupported,
    FrozenBundleCorrupt,
    FrozenDescriptorConflict,
    FrozenEnvironmentCorrupt,
    FrozenEnvironmentIncompatible,
    FrozenEnvironmentUnavailable,
    FrozenSourceDrift,
    FrozenSourceInvalid,
)
from tests.test_frozen_artifact_ledger import _frozen_manifest

runner = CliRunner()


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (FrozenSourceInvalid(), "frozen_source_invalid", False),
        (FrozenSourceDrift(), "frozen_source_drift", False),
        (FrozenAssetsUnsupported(), "frozen_assets_unsupported", False),
        (FrozenBundleCorrupt(), "frozen_bundle_corrupt", False),
        (FrozenEnvironmentUnavailable(), "frozen_environment_unavailable", True),
        (FrozenEnvironmentIncompatible(), "frozen_environment_incompatible", False),
        (FrozenEnvironmentCorrupt(), "frozen_environment_corrupt", False),
        (FrozenDescriptorConflict(), "frozen_descriptor_conflict", False),
        (ArtifactNotFound(), "artifact_not_found", False),
    ],
)
def test_frozen_error_taxonomy(exc, code, retryable) -> None:
    assert error_code(exc) == code
    assert is_retryable(code) is retryable


def test_prepare_command_emits_only_normative_fields(tmp_path, monkeypatch) -> None:
    manifest = _frozen_manifest()
    result = type("Result", (), {
        "strategy": "s", "artifact_id": 7, "research_gate_id": 3, "manifest": manifest,
    })()
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "algua.cli.deployment_cmd.prepare_frozen_artifact", lambda *_args, **_kwargs: result,
    )

    invocation = runner.invoke(app, ["deployment", "prepare", "s"])

    assert invocation.exit_code == 0, invocation.stdout
    assert json.loads(invocation.stdout) == {
        "ok": True,
        "strategy": "s",
        "artifact_id": 7,
        "manifest_digest": manifest.digest,
        "source_ref": manifest.source_ref,
        "bundle": {"digest": manifest.bundle.digest, "locator": manifest.bundle.locator},
        "environment": {
            "key": manifest.environment.key.digest,
            "digest": manifest.environment.digest,
            "inventory_digest": manifest.environment.inventory_digest,
            "locator": manifest.environment.locator,
        },
        "verified": True,
    }


def test_verify_command_is_offline_and_read_only(tmp_path, monkeypatch) -> None:
    manifest = _frozen_manifest()
    result = type("Result", (), {
        "strategy": "s", "artifact_id": 7, "manifest": manifest,
    })()
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    calls: list[tuple[str, str]] = []

    def verify(_repo, digest, *, store_root):
        calls.append((digest, str(store_root)))
        return result

    monkeypatch.setattr("algua.cli.deployment_cmd.verify_frozen_artifact", verify)
    invocation = runner.invoke(app, ["deployment", "verify", manifest.digest])

    assert invocation.exit_code == 0, invocation.stdout
    assert json.loads(invocation.stdout)["manifest_digest"] == manifest.digest
    assert calls == [(manifest.digest, str(tmp_path / "data"))]


def test_frozen_command_error_is_bounded_and_typed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setattr(
        "algua.cli.deployment_cmd.verify_frozen_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ArtifactNotFound()),
    )

    invocation = runner.invoke(app, ["deployment", "verify", "0" * 64])

    assert invocation.exit_code == 1
    assert json.loads(invocation.stdout) == {
        "ok": False,
        "error": "artifact descriptor was not found",
        "code": "artifact_not_found",
        "retryable": False,
    }


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (FrozenSourceInvalid(), "frozen_source_invalid", False),
        (FrozenSourceDrift(), "frozen_source_drift", False),
        (FrozenAssetsUnsupported(), "frozen_assets_unsupported", False),
        (FrozenBundleCorrupt(), "frozen_bundle_corrupt", False),
        (FrozenEnvironmentUnavailable(), "frozen_environment_unavailable", True),
        (FrozenEnvironmentIncompatible(), "frozen_environment_incompatible", False),
        (FrozenEnvironmentCorrupt(), "frozen_environment_corrupt", False),
        (FrozenDescriptorConflict(), "frozen_descriptor_conflict", False),
        (ArtifactNotFound(), "artifact_not_found", False),
    ],
)
def test_every_frozen_failure_has_a_bounded_json_envelope(
    tmp_path, monkeypatch, exc, code, retryable,
) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setattr(
        "algua.cli.deployment_cmd.verify_frozen_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    invocation = runner.invoke(app, ["deployment", "verify", "0" * 64])
    payload = json.loads(invocation.stdout)

    assert invocation.exit_code == 1
    assert payload["code"] == code
    assert payload["retryable"] is retryable
    assert 0 < len(payload["error"].encode()) <= 8 * 1024
    assert set(payload) == {"ok", "error", "code", "retryable"}
