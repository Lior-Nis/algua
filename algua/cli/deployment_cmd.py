"""Thin JSON surface for non-activating frozen artifact preparation and verification."""
from __future__ import annotations

import functools
import sqlite3
from pathlib import Path

import typer

from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.registry.artifact_errors import (
    ArtifactNotFound,
    FrozenArtifactError,
    FrozenDescriptorConflict,
    FrozenEnvironmentUnavailable,
)
from algua.registry.artifact_preparation import prepare_frozen_artifact
from algua.registry.artifact_verification import verify_frozen_artifact
from algua.registry.db import registry_conn
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.store import SqliteStrategyRepository

deployment_app = typer.Typer(
    help="Prepare and verify immutable planner artifacts", no_args_is_help=True)


def _sanitize_unexpected(message: str):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                raise sqlite3.OperationalError("registry is unavailable") from exc
            except (
                FrozenArtifactError, FrozenDescriptorConflict, FrozenEnvironmentUnavailable,
                ArtifactNotFound,
            ):
                raise
            except Exception as exc:
                raise FrozenDescriptorConflict(message) from exc
        return wrapped
    return decorate


def _payload(strategy: str, artifact_id: int, manifest: FrozenManifest) -> dict:
    return {
        "ok": True,
        "strategy": strategy,
        "artifact_id": artifact_id,
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


@deployment_app.command("prepare")
@json_errors
@_sanitize_unexpected("frozen artifact preparation failed unexpectedly")
def prepare(name: str) -> None:
    settings = get_settings()
    with registry_conn() as conn:
        result = prepare_frozen_artifact(
            SqliteStrategyRepository(conn), name,
            repo_root=Path(__file__).resolve().parents[2], store_root=settings.data_dir,
        )
    emit(_payload(result.strategy, result.artifact_id, result.manifest))


@deployment_app.command("verify")
@json_errors
@_sanitize_unexpected("frozen artifact verification failed unexpectedly")
def verify(manifest_digest: str) -> None:
    settings = get_settings()
    with registry_conn() as conn:
        result = verify_frozen_artifact(
            SqliteStrategyRepository(conn), manifest_digest, store_root=settings.data_dir)
    emit(_payload(result.strategy, result.artifact_id, result.manifest))
