"""Runtime resolution for the temporary working-tree deployment boundary."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

from algua.registry.deployment import DeploymentError
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import DeploymentRecord, SqliteStrategyRepository


def resolve_tick(
    conn: sqlite3.Connection, strategy_id: int, name: str,
    identity_loader: Callable[[str], ArtifactIdentity],
) -> tuple[DeploymentRecord | None, ArtifactIdentity]:
    """Verify the active descriptor and current identity before any tick effects.

    The working-tree/legacy resolver of the live lane (and of non-frozen paper tenants): a frozen
    row is refused with ``frozen_live_unsupported`` by ``require_tick_deployment`` before the
    checkout identity is computed (Story 1.3c §2)."""
    deployment = SqliteStrategyRepository(conn).require_tick_deployment(strategy_id)
    identity = identity_loader(name)
    if deployment is not None and (
        deployment.code_hash != identity.code_hash
        or deployment.config_hash != identity.config_hash
        or deployment.dependency_hash != identity.dependency_hash
    ):
        raise DeploymentError(f"{name} working-tree identity drifted from active deployment")
    return deployment, identity
