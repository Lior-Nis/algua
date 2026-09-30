"""Deployment fixture helpers: the immutable v47 migration cohort and synthetic frozen descriptors.

Production has no post-migration enrollment path. Tests that exercise compatibility behavior must
therefore make the privilege escalation explicit by temporarily dropping and restoring the schema
trigger, rather than teaching runtime code a back door.

``frozen_manifest`` builds a canonical Story 1.3b descriptor without Git, uv or published content,
so admission tests can record and bind a real ``source_kind="frozen"`` ledger row.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from algua.registry.artifact_contract import ArtifactFile, BundleDescriptor
from algua.registry.environment_contract import (
    EnvironmentDescriptor,
    EnvironmentKey,
    InterpreterIdentity,
)
from algua.registry.frozen_manifest_contract import FrozenManifest

_LATE_INSERT_TRIGGER = """
CREATE TRIGGER trg_legacy_deployment_strategies_no_late_insert
BEFORE INSERT ON legacy_deployment_strategies
WHEN EXISTS (SELECT 1 FROM deployment_migrations WHERE id=1)
BEGIN
    SELECT RAISE(ABORT, 'legacy deployment cohort was fixed at migration');
END
"""


def force_legacy_strategy(
    conn: sqlite3.Connection, strategy_id: int, *, stage: str = "paper",
) -> None:
    """Fabricate a migration-time tenant for compatibility-only tests."""
    conn.execute("DROP TRIGGER IF EXISTS trg_legacy_deployment_strategies_no_late_insert")
    try:
        conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, strategy_id))
        conn.execute(
            "INSERT OR IGNORE INTO legacy_deployment_strategies"
            "(strategy_id, original_stage, marked_at) VALUES (?,?,?)",
            (strategy_id, stage, "test-fixture"),
        )
    finally:
        conn.execute(_LATE_INSERT_TRIGGER)
    conn.commit()


def frozen_manifest(
    *, code_hash: str, config_hash: str, dependency_hash: str,
    resolved_config: dict[str, Any], universe_name: str | None,
) -> FrozenManifest:
    """A canonical frozen descriptor for ``identity`` over synthetic bundle/environment digests."""
    interpreter = InterpreterIdentity(
        implementation="CPython", version="3.12.11", cache_tag="cpython-312",
        soabi="cpython-312-x86_64-linux-gnu", platform_tag="linux-x86_64",
        os_name="linux", machine="x86_64",
    )
    key = EnvironmentKey(
        build_inputs_digest="1" * 64, dependency_hash=dependency_hash,
        interpreter=interpreter, uv_version="uv 0.9.26",
        create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    bundle = BundleDescriptor.from_files((
        ArtifactFile("algua/__init__.py", "100644", 0,
                     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ))
    return FrozenManifest(
        source_ref="a" * 40, code_hash=code_hash, config_hash=config_hash,
        dependency_hash=dependency_hash, resolved_config=resolved_config,
        universe_name=universe_name, bundle=bundle,
        environment=EnvironmentDescriptor(key, "2" * 64, interpreter),
    )
