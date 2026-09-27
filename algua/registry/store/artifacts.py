"""SQLite operations for immutable artifact descriptors before deployment activation."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from algua.contracts.lifecycle import Actor, Stage
from algua.registry.deployment import DeploymentError, DeploymentManifest


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ArtifactRecord:
    id: int
    descriptor: DeploymentManifest

    def manifest(self) -> DeploymentManifest:
        return self.descriptor

    def frozen_manifest(self):
        from algua.registry.artifact_recording import parse_frozen_deployment_manifest

        return parse_frozen_deployment_manifest(self.descriptor)


@dataclass(frozen=True)
class FrozenQualification:
    strategy_id: int
    research_gate_id: int
    universe_name: str | None


def _manifest_from_row(row: sqlite3.Row) -> DeploymentManifest:
    return DeploymentManifest(**{
        field: row[field] for field in DeploymentManifest.__dataclass_fields__
    })


class ArtifactLedgerMixin:
    _conn: sqlite3.Connection

    def resolve_deployment_artifact_locked(self, manifest: DeploymentManifest) -> int:
        """Insert a descriptor or verify the byte-identical row already at its digest."""
        row = self._conn.execute(
            "SELECT * FROM deployment_artifacts WHERE manifest_digest=?",
            (manifest.manifest_digest,),
        ).fetchone()
        if row is not None:
            fields = tuple(DeploymentManifest.__dataclass_fields__)
            if any(row[field] != getattr(manifest, field) for field in fields):
                raise DeploymentError(
                    "deployment artifact digest collision or corrupt stored descriptor")
            return int(row["id"])
        cur = self._conn.execute(
            "INSERT INTO deployment_artifacts("
            "manifest_digest, manifest_json, code_hash, config_hash, dependency_hash,"
            " resolved_config_json, universe_name, environment_digest, python_implementation,"
            " python_version, abi_tag, platform_tag, planner_protocol_version, source_kind,"
            " source_ref, asset_digests_json, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*[getattr(manifest, field) for field in DeploymentManifest.__dataclass_fields__],
             _now()),
        )
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def deployment_artifact_by_digest(self, manifest_digest: str) -> ArtifactRecord:
        row = self._conn.execute(
            "SELECT * FROM deployment_artifacts WHERE manifest_digest=?", (manifest_digest,),
        ).fetchone()
        if row is None:
            raise LookupError("deployment artifact not found")
        return ArtifactRecord(id=int(row["id"]), descriptor=_manifest_from_row(row))

    def qualify_frozen_candidate(
        self, name: str, *, code_hash: str, config_hash: str, dependency_hash: str,
    ) -> FrozenQualification:
        strategy = self._conn.execute(
            "SELECT id, stage FROM strategies WHERE name=?", (name,),
        ).fetchone()
        if strategy is None or strategy["stage"] != Stage.CANDIDATE.value:
            raise DeploymentError("strategy is not a current candidate")
        strategy_id = int(strategy["id"])
        newest = self._conn.execute(
            "SELECT id, actor, consumed, universe_name, created_at FROM gate_evaluations"
            " WHERE strategy_id=? AND passed=1 AND code_hash=? AND config_hash=?"
            " AND dependency_hash=? ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        if newest is None:
            raise DeploymentError(
                "candidate has no newest qualifying research gate for this identity")
        entry = self._conn.execute(
            "SELECT code_hash, config_hash, dependency_hash, created_at"
            " FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'"
            " ORDER BY id DESC LIMIT 1", (strategy_id,),
        ).fetchone()
        if entry is None or (
            entry["code_hash"] != code_hash or entry["config_hash"] != config_hash
            or entry["dependency_hash"] != dependency_hash
            or entry["created_at"] < newest["created_at"]
        ):
            raise DeploymentError(
                "research gate did not justify the current candidate episode and identity")
        eligible = (
            newest["actor"] == Actor.AGENT.value and int(newest["consumed"]) == 1
        ) or (
            newest["actor"] == Actor.HUMAN.value and int(newest["consumed"]) == 0
        )
        if not eligible:
            raise DeploymentError("research gate actor/consumption state is not deployable")
        gate_id = int(newest["id"])
        if self._conn.execute(
            "SELECT 1 FROM strategy_deployments WHERE research_gate_id=?", (gate_id,),
        ).fetchone() is not None:
            raise DeploymentError("research gate already anchored a committed deployment")
        return FrozenQualification(strategy_id, gate_id, newest["universe_name"])

    def record_frozen_artifact(
        self, name: str, manifest: DeploymentManifest, *, research_gate_id: int,
    ) -> int:
        from algua.registry.artifact_recording import parse_frozen_deployment_manifest

        parse_frozen_deployment_manifest(manifest)
        if self._conn.in_transaction:
            raise RuntimeError("record_frozen_artifact must run outside an open transaction")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            qualification = self.qualify_frozen_candidate(
                name, code_hash=manifest.code_hash, config_hash=manifest.config_hash,
                dependency_hash=manifest.dependency_hash,
            )
            if qualification.research_gate_id != research_gate_id:
                raise DeploymentError(
                    "research gate is not the newest qualifying gate for this candidate+identity")
            if qualification.universe_name != manifest.universe_name:
                raise DeploymentError("deployment universe binding does not match research gate")
            artifact_id = self.resolve_deployment_artifact_locked(manifest)
            self._conn.commit()
            return artifact_id
        except BaseException:
            self._conn.rollback()
            raise
