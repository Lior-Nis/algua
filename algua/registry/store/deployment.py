"""SQLite operations for immutable deployment descriptors and epochs."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.deployment import DeploymentError, DeploymentManifest
from algua.registry.repository import StrategyRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class DeploymentRecord:
    id: int
    strategy_id: int
    artifact_id: int
    research_gate_id: int
    activated_at: str
    retired_at: str | None
    manifest_digest: str
    manifest_json: str
    code_hash: str
    config_hash: str
    dependency_hash: str
    universe_name: str | None
    resolved_config_json: str
    environment_digest: str
    python_implementation: str
    python_version: str
    abi_tag: str
    platform_tag: str
    planner_protocol_version: int
    source_kind: str
    source_ref: str
    asset_digests_json: str

    def manifest(self) -> DeploymentManifest:
        return DeploymentManifest(
            manifest_digest=self.manifest_digest, manifest_json=self.manifest_json,
            code_hash=self.code_hash, config_hash=self.config_hash,
            dependency_hash=self.dependency_hash,
            resolved_config_json=self.resolved_config_json, universe_name=self.universe_name,
            environment_digest=self.environment_digest,
            python_implementation=self.python_implementation,
            python_version=self.python_version, abi_tag=self.abi_tag,
            platform_tag=self.platform_tag,
            planner_protocol_version=self.planner_protocol_version,
            source_kind=self.source_kind, source_ref=self.source_ref,
            asset_digests_json=self.asset_digests_json,
        )


class DeploymentLedgerMixin:
    _conn: sqlite3.Connection

    if TYPE_CHECKING:
        def _apply_transition_locked(
            self, rec: StrategyRecord, to: Stage, actor: Actor, reason: str | None,
            code_hash: str | None, config_hash: str | None, dependency_hash: str | None,
            consume_gate_id: int | None, consume_forward_gate_id: int | None, now: str,
            *, revoke_allocation: bool = False, live_authorization=None,
        ) -> StrategyRecord: ...

    def resolve_deployment_artifact_locked(self, manifest: DeploymentManifest) -> int:
        """Insert a descriptor or verify the byte-identical row already at its digest."""
        row = self._conn.execute(
            "SELECT * FROM deployment_artifacts WHERE manifest_digest=?",
            (manifest.manifest_digest,),
        ).fetchone()
        if row is not None:
            immutable_fields = (
                "manifest_digest", "manifest_json", "code_hash", "config_hash",
                "dependency_hash", "resolved_config_json", "universe_name",
                "environment_digest", "python_implementation", "python_version", "abi_tag",
                "platform_tag", "planner_protocol_version", "source_kind", "source_ref",
                "asset_digests_json",
            )
            if any(row[field] != getattr(manifest, field) for field in immutable_fields):
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
            (
                manifest.manifest_digest, manifest.manifest_json, manifest.code_hash,
                manifest.config_hash, manifest.dependency_hash, manifest.resolved_config_json,
                manifest.universe_name, manifest.environment_digest,
                manifest.python_implementation, manifest.python_version, manifest.abi_tag,
                manifest.platform_tag, manifest.planner_protocol_version, manifest.source_kind,
                manifest.source_ref, manifest.asset_digests_json, _now(),
            ),
        )
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def active_deployment(self, strategy_id: int) -> DeploymentRecord | None:
        row = self._conn.execute(
            "SELECT d.id, d.strategy_id, d.artifact_id, d.research_gate_id, d.activated_at,"
            " d.retired_at, a.manifest_digest, a.manifest_json, a.code_hash, a.config_hash,"
            " a.dependency_hash, a.universe_name, a.resolved_config_json,"
            " a.environment_digest, a.python_implementation, a.python_version, a.abi_tag,"
            " a.platform_tag, a.planner_protocol_version, a.source_kind, a.source_ref,"
            " a.asset_digests_json"
            " FROM strategy_deployments d JOIN deployment_artifacts a ON a.id=d.artifact_id"
            " WHERE d.strategy_id=? AND d.retired_at IS NULL",
            (strategy_id,),
        ).fetchone()
        return DeploymentRecord(**dict(row)) if row is not None else None

    def require_tick_deployment(
        self, strategy_id: int, *, repo_root=None,
    ) -> DeploymentRecord | None:
        """Verify an active working-tree deployment, or admit only the fixed legacy cohort."""
        from pathlib import Path

        from algua.registry.deployment import verify_working_tree_manifest

        deployment = self.active_deployment(strategy_id)
        if deployment is not None:
            root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
            verify_working_tree_manifest(deployment.manifest(), repo_root=root)
            return deployment
        if self.is_legacy_deployment_strategy(strategy_id):
            return None
        raise DeploymentError(
            "strategy has no active deployment and is not in the fixed legacy cohort")

    def is_legacy_deployment_strategy(self, strategy_id: int) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM legacy_deployment_strategies WHERE strategy_id=?", (strategy_id,)
        ).fetchone() is not None

    def retire_active_deployment_locked(self, strategy_id: int) -> None:
        cur = self._conn.execute(
            "UPDATE strategy_deployments SET retired_at=?"
            " WHERE strategy_id=? AND retired_at IS NULL",
            (_now(), strategy_id),
        )
        if cur.rowcount != 1:
            raise DeploymentError("strategy has no active deployment to retire")

    def intake_candidate_to_paper(
        self,
        rec: StrategyRecord,
        capital: float,
        actor: Actor,
        account_equity: float,
        max_concurrent: int,
        *,
        deployment_manifest: DeploymentManifest,
        research_gate_id: int,
    ) -> StrategyRecord:
        """Atomically admit capital, open an epoch, and CAS candidate -> paper."""
        from algua.registry import allocations

        if self._conn.in_transaction:
            raise RuntimeError(
                "intake_candidate_to_paper must run at top level, not inside an open transaction")
        if rec.stage is not Stage.CANDIDATE:
            raise TransitionError(
                f"{rec.name!r} is not a candidate (stage {rec.stage.value!r})")
        now = _now()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            newest = self._conn.execute(
                "SELECT id, actor, consumed, universe_name, created_at FROM gate_evaluations"
                " WHERE strategy_id=? AND passed=1 AND code_hash=? AND config_hash=?"
                " AND dependency_hash=? ORDER BY id DESC LIMIT 1",
                (rec.id, deployment_manifest.code_hash, deployment_manifest.config_hash,
                 deployment_manifest.dependency_hash),
            ).fetchone()
            if newest is None or int(newest["id"]) != research_gate_id:
                raise DeploymentError(
                    "research gate is not the newest qualifying gate for this candidate+identity")
            candidate_entry = self._conn.execute(
                "SELECT code_hash, config_hash, dependency_hash, created_at"
                " FROM stage_transitions WHERE strategy_id=? AND to_stage='candidate'"
                " ORDER BY id DESC LIMIT 1",
                (rec.id,),
            ).fetchone()
            if candidate_entry is None or (
                candidate_entry["code_hash"] != deployment_manifest.code_hash
                or candidate_entry["config_hash"] != deployment_manifest.config_hash
                or candidate_entry["dependency_hash"] != deployment_manifest.dependency_hash
                or candidate_entry["created_at"] < newest["created_at"]
            ):
                raise DeploymentError(
                    "research gate did not justify the current candidate episode and identity")
            eligible = (
                (newest["actor"] == Actor.AGENT.value and int(newest["consumed"]) == 1)
                or (newest["actor"] == Actor.HUMAN.value and int(newest["consumed"]) == 0)
            )
            if not eligible:
                raise DeploymentError("research gate actor/consumption state is not deployable")
            if newest["universe_name"] != deployment_manifest.universe_name:
                raise DeploymentError("deployment universe binding does not match research gate")
            if self._conn.execute(
                "SELECT 1 FROM strategy_deployments WHERE research_gate_id=?",
                (research_gate_id,),
            ).fetchone() is not None:
                raise DeploymentError("research gate already anchored a committed deployment")
            count = allocations.active_paper_lane_count(self._conn)
            if count >= max_concurrent:
                raise allocations.CountCapReached(
                    f"paper book at capacity ({count}/{max_concurrent} active tenants)")
            artifact_id = self.resolve_deployment_artifact_locked(deployment_manifest)
            cur = self._conn.execute(
                "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
                " activated_at) VALUES (?,?,?,?)",
                (rec.id, artifact_id, research_gate_id, now),
            )
            if cur.lastrowid is None:
                raise DeploymentError("deployment activation produced no identity")
            allocations.allocate_locked(
                self._conn, rec.id, capital, actor.value, account_equity)
            result = self._apply_transition_locked(
                rec, Stage.PAPER, actor, "operator paper intake",
                None, None, None, None, None, now, revoke_allocation=False)
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return result
