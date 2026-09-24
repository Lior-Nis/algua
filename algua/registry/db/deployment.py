"""Deployment identity and explicit evidence-epoch schema (v47)."""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS deployment_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_digest TEXT NOT NULL UNIQUE,
    manifest_json TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dependency_hash TEXT NOT NULL,
    resolved_config_json TEXT NOT NULL,
    universe_name TEXT,
    environment_digest TEXT NOT NULL,
    python_implementation TEXT NOT NULL,
    python_version TEXT NOT NULL,
    abi_tag TEXT NOT NULL,
    platform_tag TEXT NOT NULL,
    planner_protocol_version INTEGER NOT NULL,
    source_kind TEXT NOT NULL CHECK(source_kind IN ('working_tree', 'frozen')),
    source_ref TEXT NOT NULL,
    asset_digests_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_deployment_artifacts_no_update
BEFORE UPDATE ON deployment_artifacts BEGIN
    SELECT RAISE(ABORT, 'deployment_artifacts are immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_deployment_artifacts_no_delete
BEFORE DELETE ON deployment_artifacts BEGIN
    SELECT RAISE(ABORT, 'deployment_artifacts are immutable');
END;

CREATE TABLE IF NOT EXISTS strategy_deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    artifact_id INTEGER NOT NULL REFERENCES deployment_artifacts(id),
    research_gate_id INTEGER NOT NULL UNIQUE REFERENCES gate_evaluations(id),
    activated_at TEXT NOT NULL,
    retired_at TEXT,
    superseded_by INTEGER REFERENCES strategy_deployments(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_strategy_deployments_active
    ON strategy_deployments(strategy_id) WHERE retired_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_strategy_deployments_artifact
    ON strategy_deployments(artifact_id);
CREATE TRIGGER IF NOT EXISTS trg_strategy_deployments_restricted_update
BEFORE UPDATE ON strategy_deployments WHEN NOT (
    OLD.retired_at IS NULL AND NEW.retired_at IS NOT NULL
    AND NEW.id=OLD.id AND NEW.strategy_id=OLD.strategy_id
    AND NEW.artifact_id=OLD.artifact_id AND NEW.research_gate_id=OLD.research_gate_id
    AND NEW.activated_at=OLD.activated_at AND NEW.superseded_by IS OLD.superseded_by
) BEGIN
    SELECT RAISE(ABORT, 'strategy_deployments only permit one-way retirement');
END;
CREATE TRIGGER IF NOT EXISTS trg_strategy_deployments_no_delete
BEFORE DELETE ON strategy_deployments BEGIN
    SELECT RAISE(ABORT, 'strategy_deployments are append-preserving');
END;

-- Fixed compatibility cohort captured exactly once by migrate(). Absence from this table is not
-- evidence of legacy status, so a post-v47 admission cannot manufacture a NULL-deployment path.
CREATE TABLE IF NOT EXISTS legacy_deployment_strategies (
    strategy_id INTEGER PRIMARY KEY REFERENCES strategies(id),
    original_stage TEXT NOT NULL,
    marked_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_legacy_deployment_strategies_no_update
BEFORE UPDATE ON legacy_deployment_strategies BEGIN
    SELECT RAISE(ABORT, 'legacy deployment cohort is immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_legacy_deployment_strategies_no_delete
BEFORE DELETE ON legacy_deployment_strategies BEGIN
    SELECT RAISE(ABORT, 'legacy deployment cohort is immutable');
END;
CREATE TABLE IF NOT EXISTS deployment_migrations (
    id INTEGER PRIMARY KEY CHECK(id=1),
    legacy_cohort_captured_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trg_legacy_deployment_strategies_no_late_insert
BEFORE INSERT ON legacy_deployment_strategies
WHEN EXISTS (SELECT 1 FROM deployment_migrations WHERE id=1)
BEGIN
    SELECT RAISE(ABORT, 'legacy deployment cohort was fixed at migration');
END;
CREATE TRIGGER IF NOT EXISTS trg_deployment_migrations_no_update
BEFORE UPDATE ON deployment_migrations BEGIN
    SELECT RAISE(ABORT, 'deployment migration marker is immutable');
END;
CREATE TRIGGER IF NOT EXISTS trg_deployment_migrations_no_delete
BEFORE DELETE ON deployment_migrations BEGIN
    SELECT RAISE(ABORT, 'deployment migration marker is immutable');
END;
"""
