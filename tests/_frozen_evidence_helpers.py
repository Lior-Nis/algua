"""Frozen invocation evidence fixtures (Story 1.3d).

A frozen tick can only be recorded carrying a link to a successful Phase B invocation of the same
deployment and snapshot (the v48 ``tick_snapshots_frozen_link`` trigger). Tests that stamp frozen
ticks record that evidence through the production recorder rather than bypassing the trigger.

``seed_deployment`` inserts a strategy, a research gate, an artifact of the requested
``source_kind`` and an active deployment with raw SQL, so it works on a v47 database as well as a
current one (the migration tests seed a v47 database before migrating it).
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from algua.contracts.frozen_evidence import FrozenAttempt
from algua.registry.store.frozen_evidence import record_frozen_invocation

IDENTITY = ("c" * 32, "g" * 32, "d" * 64)


def attempt(**overrides: Any) -> FrozenAttempt:
    """A successful Phase A attempt (``snapshot_required``); override any column."""
    fields: dict[str, Any] = {
        "deployment_id": 1, "request_id": "a" * 32, "phase": "a", "phase_a_invocation_id": None,
        "snapshot_id": "snap-1", "bars_start": "2026-06-01T00:00:00+00:00",
        "bars_end": "2026-09-29T00:00:00+00:00", "request_json": '{"request":"a"}',
        "request_sha256": "1" * 64, "bars_sha256": "2" * 64, "phase_a_binding": "binding-1",
        "result_kind": "snapshot_required", "result_sha256": "3" * 64, "failure_code": None,
        "returncode": 0, "signal": None, "timed_out": False, "stdout_exceeded": False,
        "stderr_truncated": False, "diagnostic": None,
        "started_at": "2026-09-30T20:00:00+00:00", "ended_at": "2026-09-30T20:00:01+00:00",
    }
    fields.update(overrides)
    return FrozenAttempt(**fields)


def failed_attempt(**overrides: Any) -> FrozenAttempt:
    """A failed attempt (a timeout by default); override any column."""
    fields: dict[str, Any] = {
        "result_kind": None, "result_sha256": None, "failure_code": "frozen_timeout",
        "returncode": None, "signal": 9, "timed_out": True, "diagnostic": "timed out",
    }
    fields.update(overrides)
    return attempt(**fields)


def record_final_invocation(
    conn: sqlite3.Connection, *, deployment_id: int, snapshot_id: str,
    result_kind: str = "decision", request_id: str | None = None,
) -> int:
    """Record a successful Phase A then Phase B attempt of one tick; return the Phase B id."""
    rid = request_id or uuid.uuid4().hex
    phase_a = record_frozen_invocation(conn, attempt(
        deployment_id=deployment_id, request_id=rid, snapshot_id=snapshot_id))
    return record_frozen_invocation(conn, attempt(
        deployment_id=deployment_id, request_id=rid, phase="b", phase_a_invocation_id=phase_a,
        snapshot_id=snapshot_id, result_kind=result_kind, request_json='{"request":"b"}'))


def seed_deployment(
    conn: sqlite3.Connection, name: str, *, source_kind: str, stage: str = "paper",
) -> tuple[int, int]:
    """Insert a strategy with one active deployment of ``source_kind``; return both ids."""
    strategy_id = conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,'t','t')",
        (name, stage),
    ).lastrowid
    gate_id = conn.execute(
        "INSERT INTO gate_evaluations(strategy_id, passed, n_funnel, own_lifetime_combos,"
        " windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok, pit_override,"
        " holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,"
        " data_source, snapshot_id, period_start, period_end, holdout_frac, actor, consumed,"
        " decision_json, universe_name, created_at) VALUES"
        " (?,1,1,1,1,90,'measured',1,0,63,63,?,?,?,'test','snap','2024-01-01',"
        " '2024-12-31',0.2,'human',0,'{}','u','t')",
        (strategy_id, *IDENTITY),
    ).lastrowid
    artifact_id = conn.execute(
        "INSERT INTO deployment_artifacts(manifest_digest, manifest_json, code_hash,"
        " config_hash, dependency_hash, resolved_config_json, universe_name,"
        " environment_digest, python_implementation, python_version, abi_tag, platform_tag,"
        " planner_protocol_version, source_kind, source_ref, asset_digests_json, created_at)"
        " VALUES (?,'{}',?,?,?,'{}','u','e','CPython','3.12','abi','platform',1,?,'ref',"
        " '[]','t')",
        (f"manifest-{name}", *IDENTITY, source_kind),
    ).lastrowid
    deployment_id = conn.execute(
        "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
        " activated_at) VALUES (?,?,?,'2026-09-01T00:00:00+00:00')",
        (strategy_id, artifact_id, gate_id),
    ).lastrowid
    conn.commit()
    assert strategy_id is not None and deployment_id is not None
    return int(strategy_id), int(deployment_id)
