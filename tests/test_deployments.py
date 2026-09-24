from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from dataclasses import replace

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.execution.order_state import latest_tick_snapshot, record_tick_snapshot
from algua.operator.schedule import operator_run_lock
from algua.registry.allocations import active_allocation
from algua.registry.db import connect, migrate
from algua.registry.deployment import (
    DeploymentError,
    build_working_tree_manifest,
    verify_working_tree_manifest,
)
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy
from tests._deployment_helpers import force_legacy_strategy

IDENTITY = ArtifactIdentity("code", "config", "dependency")
CONFIG = {"name": "s", "universe": ["AAPL"], "params": {"lookback": 20}}


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "algua").mkdir(parents=True)
    (repo / "algua" / "strategy.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (repo / "uv.lock").write_text("version = 1\n")
    (repo / ".python-version").write_text("3.12\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def test_working_tree_manifest_is_canonical_and_verifiable(clean_repo):
    manifest = build_working_tree_manifest(
        identity=IDENTITY,
        resolved_config=CONFIG,
        universe_name="liquid-us",
        repo_root=clean_repo,
    )
    payload = json.loads(manifest.manifest_json)

    assert manifest.source_ref == _git(clean_repo, "rev-parse", "HEAD")
    assert payload["identity"] == {
        "code_hash": "code", "config_hash": "config", "dependency_hash": "dependency"
    }
    assert payload["resolved_config"] == CONFIG
    assert payload["universe_name"] == "liquid-us"
    assert payload["planner_protocol_version"] == 1
    assert len(manifest.manifest_digest) == 64
    verify_working_tree_manifest(manifest, repo_root=clean_repo)


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked", "ignored"])
def test_working_tree_manifest_rejects_an_incomplete_read_set(clean_repo, dirty_kind):
    if dirty_kind == "tracked":
        (clean_repo / "algua" / "strategy.py").write_text("VALUE = 2\n")
    elif dirty_kind == "staged":
        (clean_repo / "pyproject.toml").write_text("[project]\nname='changed'\n")
        _git(clean_repo, "add", "pyproject.toml")
    else:
        path = clean_repo / "algua" / f"{dirty_kind}.py"
        path.write_text("VALUE = 3\n")
        if dirty_kind == "ignored":
            (clean_repo / ".gitignore").write_text("algua/ignored.py\n")
            _git(clean_repo, "add", ".gitignore")
            _git(clean_repo, "commit", "-qm", "ignore fixture")

    with pytest.raises(DeploymentError, match="working tree"):
        build_working_tree_manifest(
            identity=IDENTITY,
            resolved_config=CONFIG,
            universe_name="liquid-us",
            repo_root=clean_repo,
        )


def test_external_asset_digest_is_verified(clean_repo, tmp_path):
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"model-v1")
    manifest = build_working_tree_manifest(
        identity=IDENTITY,
        resolved_config=CONFIG,
        universe_name="liquid-us",
        repo_root=clean_repo,
        asset_paths=(asset,),
    )
    asset.write_bytes(b"model-v2")
    with pytest.raises(DeploymentError, match="manifest drift"):
        verify_working_tree_manifest(manifest, repo_root=clean_repo)


def test_external_asset_symlink_is_rejected(clean_repo, tmp_path):
    target = tmp_path / "model-v1.bin"
    target.write_bytes(b"model-v1")
    asset = tmp_path / "model.bin"
    asset.symlink_to(target)

    with pytest.raises(DeploymentError, match="without symlinks"):
        build_working_tree_manifest(
            identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
            repo_root=clean_repo, asset_paths=(asset,),
        )


def test_sourceless_bytecode_is_not_treated_as_generated(clean_repo):
    cache = clean_repo / "algua" / "__pycache__"
    cache.mkdir()
    (cache / "injected.cpython-312.pyc").write_bytes(b"not-derived")

    with pytest.raises(DeploymentError, match="untracked source/config"):
        build_working_tree_manifest(
            identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
            repo_root=clean_repo,
        )


def test_manifest_rejects_denormalized_descriptor_disagreement(clean_repo):
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )

    with pytest.raises(DeploymentError, match="descriptor fields disagree"):
        verify_working_tree_manifest(
            replace(manifest, code_hash="other"), repo_root=clean_repo)


def test_artifact_insert_is_idempotent_but_digest_disagreement_fails(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    manifest = build_working_tree_manifest(
        identity=IDENTITY,
        resolved_config=CONFIG,
        universe_name="liquid-us",
        repo_root=clean_repo,
    )

    with conn:
        first = repo.resolve_deployment_artifact_locked(manifest)
        second = repo.resolve_deployment_artifact_locked(manifest)
    assert first == second

    conn.execute("DROP TRIGGER trg_deployment_artifacts_no_update")
    conn.execute(
        "UPDATE deployment_artifacts SET manifest_json='{}' WHERE id=?", (first,)
    )
    conn.commit()
    with conn, pytest.raises(DeploymentError, match="digest collision|corrupt"):
        repo.resolve_deployment_artifact_locked(manifest)


def test_artifact_reuse_rejects_denormalized_field_corruption(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )
    with conn:
        artifact_id = repo.resolve_deployment_artifact_locked(manifest)
    conn.execute("DROP TRIGGER trg_deployment_artifacts_no_update")
    conn.execute(
        "UPDATE deployment_artifacts SET code_hash='other' WHERE id=?", (artifact_id,)
    )
    conn.commit()

    with conn, pytest.raises(DeploymentError, match="corrupt stored descriptor"):
        repo.resolve_deployment_artifact_locked(manifest)


def test_deployment_artifact_rows_are_immutable(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    manifest = build_working_tree_manifest(
        identity=IDENTITY,
        resolved_config=CONFIG,
        universe_name="liquid-us",
        repo_root=clean_repo,
    )
    with conn:
        repo.resolve_deployment_artifact_locked(manifest)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with conn:
            conn.execute("UPDATE deployment_artifacts SET source_ref='x'")


def _candidate_with_gate(repo, *, actor: str = "human", consumed: int = 0):
    rec = repo.add("s")
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1,
        windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
        pit_ok=True, pit_override=False, holdout_n_bars=63,
        min_holdout_observations=63, code_hash="code", config_hash="config",
        dependency_hash="dependency", data_source="test", snapshot_id="snap",
        period_start="2024-01-01", period_end="2024-12-31", holdout_frac=0.2,
        actor=actor, decision_json="{}", universe_name="liquid-us",
    )
    if consumed:
        repo._conn.execute(
            "UPDATE gate_evaluations SET consumed=? WHERE id=?", (consumed, gate_id)
        )
    gate_created_at = repo._conn.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,)
    ).fetchone()["created_at"]
    repo._conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    repo._conn.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", actor, "fixture", "code", "config",
         "dependency", gate_created_at),
    )
    repo._conn.commit()
    return repo.get("s"), gate_id


def test_candidate_intake_atomically_opens_deployment(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )

    out = repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
        max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
    )

    assert out.stage is Stage.PAPER
    assert active_allocation(conn, rec.id) is not None
    deployment = repo.active_deployment(rec.id)
    assert deployment is not None
    assert deployment.research_gate_id == gate_id
    assert deployment.manifest_digest == manifest.manifest_digest


def test_generic_candidate_to_paper_transition_cannot_bypass_intake(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("s")
    conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    conn.commit()

    with pytest.raises(TransitionError, match="deployment intake"):
        transition_strategy(repo, "s", Stage.PAPER, Actor.HUMAN)
    with pytest.raises(TransitionError, match="deployment intake"):
        repo.apply_transition(repo.get("s"), Stage.PAPER, Actor.HUMAN)


def test_intake_rejects_gate_not_bound_to_current_candidate_episode(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    conn.execute(
        "UPDATE stage_transitions SET code_hash='other' WHERE strategy_id=?"
        " AND to_stage='candidate'", (rec.id,),
    )
    conn.commit()
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )

    with pytest.raises(DeploymentError, match="current candidate episode"):
        repo.intake_candidate_to_paper(
            rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
            max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
        )


def test_committed_gate_cannot_anchor_a_second_epoch(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo, actor="agent", consumed=1)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )
    repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
        max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
    )
    deployment = repo.active_deployment(rec.id)
    assert deployment is not None
    with conn:
        repo.retire_active_deployment_locked(rec.id)
        conn.execute("DELETE FROM strategy_allocations WHERE strategy_id=?", (rec.id,))
        conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))

    with pytest.raises(DeploymentError, match="already anchored"):
        repo.intake_candidate_to_paper(
            repo.get("s"), capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
            max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
        )


def test_intake_rolls_back_artifact_epoch_and_allocation_on_stage_failure(
    clean_repo, tmp_path, monkeypatch,
):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )

    def fail_transition(*_args, **_kwargs):
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(repo, "_apply_transition_locked", fail_transition)
    with pytest.raises(RuntimeError, match="injected"):
        repo.intake_candidate_to_paper(
            rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
            max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
        )

    assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM strategy_deployments").fetchone()[0] == 0
    assert active_allocation(conn, rec.id) is None
    assert repo.get("s").stage is Stage.CANDIDATE


def _record_tick(conn, strategy_id: int, *, deployment_id: int | None):
    record_tick_snapshot(
        conn, "s", tick_ts="2026-09-24T20:00:00+00:00",
        decision_ts="2026-09-23T20:00:00+00:00", equity=1000.0,
        peak_equity=1000.0, positions={}, n_submitted=0, reconcile_ok=True,
        lane="paper", strategy_id=strategy_id, code_hash="code", config_hash="config",
        dependency_hash="dependency", account_id="paper-account", cash=1000.0,
        clock_source="broker", deployment_id=deployment_id,
    )


def test_tick_snapshot_is_guarded_by_active_deployment(clean_repo, tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )
    repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
        max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
    )
    deployment = repo.active_deployment(rec.id)
    assert deployment is not None

    _record_tick(conn, rec.id, deployment_id=deployment.id)
    assert latest_tick_snapshot(conn, "s")["deployment_id"] == deployment.id
    with conn:
        repo.retire_active_deployment_locked(rec.id)
    with pytest.raises(DeploymentError, match="active deployment"):
        _record_tick(conn, rec.id, deployment_id=deployment.id)


def test_null_deployment_tick_requires_fixed_legacy_cohort(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("s")
    with pytest.raises(DeploymentError, match="legacy cohort"):
        _record_tick(conn, rec.id, deployment_id=None)
    with pytest.raises(sqlite3.IntegrityError, match="fixed at migration"):
        conn.execute(
            "INSERT INTO legacy_deployment_strategies(strategy_id, original_stage, marked_at)"
            " VALUES (?,?,?)", (rec.id, "paper", "2026-09-24T00:00:00+00:00"),
        )
    force_legacy_strategy(conn, rec.id)
    _record_tick(conn, rec.id, deployment_id=None)
    assert latest_tick_snapshot(conn, "s")["deployment_id"] is None


def test_legacy_tick_must_match_strategy_name_and_lane(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("s")
    force_legacy_strategy(conn, rec.id)

    with pytest.raises(DeploymentError, match="legacy cohort"):
        record_tick_snapshot(
            conn, "other", tick_ts="2026-09-24T20:00:00+00:00", decision_ts=None,
            equity=1.0, peak_equity=1.0, positions={}, n_submitted=0, reconcile_ok=True,
            lane="paper", strategy_id=rec.id, code_hash="code", config_hash="config",
            dependency_hash="dependency", account_id="paper-account", cash=1.0,
            clock_source="broker", deployment_id=None,
        )
    with pytest.raises(DeploymentError, match="legacy cohort"):
        record_tick_snapshot(
            conn, "s", tick_ts="2026-09-24T20:00:00+00:00", decision_ts=None,
            equity=1.0, peak_equity=1.0, positions={}, n_submitted=0, reconcile_ok=True,
            lane="live", strategy_id=rec.id, code_hash="code", config_hash="config",
            dependency_hash="dependency", account_id="paper-account", cash=1.0,
            clock_source="broker", deployment_id=None,
        )


@pytest.mark.parametrize(
    ("target", "reason", "retired"),
    [
        (Stage.CANDIDATE, None, True),
        (Stage.DORMANT, "pause", False),
        (Stage.RETIRED, None, True),
    ],
)
def test_transition_retirement_matrix_is_atomic(
    clean_repo, tmp_path, target, reason, retired,
):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )
    repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
        max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
    )
    deployment = repo.active_deployment(rec.id)
    assert deployment is not None

    transition_strategy(repo, "s", target, Actor.HUMAN, reason=reason)

    stored = conn.execute(
        "SELECT retired_at FROM strategy_deployments WHERE id=?", (deployment.id,)
    ).fetchone()
    assert (stored["retired_at"] is not None) is retired


def test_programmatic_retirement_contends_on_operator_lock(
    clean_repo, tmp_path, monkeypatch,
):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo)
    manifest = build_working_tree_manifest(
        identity=IDENTITY, resolved_config=CONFIG, universe_name="liquid-us",
        repo_root=clean_repo,
    )
    repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0,
        max_concurrent=1, deployment_manifest=manifest, research_gate_id=gate_id,
    )
    lock_path = tmp_path / "operator.lock"
    monkeypatch.setattr("algua.operator.deployment_lock._operator_lock_path", lambda: lock_path)

    with operator_run_lock(lock_path, job="paper", host=socket.gethostname(), pid=os.getpid()):
        with pytest.raises(TransitionError, match="operator.lock is held"):
            transition_strategy(repo, "s", Stage.CANDIDATE, Actor.HUMAN)

    assert repo.get("s").stage is Stage.PAPER
    assert repo.active_deployment(rec.id) is not None
