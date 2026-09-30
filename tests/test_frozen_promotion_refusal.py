"""Story 1.3c §9 (CAP-5): a frozen deployment cannot be forward-qualified before Story 1.3d.

`paper promote` and the forward gate read the strategy's ACTIVE deployment FIRST. A frozen one is
refused with the stable, non-retryable `frozen_qualification_pending` before actor authentication,
checkout identity hashing, gate evaluation, token minting or any stage change. Working-tree and
legacy strategies keep today's path exactly.

The frozen fixture is admitted through the REAL `intake_candidate_to_paper` primitive with a
canonical frozen descriptor, so the refusal is exercised against a genuine `source_kind='frozen'`
ledger row rather than a hand-written one.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from algua.cli.errors import error_code, is_retryable
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry.artifact_errors import FrozenQualificationPending
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.db import connect, migrate
from algua.registry.forward_promotion import refuse_frozen_promotion, run_forward_gate
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import ForwardGateCriteria
from tests._deployment_helpers import force_legacy_strategy, frozen_manifest

runner = CliRunner()

NAME = "frozen_s"
IDENTITY = ArtifactIdentity("b" * 32, "c" * 32, "d" * 64)
CONFIG = {"name": NAME, "universe": ["AAPL"], "params": {"lookback": 20}}


def _registry_conn() -> sqlite3.Connection:
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _candidate_with_gate(repo: SqliteStrategyRepository, name: str = NAME) -> tuple:
    """A candidate whose newest passing research gate justifies ``IDENTITY`` (the shape intake
    requires before it opens a deployment epoch)."""
    rec = repo.add(name)
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash=IDENTITY.code_hash,
        config_hash=IDENTITY.config_hash, dependency_hash=IDENTITY.dependency_hash,
        data_source="test", snapshot_id="snap", period_start="2024-01-01",
        period_end="2024-12-31", holdout_frac=0.2, actor="human", decision_json="{}",
        universe_name="liquid-us",
    )
    created_at = repo._conn.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()[0]
    repo._conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    repo._conn.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", "human", "fixture", *IDENTITY, created_at),
    )
    repo._conn.commit()
    return repo.get(name), gate_id


def _admit_frozen(conn: sqlite3.Connection, name: str = NAME) -> int:
    """Admit ``name`` into paper through the real frozen intake; returns its strategy id."""
    repo = SqliteStrategyRepository(conn)
    rec, gate_id = _candidate_with_gate(repo, name)
    descriptor = frozen_deployment_manifest(frozen_manifest(
        code_hash=IDENTITY.code_hash, config_hash=IDENTITY.config_hash,
        dependency_hash=str(IDENTITY.dependency_hash), resolved_config=CONFIG,
        universe_name="liquid-us",
    ))
    out = repo.intake_candidate_to_paper(
        rec, capital=1000.0, actor=Actor.AGENT, account_equity=1000.0, max_concurrent=1,
        deployment_manifest=descriptor, research_gate_id=gate_id,
    )
    assert out.stage is Stage.PAPER
    deployment = repo.active_deployment(rec.id)
    assert deployment is not None and deployment.source_kind == "frozen"
    return rec.id


def _snapshot(conn: sqlite3.Connection, strategy_id: int) -> dict:
    """Every row a promotion attempt could leave behind."""
    def count(sql: str, *args) -> int:
        return int(conn.execute(sql, args).fetchone()[0])

    return {
        "stage": conn.execute(
            "SELECT stage FROM strategies WHERE id=?", (strategy_id,)).fetchone()[0],
        "transitions": count(
            "SELECT COUNT(*) FROM stage_transitions WHERE strategy_id=?", strategy_id),
        "forward_rows": count("SELECT COUNT(*) FROM forward_gate_evaluations"),
        "challenges": count("SELECT COUNT(*) FROM actor_challenges"),
        "promote_audits": count("SELECT COUNT(*) FROM audit_log WHERE action='paper_promote'"),
    }


class _Spies:
    """Recording tripwires on every step the refusal must precede. Each one records its name
    and raises, so a step that runs is both visible in ``calls`` and stops the command."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        targets = {
            # actor authentication (the CLI's bound name) and every checkout identity hash
            "authenticate_actor": "algua.cli.paper_cmd.authenticate_actor",
            "compute_artifact_hashes": "algua.registry.approvals.compute_artifact_hashes",
            "forward_identity": "algua.registry.forward_promotion.compute_artifact_hashes",
            "transition_identity": "algua.registry.transitions._compute_hashes",
            # preflight, broker construction and gate evaluation
            "preflight": "algua.cli.paper_cmd.forward_promotion_preflight",
            "broker": "algua.cli.paper_cmd._alpaca_broker_from_settings",
            "assemble_evidence": "algua.registry.forward_promotion.assemble_forward_evidence",
            "evaluate_gate": "algua.registry.forward_promotion.evaluate_forward_gate",
        }
        for label, target in targets.items():
            monkeypatch.setattr(target, self._tripwire(label))
        # token minting and the promoting stage change
        for method in ("record_forward_gate_evaluation", "record_forward_pass_and_promote"):
            monkeypatch.setattr(SqliteStrategyRepository, method, self._tripwire(method))

    def _tripwire(self, label: str):
        def _fn(*_args, **_kwargs):
            self.calls.append(label)
            raise AssertionError(f"{label} ran before the frozen refusal")
        return _fn


# ---------------------------------------------------------------------------
# The stable code
# ---------------------------------------------------------------------------

def test_frozen_qualification_pending_is_a_stable_non_retryable_code():
    exc = FrozenQualificationPending()
    assert error_code(exc) == "frozen_qualification_pending"
    assert is_retryable("frozen_qualification_pending") is False
    assert "frozen" in str(exc)


# ---------------------------------------------------------------------------
# `paper promote` (the CLI entry point)
# ---------------------------------------------------------------------------

def test_paper_promote_refuses_a_frozen_deployment_before_any_work(monkeypatch):
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _snapshot(conn, sid)
    spies = _Spies(monkeypatch)

    result = runner.invoke(app, ["paper", "promote", NAME])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "frozen_qualification_pending"
    assert payload["retryable"] is False
    assert spies.calls == []
    with closing(_registry_conn()) as conn:
        after = _snapshot(conn, sid)
    assert after == before
    assert after["stage"] == "paper" and after["forward_rows"] == 0


def test_paper_promote_human_refusal_precedes_the_challenge(monkeypatch):
    """A declared human is refused BEFORE authentication: no challenge is issued or persisted
    (issuing one would first hash the checkout)."""
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _snapshot(conn, sid)

    result = runner.invoke(app, ["paper", "promote", NAME, "--actor", "human"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "frozen_qualification_pending"
    assert payload.get("action") != "human_actor_challenge"
    with closing(_registry_conn()) as conn:
        after = _snapshot(conn, sid)
    assert after == before and after["challenges"] == 0


def test_paper_promote_legacy_strategy_path_is_unchanged(monkeypatch):
    """A legacy-cohort tenant (no deployment) still reaches the forward gate and is refused there
    exactly as before, never with the frozen code."""
    from tests.test_forward_promotion import FakeCalendar

    with closing(_registry_conn()) as conn:
        rec = SqliteStrategyRepository(conn).add("legacy_s")
        force_legacy_strategy(conn, rec.id)

    class _Broker:
        def account_activities_window(self, after, until):
            return []

    monkeypatch.setattr(
        "algua.registry.forward_promotion.compute_artifact_hashes", lambda name: IDENTITY)
    monkeypatch.setattr("algua.cli.paper_cmd.get_calendar", lambda: FakeCalendar())
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _Broker)

    result = runner.invoke(app, ["paper", "promote", "legacy_s"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "invalid_input"
    assert "requires one active deployment epoch" in payload["error"]


# ---------------------------------------------------------------------------
# The forward gate (defense in depth: direct callers are refused too)
# ---------------------------------------------------------------------------

def test_run_forward_gate_refuses_a_frozen_deployment_directly(monkeypatch):
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _snapshot(conn, sid)
        spies = _Spies(monkeypatch)
        with pytest.raises(FrozenQualificationPending):
            run_forward_gate(
                SqliteStrategyRepository(conn), conn, name=NAME, actor=Actor.AGENT,
                criteria=ForwardGateCriteria(), calendar=object(),  # type: ignore[arg-type]
                now=datetime.now(UTC), activities_fetch=lambda a, u: [])
        assert spies.calls == []
        assert _snapshot(conn, sid) == before


def test_refuse_frozen_promotion_passes_working_tree_and_legacy_strategies():
    with closing(_registry_conn()) as conn:
        repo = SqliteStrategyRepository(conn)
        legacy = repo.add("no_deployment")
        refuse_frozen_promotion(conn, legacy.id)  # no active deployment: not frozen

        rec = repo.add("working_tree_s")
        gate_id = repo.record_gate_evaluation(
            rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
            funnel_window_days=90, breadth_provenance="measured", pit_ok=True,
            pit_override=False, holdout_n_bars=63, min_holdout_observations=63,
            code_hash="c", config_hash="g", dependency_hash="d", data_source="test",
            snapshot_id="snap", period_start="2024-01-01", period_end="2024-12-31",
            holdout_frac=0.2, actor="human", decision_json="{}", universe_name="u")
        conn.execute(
            "INSERT INTO deployment_artifacts(manifest_digest, manifest_json, code_hash,"
            " config_hash, dependency_hash, resolved_config_json, universe_name,"
            " environment_digest, python_implementation, python_version, abi_tag, platform_tag,"
            " planner_protocol_version, source_kind, source_ref, asset_digests_json, created_at)"
            " VALUES ('wt','{}','c','g','d','{}','u','e','CPython','3.12','abi','platform',1,"
            " 'working_tree','ref','[]','t')")
        artifact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
            " activated_at) VALUES (?,?,?,'2025-01-01T00:00:00+00:00')",
            (rec.id, artifact_id, gate_id))
        conn.commit()
        refuse_frozen_promotion(conn, rec.id)  # working tree: today's path

        frozen_id = _admit_frozen(conn)
        with pytest.raises(FrozenQualificationPending):
            refuse_frozen_promotion(conn, frozen_id)
