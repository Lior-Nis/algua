"""Story 1.3d §5 (CAP-4): the frozen promotion chokepoint and the refusals that remain.

`paper promote` reads the strategy's ACTIVE deployment in the slot where Story 1.3c refused a frozen
one. A frozen deployment's identity now comes from its recorded descriptor, verified there, before
actor authentication: content that does not verify (here, a synthetic descriptor whose bundle was
never published) refuses with `frozen_content_unavailable` before authentication, checkout identity
hashing, gate evaluation, token minting or any stage change. `frozen_qualification_pending` is gone.
Working-tree and legacy strategies keep today's order and identity sites exactly.

The raw `registry transition` edge to `forward_tested` still refuses a frozen deployment (as
`wrong_stage`: `paper promote` is the only way in), and go-live keeps `frozen_live_unsupported`.

The frozen fixture is admitted through the REAL `intake_candidate_to_paper` primitive with a
canonical frozen descriptor, so every refusal is exercised against a genuine `source_kind='frozen'`
ledger row rather than a hand-written one. Promotion SUCCESS from linked evidence, and the refusals
of corrupt, replaced and permission-drifted content, are in tests/test_frozen_promotion.py.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from algua.cli.errors import _registry
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry import artifact_errors
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.db import connect, migrate
from algua.registry.deployment import DeploymentError
from algua.registry.forward_promotion import promotion_identity, run_forward_gate
from algua.registry.frozen_tenant_errors import FrozenContentUnavailable
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import ForwardGateCriteria
from algua.strategies.base import config_hash
from algua.strategies.loader import _loaded_for_test
from tests._deployment_helpers import force_legacy_strategy, frozen_manifest
from tests.test_frozen_runtime import _config, _recorded

runner = CliRunner()

NAME = "frozen_s"
_STRATEGY_CONFIG = _config(NAME)
CONFIG = _recorded(_STRATEGY_CONFIG)  # decodable, so the refusal is the (unpublished) content
IDENTITY = ArtifactIdentity(
    "b" * 32, config_hash(_loaded_for_test(_STRATEGY_CONFIG)), "d" * 64)


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path):
    """The content store the chokepoint verifies against: empty, so the synthetic descriptor's
    bundle is never found (conftest isolates the registry, not the data dir)."""
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))


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


WT_IDENTITY = ArtifactIdentity("c", "g", "d")


def _working_tree_strategy(conn: sqlite3.Connection, name: str):
    """A paper strategy with one active working-tree deployment of ``WT_IDENTITY``."""
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(name)
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True,
        pit_override=False, holdout_n_bars=63, min_holdout_observations=63,
        code_hash=WT_IDENTITY.code_hash, config_hash=WT_IDENTITY.config_hash,
        dependency_hash=WT_IDENTITY.dependency_hash, data_source="test", snapshot_id="snap",
        period_start="2024-01-01", period_end="2024-12-31", holdout_frac=0.2, actor="human",
        decision_json="{}", universe_name="u")
    artifact_id = conn.execute(
        "INSERT INTO deployment_artifacts(manifest_digest, manifest_json, code_hash,"
        " config_hash, dependency_hash, resolved_config_json, universe_name,"
        " environment_digest, python_implementation, python_version, abi_tag, platform_tag,"
        " planner_protocol_version, source_kind, source_ref, asset_digests_json, created_at)"
        " VALUES (?,'{}',?,?,?,'{}','u','e','CPython','3.12','abi','platform',1,"
        " 'working_tree','ref','[]','t')", (f"wt-{name}", *WT_IDENTITY)).lastrowid
    conn.execute(
        "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
        " activated_at) VALUES (?,?,?,'2025-01-01T00:00:00+00:00')",
        (rec.id, artifact_id, gate_id))
    conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (rec.id,))
    conn.commit()
    return repo.get(name)


def _stop(label: str, calls: list[str] | None = None):
    def _fn(*_args, **_kwargs):
        if calls is not None:
            calls.append(label)
        raise RuntimeError(f"stop at {label}")
    return _fn


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
# The retired code
# ---------------------------------------------------------------------------

def test_frozen_qualification_pending_is_gone():
    assert not hasattr(artifact_errors, "FrozenQualificationPending")
    assert "frozen_qualification_pending" not in {code for _typ, code in _registry()}
    envelope_doc = Path(__file__).resolve().parents[1] / "docs/contracts/cli-error-envelope.md"
    assert "frozen_qualification_pending" not in envelope_doc.read_text()


# ---------------------------------------------------------------------------
# `paper promote` (the CLI entry point): the chokepoint runs first for a frozen deployment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("actor", ["agent", "human"])
def test_paper_promote_verifies_frozen_content_before_any_work(monkeypatch, actor):
    """The synthetic descriptor's bundle was never published: the fresh verification refuses in
    the chokepoint slot, before authentication (a human gets no challenge, which would bind an
    unverified identity), preflight, broker, evidence, evaluation, token or stage change."""
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _snapshot(conn, sid)
        deployment_id = SqliteStrategyRepository(conn).active_deployment(sid).id
    spies = _Spies(monkeypatch)

    result = runner.invoke(app, ["paper", "promote", NAME, "--actor", actor])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["code"] == "frozen_content_unavailable"
    assert payload["deployment_id"] == deployment_id
    assert payload["retryable"] is False
    assert payload.get("action") != "human_actor_challenge"
    assert spies.calls == []
    with closing(_registry_conn()) as conn:
        after = _snapshot(conn, sid)
    assert after == before
    assert after["stage"] == "paper" and after["forward_rows"] == 0 and after["challenges"] == 0


def test_paper_promote_working_tree_order_and_identity_site_are_unchanged(monkeypatch):
    """A working-tree strategy is hashed from the checkout exactly where it always was: after
    authentication, preflight and broker construction, immediately before the gate. The frozen
    chokepoint never runs for it."""
    order: list[str] = []
    with closing(_registry_conn()) as conn:
        rec = _working_tree_strategy(conn, "wt_order")

    import algua.cli.paper_cmd as paper_cmd
    import algua.registry.forward_promotion as forward_promotion

    real_auth = paper_cmd.authenticate_actor
    real_preflight = paper_cmd.forward_promotion_preflight

    def auth(*args, **kwargs):
        order.append("authenticate")
        return real_auth(*args, **kwargs)

    def preflight(*args, **kwargs):
        order.append("preflight")
        return real_preflight(*args, **kwargs)

    class _Broker:
        def account_activities_window(self, after, until):
            return []

    def broker():
        order.append("broker")
        return _Broker()

    def hashes(name):
        order.append("checkout_identity")
        return WT_IDENTITY

    def verifier(*_args, **_kwargs):
        raise AssertionError("a working-tree promotion never touches frozen content")

    monkeypatch.setattr(paper_cmd, "authenticate_actor", auth)
    monkeypatch.setattr(paper_cmd, "forward_promotion_preflight", preflight)
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", broker)
    monkeypatch.setattr(forward_promotion, "compute_artifact_hashes", hashes)
    monkeypatch.setattr(forward_promotion, "FrozenContentVerifier", verifier)
    monkeypatch.setattr(
        "algua.registry.deployment.verify_working_tree_manifest", lambda manifest, repo_root: None)
    monkeypatch.setattr(forward_promotion, "assemble_forward_evidence", _stop("assemble", order))

    result = runner.invoke(app, ["paper", "promote", rec.name])

    assert result.exit_code == 1, result.stdout
    assert order == ["authenticate", "preflight", "broker", "checkout_identity", "assemble"]


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
# The chokepoint and the forward gate (direct callers)
# ---------------------------------------------------------------------------

def test_promotion_identity_keeps_the_working_tree_and_legacy_paths(monkeypatch):
    hashed: list[str] = []
    monkeypatch.setattr(
        "algua.registry.forward_promotion.compute_artifact_hashes",
        lambda name: hashed.append(name) or WT_IDENTITY)
    monkeypatch.setattr(
        "algua.registry.deployment.verify_working_tree_manifest", lambda manifest, repo_root: None)
    with closing(_registry_conn()) as conn:
        repo = SqliteStrategyRepository(conn)
        rec = _working_tree_strategy(conn, "working_tree_s")
        deployment, identity = promotion_identity(conn, rec, data_dir=Path("/nonexistent"))
        assert identity == WT_IDENTITY and deployment == repo.active_deployment(rec.id)

        legacy = repo.add("no_deployment")
        force_legacy_strategy(conn, legacy.id)
        with pytest.raises(DeploymentError, match="requires one active deployment epoch"):
            promotion_identity(conn, repo.get("no_deployment"), data_dir=Path("/nonexistent"))

        drifted = _working_tree_strategy(conn, "drifted_s")
        monkeypatch.setattr(
            "algua.registry.forward_promotion.compute_artifact_hashes",
            lambda name: WT_IDENTITY._replace(code_hash="drift"))
        with pytest.raises(DeploymentError, match="does not match the working tree"):
            promotion_identity(conn, drifted, data_dir=Path("/nonexistent"))
    assert hashed == ["working_tree_s", "no_deployment"]


def test_promotion_identity_refuses_unverified_frozen_content_without_the_checkout(monkeypatch):
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        spies = _Spies(monkeypatch)
        with pytest.raises(FrozenContentUnavailable) as info:
            promotion_identity(
                conn, SqliteStrategyRepository(conn).get(NAME), data_dir=get_settings().data_dir)
        assert info.value.deployment_id == SqliteStrategyRepository(conn).active_deployment(sid).id
        assert spies.calls == []


def test_run_forward_gate_rechecks_the_identity_against_the_deployment(monkeypatch):
    """Defense in depth for a direct caller: a frozen deployment paired with any identity but its
    descriptor's (for example a checkout hash) is refused before evidence, evaluation or rows."""
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _snapshot(conn, sid)
        repo = SqliteStrategyRepository(conn)
        deployment = repo.active_deployment(sid)
        spies = _Spies(monkeypatch)
        with pytest.raises(DeploymentError, match="does not match"):
            run_forward_gate(
                repo, conn, name=NAME, actor=Actor.AGENT, criteria=ForwardGateCriteria(),
                calendar=object(),  # type: ignore[arg-type]
                now=datetime.now(UTC), activities_fetch=lambda a, u: [],
                deployment=deployment, identity=IDENTITY._replace(code_hash="checkout"))
        assert spies.calls == []
        assert _snapshot(conn, sid) == before


# ---------------------------------------------------------------------------
# `registry transition`: the raw forward edge and the go-live ceremony. Both would otherwise hash
# the CHECKOUT (importing its strategy module) and pin that code_hash for a frozen tenant.
# ---------------------------------------------------------------------------

_CHECKOUT_TARGETS = {
    "cli_identity": "algua.cli.registry_cmd.compute_artifact_hashes",
    "transition_identity": "algua.registry.transitions._compute_hashes",
    "approvals_identity": "algua.registry.approvals.compute_artifact_hashes",
    "checkout_import": "algua.registry.approvals.load_strategy",
    "checkout_config": "algua.strategies.loader.load_strategy_config",
    "forward_token": "algua.registry.transitions._validate_forward_gate",
    "certificate": "algua.registry.transitions._default_forward_certificate_verifier",
    "issue_challenge": "algua.registry.live_gate.issue_challenge",
    "verify_signature": "algua.registry.live_gate.verify_pending",
}


def _checkout_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Tripwires on every checkout hash/import, the forward token, the go-live certificate and
    challenge steps, and the stage write. Returns the list each tripped step records into."""
    spies = _Spies.__new__(_Spies)
    spies.calls = []
    for label, target in _CHECKOUT_TARGETS.items():
        monkeypatch.setattr(target, spies._tripwire(label))
    monkeypatch.setattr(
        SqliteStrategyRepository, "apply_transition", spies._tripwire("apply_transition"))
    return spies.calls


def _live_snapshot(conn: sqlite3.Connection, strategy_id: int) -> dict:
    return {
        **_snapshot(conn, strategy_id),
        "live_challenges": conn.execute("SELECT COUNT(*) FROM live_challenges").fetchone()[0],
        "live_authorizations": conn.execute(
            "SELECT COUNT(*) FROM live_authorizations").fetchone()[0],
    }


def _admit_frozen_at(stage: str) -> int:
    """A frozen tenant moved to ``stage`` by hand: the shape a pre-fix raw transition left."""
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, sid))
        conn.commit()
    return sid


@pytest.mark.parametrize("actor", ["human", "agent"])
def test_registry_transition_to_forward_tested_refuses_a_frozen_deployment_first(
    monkeypatch, actor,
):
    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _live_snapshot(conn, sid)
    calls = _checkout_spies(monkeypatch)

    result = runner.invoke(app, [
        "registry", "transition", NAME, "--to", "forward_tested", "--actor", actor,
        "--reason", "raw forward edge"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "wrong_stage"
    assert "reach forward_tested only through paper promote" in payload["error"]
    assert payload["retryable"] is False
    assert calls == []
    with closing(_registry_conn()) as conn:
        assert _live_snapshot(conn, sid) == before
    assert before["stage"] == "paper"


def test_transition_strategy_refuses_the_frozen_forward_edge_directly(monkeypatch):
    from algua.registry.transitions import transition_strategy

    with closing(_registry_conn()) as conn:
        sid = _admit_frozen(conn)
        before = _live_snapshot(conn, sid)
        calls = _checkout_spies(monkeypatch)
        with pytest.raises(TransitionError, match="forward_tested only through paper promote"):
            transition_strategy(
                SqliteStrategyRepository(conn), NAME, Stage.FORWARD_TESTED, Actor.HUMAN, "raw")
        assert calls == []
        assert _live_snapshot(conn, sid) == before


def test_go_live_challenge_refuses_a_frozen_deployment_before_hashing(monkeypatch):
    sid = _admit_frozen_at("forward_tested")
    with closing(_registry_conn()) as conn:
        before = _live_snapshot(conn, sid)
        deployment_id = SqliteStrategyRepository(conn).active_deployment(sid).id
    calls = _checkout_spies(monkeypatch)

    result = runner.invoke(
        app, ["registry", "transition", NAME, "--to", "live", "--actor", "human"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "frozen_live_unsupported"
    assert payload.get("action") != "go_live_challenge"
    assert payload["retryable"] is False
    assert calls == []
    with closing(_registry_conn()) as conn:
        after = _live_snapshot(conn, sid)
    assert after == before and after["live_challenges"] == 0
    assert after["stage"] == "forward_tested"
    assert deployment_id is not None


def test_go_live_signature_completion_refuses_a_frozen_deployment_before_hashing(
    monkeypatch, tmp_path,
):
    sid = _admit_frozen_at("forward_tested")
    with closing(_registry_conn()) as conn:
        before = _live_snapshot(conn, sid)
    signature = tmp_path / "challenge.sig"
    signature.write_bytes(b"not a signature")
    calls = _checkout_spies(monkeypatch)

    result = runner.invoke(app, [
        "registry", "transition", NAME, "--to", "live", "--actor", "human",
        "--signature", str(signature)])

    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["code"] == "frozen_live_unsupported"
    assert calls == []
    with closing(_registry_conn()) as conn:
        after = _live_snapshot(conn, sid)
    assert after == before and after["live_authorizations"] == 0


def test_transition_strategy_refuses_a_frozen_go_live_before_any_verifier(monkeypatch):
    from algua.registry.frozen_tenant_errors import FrozenLiveUnsupported
    from algua.registry.transitions import transition_strategy

    sid = _admit_frozen_at("forward_tested")
    consulted: list[str] = []

    def verifier(*_args, **_kwargs):
        consulted.append("verifier")
        return True

    with closing(_registry_conn()) as conn:
        before = _live_snapshot(conn, sid)
        calls = _checkout_spies(monkeypatch)
        with pytest.raises(FrozenLiveUnsupported) as info:
            transition_strategy(
                SqliteStrategyRepository(conn), NAME, Stage.LIVE, Actor.HUMAN, "go live",
                approval_verifier=verifier, forward_certificate_verifier=verifier)
        assert info.value.deployment_id == SqliteStrategyRepository(conn).active_deployment(
            sid).id
        assert calls == [] and consulted == []
        assert _live_snapshot(conn, sid) == before


def test_go_live_actor_wall_still_fires_first_for_a_frozen_deployment(monkeypatch):
    """The human-actor wall keeps its place ahead of the frozen refusal on both go-live paths."""
    _admit_frozen_at("forward_tested")
    calls = _checkout_spies(monkeypatch)

    result = runner.invoke(
        app, ["registry", "transition", NAME, "--to", "live", "--actor", "agent"])

    assert result.exit_code == 1, result.stdout
    assert "requires a human actor" in json.loads(result.stdout)["error"]
    assert calls == []


def _legacy_at(name: str, stage: str) -> int:
    with closing(_registry_conn()) as conn:
        rec = SqliteStrategyRepository(conn).add(name)
        force_legacy_strategy(conn, rec.id, stage=stage)
    return rec.id


def test_legacy_raw_forward_transition_still_pins_the_checkout_identity(monkeypatch):
    """Unchanged for a non-frozen strategy: the raw human edge hashes and pins identity."""
    sid = _legacy_at("legacy_fwd", "paper")
    hashed: list[str] = []
    monkeypatch.setattr(
        "algua.registry.transitions._compute_hashes",
        lambda name: hashed.append(name) or IDENTITY)

    result = runner.invoke(app, [
        "registry", "transition", "legacy_fwd", "--to", "forward_tested", "--actor", "human",
        "--reason", "raw"])

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["stage"] == "forward_tested"
    assert hashed == ["legacy_fwd"]
    with closing(_registry_conn()) as conn:
        row = conn.execute(
            "SELECT code_hash, config_hash, dependency_hash FROM stage_transitions"
            " WHERE strategy_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
    assert tuple(row) == tuple(IDENTITY)


def test_legacy_go_live_challenge_still_hashes_then_checks_the_certificate(monkeypatch):
    from algua.contracts.lifecycle import TransitionError

    _legacy_at("legacy_live", "forward_tested")
    seen: list[tuple] = []
    monkeypatch.setattr(
        "algua.cli.registry_cmd.compute_artifact_hashes",
        lambda name: seen.append(("hash", name)) or IDENTITY)

    def no_certificate(_repo, name, _sid, identity):
        seen.append(("certificate", name, identity))
        raise TransitionError("no forward certificate")

    monkeypatch.setattr(
        "algua.registry.transitions._default_forward_certificate_verifier",
        lambda: no_certificate)

    result = runner.invoke(
        app, ["registry", "transition", "legacy_live", "--to", "live", "--actor", "human"])

    assert result.exit_code == 1, result.stdout
    assert "no forward certificate" in json.loads(result.stdout)["error"]
    assert seen == [("hash", "legacy_live"), ("certificate", "legacy_live", IDENTITY)]
