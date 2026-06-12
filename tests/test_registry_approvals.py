import inspect

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.approvals import (
    compute_artifact_hashes,
    has_valid_approval,
    record_approval,
)
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

STRATEGY = "cross_sectional_momentum"  # a real, loadable strategy module


@pytest.fixture()
def repo(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return SqliteStrategyRepository(c)


def _passing_certificate(repo, name, strategy_id):
    """Injected stand-in for the live wall's forward-certificate check (#124), so these tests
    keep exercising their named invariant — the APPROVAL/actor walls behind it."""
    return {"id": 1, "created_at": "2026-06-10T00:00:00+00:00", "realized_sharpe": 1.0,
            "holdout_sharpe": 1.2, "n_forward_observations": 80, "n_concurrent_forward": 1}


def _to_live(repo, name, actor, to=Stage.LIVE):
    return transition_strategy(repo, name, to, actor,
                               forward_certificate_verifier=_passing_certificate)


def _advance_to_paper(repo, name):
    repo.add(name)
    transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT)
    # CANDIDATE via human: scaffolding to a later stage, not exercising the agent shortlist gate.
    transition_strategy(repo, name, Stage.CANDIDATE, Actor.HUMAN)
    transition_strategy(repo, name, Stage.PAPER, Actor.AGENT)


def _advance_to_forward_tested(repo, name):
    _advance_to_paper(repo, name)
    transition_strategy(repo, name, Stage.FORWARD_TESTED, Actor.HUMAN, "test setup")


def test_live_requires_approval(repo):
    _advance_to_forward_tested(repo, STRATEGY)
    with pytest.raises(TransitionError):
        _to_live(repo, STRATEGY, Actor.HUMAN)


def test_live_requires_human_actor(repo):
    _advance_to_forward_tested(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    with pytest.raises(TransitionError):
        _to_live(repo, STRATEGY, Actor.AGENT)


def test_live_succeeds_with_human_and_recorded_approval(repo):
    _advance_to_forward_tested(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = _to_live(repo, STRATEGY, Actor.HUMAN)
    assert rec.stage is Stage.LIVE


def test_string_live_engages_gate(repo):
    # Passing the raw string "live" (not Stage.LIVE) must still engage the gate.
    _advance_to_forward_tested(repo, STRATEGY)
    with pytest.raises(TransitionError):
        _to_live(repo, STRATEGY, Actor.HUMAN, to="live")


def test_string_live_succeeds_with_approval(repo):
    _advance_to_forward_tested(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = _to_live(repo, STRATEGY, Actor.HUMAN, to="live")
    assert rec.stage is Stage.LIVE


def test_approval_binds_to_real_source_not_caller_strings(repo):
    # #79: a constant hash supplied at approve time cannot satisfy the gate, because the
    # approval stores the *recomputed* source hash and the gate recomputes it again.
    _advance_to_forward_tested(repo, STRATEGY)
    rec = repo.get(STRATEGY)
    # An attacker manually inserts a constant-hash approval row, mimicking "approve --code-hash X".
    repo.record_approval(rec.id, "constant", "constant", "constant", "attacker")
    with pytest.raises(TransitionError):
        _to_live(repo, STRATEGY, Actor.HUMAN)


def test_recorded_approval_matches_computed_hashes(repo):
    repo.add(STRATEGY)
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = repo.get(STRATEGY)
    assert has_valid_approval(repo, rec.id, code_hash, config_hash, dependency_hash) is True


def test_compute_artifact_hashes_is_deterministic_and_distinct(repo):
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)
    again = compute_artifact_hashes(STRATEGY)
    assert (code_hash, config_hash, dependency_hash) == again
    assert code_hash != config_hash  # source digest is not the config digest


def test_has_valid_approval(repo):
    repo.add(STRATEGY)
    s = repo.get(STRATEGY)
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)
    assert has_valid_approval(repo, s.id, code_hash, config_hash, dependency_hash) is False
    record_approval(repo, STRATEGY, "lior")
    assert has_valid_approval(repo, s.id, code_hash, config_hash, dependency_hash) is True


def test_record_approval_rejects_blank_approver(repo):
    repo.add(STRATEGY)
    with pytest.raises(ValueError):
        record_approval(repo, STRATEGY, " ")


def test_code_hash_covers_imported_algua_helper(repo, monkeypatch):
    # #97: the strategy imports a first-party helper (algua.strategies.base — StrategyConfig).
    # If that helper's source changes after approval, the recomputed code_hash MUST change,
    # so the stale approval can no longer promote altered behavior to live.
    import algua.strategies.base as helper

    baseline, _, _ = compute_artifact_hashes(STRATEGY)

    real_getsource = inspect.getsource

    def fake_getsource(obj):
        if obj is helper:
            return real_getsource(obj) + "\n# behavior-changing edit to a first-party helper\n"
        return real_getsource(obj)

    monkeypatch.setattr(inspect, "getsource", fake_getsource)
    mutated, _, _ = compute_artifact_hashes(STRATEGY)

    assert mutated != baseline


def test_dependency_change_invalidates_prior_approval(repo, monkeypatch):
    # #5: the approved identity now pins the locked dependency set too. A uv.lock bump can
    # change fill/numerical semantics, so a prior human approval must NOT satisfy the gate once
    # the dependency_hash source changes. We monkeypatch the SHARED dependency-hash function.
    from algua.provenance import lockfile

    _advance_to_forward_tested(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = repo.get(STRATEGY)
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)
    assert has_valid_approval(repo, rec.id, code_hash, config_hash, dependency_hash) is True

    # A dependency bump changes the shared dependency hash for everyone.
    monkeypatch.setattr(lockfile, "dependency_hash", lambda: "different-locked-deps")

    new_code, new_config, new_dep = compute_artifact_hashes(STRATEGY)
    assert new_dep == "different-locked-deps"
    assert has_valid_approval(repo, rec.id, new_code, new_config, new_dep) is False
    with pytest.raises(TransitionError):
        _to_live(repo, STRATEGY, Actor.HUMAN)


def test_legacy_null_dependency_row_never_matches(repo):
    # Fail-closed: an approval row written before dependency_hash existed (NULL) must never
    # satisfy the stricter gate, even when code+config match. No `OR IS NULL` escape hatch.
    repo.add(STRATEGY)
    rec = repo.get(STRATEGY)
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)
    repo.record_approval(rec.id, code_hash, config_hash, None, "legacy")
    assert has_valid_approval(repo, rec.id, code_hash, config_hash, dependency_hash) is False
    # And a NULL probe (no lockfile present) is refused outright.
    assert has_valid_approval(repo, rec.id, code_hash, config_hash, None) is False


def test_live_transition_records_full_identity_hashes(repo):
    # Audit symmetry: a successful forward_tested -> live transition must record the full pinned
    # identity (code, config, AND dependency hash) in the stage_transitions history.
    _advance_to_forward_tested(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    code_hash, config_hash, dependency_hash = compute_artifact_hashes(STRATEGY)

    _to_live(repo, STRATEGY, Actor.HUMAN)

    live_row = repo.list_transitions(STRATEGY)[-1]
    assert live_row["to_stage"] == Stage.LIVE.value
    assert live_row["code_hash"] == code_hash
    assert live_row["config_hash"] == config_hash
    assert live_row["dependency_hash"] == dependency_hash


def test_non_live_transition_records_null_dependency_hash(repo):
    # Non-live transitions carry no hashes; dependency_hash stays NULL, exactly as code/config do.
    repo.add(STRATEGY)
    transition_strategy(repo, STRATEGY, Stage.BACKTESTED, Actor.AGENT)
    row = repo.list_transitions(STRATEGY)[-1]
    assert row["dependency_hash"] is None
    assert row["code_hash"] is None
    assert row["config_hash"] is None


def test_code_hash_ignores_thirdparty_and_stdlib_changes(repo, monkeypatch):
    # The closure is bounded to algua.* modules: mutating a third-party module's source
    # must NOT change the code_hash (otherwise the hash is non-deterministic across envs).
    import pandas as pd

    baseline, _, _ = compute_artifact_hashes(STRATEGY)

    real_getsource = inspect.getsource

    def fake_getsource(obj):
        if obj is pd:
            return "totally different pandas source"
        return real_getsource(obj)

    monkeypatch.setattr(inspect, "getsource", fake_getsource)
    unchanged, _, _ = compute_artifact_hashes(STRATEGY)

    assert unchanged == baseline
