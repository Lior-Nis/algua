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


def _advance_to_paper(repo, name):
    repo.add(name)
    transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT)
    transition_strategy(repo, name, Stage.SHORTLISTED, Actor.AGENT)
    transition_strategy(repo, name, Stage.PAPER, Actor.AGENT)


def test_live_requires_approval(repo):
    _advance_to_paper(repo, STRATEGY)
    with pytest.raises(TransitionError):
        transition_strategy(repo, STRATEGY, Stage.LIVE, Actor.HUMAN)


def test_live_requires_human_actor(repo):
    _advance_to_paper(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    with pytest.raises(TransitionError):
        transition_strategy(repo, STRATEGY, Stage.LIVE, Actor.AGENT)


def test_live_succeeds_with_human_and_recorded_approval(repo):
    _advance_to_paper(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = transition_strategy(repo, STRATEGY, Stage.LIVE, Actor.HUMAN)
    assert rec.stage is Stage.LIVE


def test_string_live_engages_gate(repo):
    # Passing the raw string "live" (not Stage.LIVE) must still engage the gate.
    _advance_to_paper(repo, STRATEGY)
    with pytest.raises(TransitionError):
        transition_strategy(repo, STRATEGY, "live", Actor.HUMAN)


def test_string_live_succeeds_with_approval(repo):
    _advance_to_paper(repo, STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = transition_strategy(repo, STRATEGY, "live", Actor.HUMAN)
    assert rec.stage is Stage.LIVE


def test_approval_binds_to_real_source_not_caller_strings(repo):
    # #79: a constant hash supplied at approve time cannot satisfy the gate, because the
    # approval stores the *recomputed* source hash and the gate recomputes it again.
    _advance_to_paper(repo, STRATEGY)
    rec = repo.get(STRATEGY)
    # An attacker manually inserts a constant-hash approval row, mimicking "approve --code-hash X".
    repo.record_approval(rec.id, "constant", "constant", "attacker")
    with pytest.raises(TransitionError):
        transition_strategy(repo, STRATEGY, Stage.LIVE, Actor.HUMAN)


def test_recorded_approval_matches_computed_hashes(repo):
    repo.add(STRATEGY)
    code_hash, config_hash = compute_artifact_hashes(STRATEGY)
    record_approval(repo, STRATEGY, "lior")
    rec = repo.get(STRATEGY)
    assert has_valid_approval(repo, rec.id, code_hash, config_hash) is True


def test_compute_artifact_hashes_is_deterministic_and_distinct(repo):
    code_hash, config_hash = compute_artifact_hashes(STRATEGY)
    again = compute_artifact_hashes(STRATEGY)
    assert (code_hash, config_hash) == again
    assert code_hash != config_hash  # source digest is not the config digest


def test_has_valid_approval(repo):
    repo.add(STRATEGY)
    s = repo.get(STRATEGY)
    code_hash, config_hash = compute_artifact_hashes(STRATEGY)
    assert has_valid_approval(repo, s.id, code_hash, config_hash) is False
    record_approval(repo, STRATEGY, "lior")
    assert has_valid_approval(repo, s.id, code_hash, config_hash) is True


def test_record_approval_rejects_blank_approver(repo):
    repo.add(STRATEGY)
    with pytest.raises(ValueError):
        record_approval(repo, STRATEGY, " ")
