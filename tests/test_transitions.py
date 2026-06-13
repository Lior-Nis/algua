import pytest
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def _paper_strategy(tmp_path):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    return repo


def test_bench_to_dormant_requires_reason(tmp_path):
    repo = _paper_strategy(tmp_path)
    with pytest.raises(TransitionError, match="reason"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="")


def test_bench_to_dormant_with_reason_succeeds(tmp_path):
    repo = _paper_strategy(tmp_path)
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT,
                              reason="seasonal signal degradation")
    assert rec.stage is Stage.DORMANT
    assert repo.list_transitions("s1")[-1]["reason"] == "seasonal signal degradation"
