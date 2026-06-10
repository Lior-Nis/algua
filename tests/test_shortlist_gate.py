import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

_IDENT = type("I", (), {"code_hash": "c0", "config_hash": "cfg0", "dependency_hash": "dep0"})


def _repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _backtested(repo, name="alpha"):
    rec = repo.add(name)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    return repo.get(name)


def _token(repo, sid, *, actor="agent"):
    return repo.record_gate_evaluation(
        sid, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash="c0", config_hash="cfg0",
        dependency_hash="dep0", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor=actor,
        decision_json="{}")


def test_agent_shortlist_refused_without_token(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "try")


def test_agent_shortlist_consumes_token_single_use(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    rec = _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    _token(repo, rec.id)
    assert transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "ok").stage \
        == Stage.CANDIDATE
    repo.apply_transition(repo.get("alpha"), Stage.BACKTESTED, Actor.AGENT, "back")
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "again")


def test_human_token_not_consumable_by_agent(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    rec = _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    _token(repo, rec.id, actor="human")  # human audit row is not an agent token
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "try")


def test_human_shortlist_exempt(tmp_path):
    repo = _repo(tmp_path)
    _backtested(repo)
    assert transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.HUMAN, "manual").stage \
        == Stage.CANDIDATE


def test_token_for_strategy_a_not_consumable_by_strategy_b(tmp_path, monkeypatch):
    # A gate token minted for strategy A must never satisfy a transition on strategy B, even when
    # both share identical recomputed identity (the strategy_id binding is the wall).
    repo = _repo(tmp_path)
    rec_a = _backtested(repo, "alpha")
    _backtested(repo, "beta")
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    _token(repo, rec_a.id)  # token belongs to alpha
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "beta", Stage.CANDIDATE, Actor.AGENT, "steal")
    # alpha's token is untouched and still consumable.
    assert transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "ok").stage \
        == Stage.CANDIDATE
