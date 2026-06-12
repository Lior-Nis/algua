import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

_IDENT = type("I", (), {"code_hash": "c0", "config_hash": "cfg0", "dependency_hash": "dep0"})
_FWD_IDENT = ArtifactIdentity("c0", "cfg0", "dep0")  # unpackable, like the real recomputed one


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


# ---------------------------------------------------------------------------
# Task 9 (#124): source-stage-scoped gate branches + the forward-tested gate
# ---------------------------------------------------------------------------

def _paper(repo, name="alpha"):
    rec = repo.add(name)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.apply_transition(repo.get(name), Stage.CANDIDATE, Actor.HUMAN, "sl")
    repo.apply_transition(repo.get(name), Stage.PAPER, Actor.AGENT, "pp")
    return repo.get(name)


def _forward_token(repo, sid, *, actor="agent"):
    return repo.record_forward_gate_evaluation(
        sid, passed=True, n_forward_observations=70, min_forward_observations=63,
        session_coverage=0.95, realized_sharpe=0.8, holdout_sharpe=1.2, degradation_factor=0.5,
        sharpe_floor=0.3, realized_vol=0.1, min_forward_vol=0.02, realized_max_drawdown=0.05,
        max_forward_drawdown=0.25, first_tick_id=1, last_tick_id=70,
        first_tick_ts="2026-01-02T00:00:00+00:00", last_tick_ts="2026-06-01T00:00:00+00:00",
        max_staleness_sessions=5, n_reconcile_failures=0, n_concurrent_forward=1,
        account_id="acct", code_hash="c0", config_hash="cfg0", dependency_hash="dep0",
        actor=actor, decision_json="{}")


def test_agent_paper_to_candidate_backstep_is_free(tmp_path):
    # The legal paper -> candidate BACK-step must not demand a shortlist token (an agent at
    # paper can never mint one — promotion preflight requires backtested). No monkeypatch on
    # purpose: the back-step must not even recompute identity, so no gate machinery may fire.
    repo = _repo(tmp_path)
    _paper(repo)
    assert transition_strategy(repo, "alpha", Stage.CANDIDATE, Actor.AGENT, "demote").stage \
        == Stage.CANDIDATE


def test_agent_forward_tested_consumes_token(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    rec = _paper(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _FWD_IDENT)
    fid = _forward_token(repo, rec.id)
    assert transition_strategy(repo, "alpha", Stage.FORWARD_TESTED, Actor.AGENT, "ok").stage \
        == Stage.FORWARD_TESTED
    row = repo._conn.execute(
        "SELECT consumed FROM forward_gate_evaluations WHERE id=?", (fid,)).fetchone()
    assert row["consumed"] == 1  # single-use: spent atomically with the stage change


def test_agent_forward_tested_refused_without_token(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _paper(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _FWD_IDENT)
    with pytest.raises(TransitionError, match="algua paper promote"):
        transition_strategy(repo, "alpha", Stage.FORWARD_TESTED, Actor.AGENT, "try")


def test_human_forward_tested_raw_pins_identity(tmp_path, monkeypatch):
    # A human raw transition needs no token, but the stage_transitions row must record the
    # RECOMPUTED identity hashes for audit (#124).
    repo = _repo(tmp_path)
    _paper(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _FWD_IDENT)
    assert transition_strategy(repo, "alpha", Stage.FORWARD_TESTED, Actor.HUMAN, "manual").stage \
        == Stage.FORWARD_TESTED
    last = repo.list_transitions("alpha")[-1]
    assert last["to_stage"] == "forward_tested"
    assert (last["code_hash"], last["config_hash"], last["dependency_hash"]) \
        == ("c0", "cfg0", "dep0")


def test_demote_then_repromote_needs_fresh_token(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    rec = _paper(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _FWD_IDENT)
    _forward_token(repo, rec.id)
    transition_strategy(repo, "alpha", Stage.FORWARD_TESTED, Actor.AGENT, "ok")
    transition_strategy(repo, "alpha", Stage.PAPER, Actor.AGENT, "demote")  # back-step is free
    with pytest.raises(TransitionError, match="algua paper promote"):  # token already consumed
        transition_strategy(repo, "alpha", Stage.FORWARD_TESTED, Actor.AGENT, "again")
