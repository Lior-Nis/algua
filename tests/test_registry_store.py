import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.repository import StrategyExists
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


@pytest.fixture()
def repo(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return SqliteStrategyRepository(c)


def _transition(repo, name, to, actor, reason=None):
    return transition_strategy(repo, name, to, actor, reason)


def test_add_creates_idea_with_initial_transition(repo):
    rec = repo.add("alpha")
    assert rec.stage is Stage.IDEA
    transitions = repo.list_transitions("alpha")
    assert len(transitions) == 1
    assert transitions[0]["to_stage"] == "idea"
    assert transitions[0]["from_stage"] is None
    assert transitions[0]["actor"] == "system"


def test_duplicate_name_raises(repo):
    repo.add("alpha")
    with pytest.raises(StrategyExists):
        repo.add("alpha")


def test_legal_transition_updates_stage_and_history(repo):
    repo.add("alpha")
    rec = _transition(repo, "alpha", Stage.BACKTESTED, Actor.AGENT, "ran backtest")
    assert rec.stage is Stage.BACKTESTED
    assert len(repo.list_transitions("alpha")) == 2


def test_transition_records_true_from_stage(repo):
    repo.add("alpha")
    _transition(repo, "alpha", Stage.BACKTESTED, Actor.AGENT)
    last = repo.list_transitions("alpha")[-1]
    assert last["from_stage"] == "idea"
    assert last["to_stage"] == "backtested"


def test_transition_accepts_enum_values_as_strings(repo):
    repo.add("alpha")
    rec = _transition(repo, "alpha", "backtested", "agent")
    assert rec.stage is Stage.BACKTESTED


def test_illegal_transition_raises(repo):
    repo.add("alpha")
    with pytest.raises(TransitionError):
        _transition(repo, "alpha", Stage.LIVE, Actor.AGENT)


def test_transition_service_allows_injected_live_approval_verifier(repo):
    repo.add("cross_sectional_momentum")
    # SHORTLISTED via human: scaffolding to a later stage, not exercising the agent shortlist gate.
    _transition(repo, "cross_sectional_momentum", Stage.BACKTESTED, Actor.AGENT)
    _transition(repo, "cross_sectional_momentum", Stage.SHORTLISTED, Actor.HUMAN)
    _transition(repo, "cross_sectional_momentum", Stage.PAPER, Actor.AGENT)

    rec = transition_strategy(
        repo,
        "cross_sectional_momentum",
        Stage.LIVE,
        Actor.HUMAN,
        approval_verifier=lambda *_args: True,
    )

    assert rec.stage is Stage.LIVE


def test_list_filters_by_stage(repo):
    repo.add("alpha")
    repo.add("beta")
    _transition(repo, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = repo.list_strategies(Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]


# --- holdout_evaluations -----------------------------------------------------

def test_record_and_query_overlapping_holdout(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Overlapping period, same data_source + holdout_frac -> collision.
    assert repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-06-01", period_end="2024-06-01", holdout_frac=0.2,
    )


def test_holdout_non_overlap_and_frac_and_data_distinguish(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Disjoint period -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2,
    )
    # Different holdout_frac -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.3,
    )
    # Different data_source -> no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
    )


def test_holdout_snapshot_identity_takes_precedence(repo):
    rec = repo.add("alpha")
    repo.record_holdout_evaluation(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapA",
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
        config_hash="cfg", reused=False,
    )
    # Same window, different snapshot id -> distinct data identity, no collision.
    assert not repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapB",
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
    )
    # Same snapshot id, overlapping window -> collision.
    assert repo.overlapping_holdout_evaluations(
        rec.id, data_source="StoreBackedProvider", snapshot_id="snapA",
        period_start="2022-06-01", period_end="2023-06-01", holdout_frac=0.2,
    )


def _record_pass(repo, sid, *, actor="agent", code="c0", config="cfg0", dep="dep0"):
    return repo.record_gate_evaluation(
        sid, passed=True, n_funnel=9, own_lifetime_combos=9, windowed_total_combos=9,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash=code, config_hash=config,
        dependency_hash=dep, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor=actor,
        decision_json="{}")


def test_windowed_search_combos_sums_recent(repo):
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 5, "{}")
    assert repo.windowed_search_combos(window_days=90) == 9


def test_find_consumable_matches_agent_passing_identity(repo):
    rec = repo.add("alpha")
    gid = _record_pass(repo, rec.id)
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") == gid


def test_find_consumable_ignores_human_failing_and_mismatch(repo):
    rec = repo.add("alpha")
    _record_pass(repo, rec.id, actor="human")            # human row is not a token
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None
    repo.record_gate_evaluation(  # failing agent row
        rec.id, passed=False, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash="c0", config_hash="cfg0",
        dependency_hash="dep0", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}")
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None
    gid = _record_pass(repo, rec.id)                     # passing agent row, but...
    assert repo.find_consumable_gate_evaluation(rec.id, "BAD", "cfg0", "dep0") is None  # identity
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", None) is None     # NULL dep
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") == gid


def test_apply_transition_consumes_token_atomically(repo):
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    rec = repo.get("alpha")
    gid = _record_pass(repo, rec.id)
    out = repo.apply_transition(rec, Stage.SHORTLISTED, Actor.AGENT, "go", consume_gate_id=gid)
    assert out.stage == Stage.SHORTLISTED
    # token consumed (single-use)
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None


def test_apply_transition_bad_token_rolls_back(repo):
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    rec = repo.get("alpha")
    with pytest.raises(TransitionError):
        repo.apply_transition(rec, Stage.SHORTLISTED, Actor.AGENT, "go", consume_gate_id=999999)
    assert repo.get("alpha").stage == Stage.BACKTESTED  # stage unchanged (rolled back)
