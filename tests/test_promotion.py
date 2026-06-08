from datetime import date

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.promotion import (
    guard_agent_relaxations,
    promotion_preflight,
    resolve_pit_ok,
)
from algua.registry.store import SqliteStrategyRepository


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_guard_agent_relaxations_refuses_agent():
    for kw in (dict(declared_combos=9, allow_holdout_reuse=False, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=True, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=False, allow_non_pit=True)):
        with pytest.raises(ValueError, match="human"):
            guard_agent_relaxations(Actor.AGENT, **kw)


def test_guard_allows_clean_agent_and_any_human():
    guard_agent_relaxations(Actor.AGENT, declared_combos=None, allow_holdout_reuse=False,
                            allow_non_pit=False)
    guard_agent_relaxations(Actor.HUMAN, declared_combos=9, allow_holdout_reuse=True,
                            allow_non_pit=True)


def test_resolve_pit_ok_requires_coverage():
    cover = [{"snapshot_id": "u1", "effective_date": "2021-06-01"}]
    late = [{"snapshot_id": "u1", "effective_date": "2022-06-01"}]
    assert resolve_pit_ok("sp", cover, date(2022, 1, 1)) is True
    assert resolve_pit_ok("sp", late, date(2022, 1, 1)) is False
    assert resolve_pit_ok(None, None, date(2022, 1, 1)) is False


def test_resolve_pit_ok_fails_closed_on_malformed_snapshot():
    # A malformed/missing effective_date must NOT raise (the holdout was already recorded) — it
    # fails closed to non-PIT (not promotable).
    missing = [{"snapshot_id": "u1"}]  # KeyError
    bad_format = [{"snapshot_id": "u1", "effective_date": "not-a-date"}]  # ValueError
    wrong_type = [{"snapshot_id": "u1", "effective_date": None}]  # TypeError
    assert resolve_pit_ok("sp", missing, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", bad_format, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", wrong_type, date(2022, 1, 1)) is False


@pytest.mark.parametrize("stages", [
    (),                                              # idea
    (Stage.BACKTESTED, Stage.SHORTLISTED),           # shortlisted
    (Stage.BACKTESTED, Stage.SHORTLISTED, Stage.PAPER),  # paper (PAPER->SHORTLISTED is legal!)
])
def test_preflight_refuses_non_backtested_source(tmp_path, stages):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (so stage is the refusal)
    for s in stages:
        rec = repo.apply_transition(rec, s, Actor.HUMAN, "setup")
    with pytest.raises(TransitionError, match="backtested"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_system_actor_before_any_holdout_or_gate_row(tmp_path):
    # SYSTEM is not a valid promote actor. The refusal is the FIRST check in preflight (before the
    # relaxation guard and before walk_forward), so no holdout is burned and no gate row is minted.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (isolate the actor check)
    with pytest.raises(ValueError, match="agent or human"):
        promotion_preflight(repo, "alpha", actor=Actor.SYSTEM, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False)
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_agent_without_measured_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    with pytest.raises(ValueError, match="search breadth"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False)


def test_preflight_resolves_measured_funnel_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 10, "{}")  # sibling effort raises the funnel bar
    ctx = promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False)
    assert ctx.own == 4 and ctx.windowed_total == 14 and ctx.n_funnel == 14
    assert ctx.provenance == "measured"
