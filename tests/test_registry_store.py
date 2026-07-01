from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.db import connect, migrate
from algua.registry.repository import (
    FdrGateOutcome,
    FdrStreamState,
    FunnelDriftError,
    FunnelSnapshot,
    StrategyExists,
)
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def test_record_exposes_metadata_defaults(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("plain")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []
    assert rec.family is None
    assert rec.derived_from is None
    assert rec.description is None


def test_null_metadata_columns_read_as_defaults(tmp_path):
    # A row written before the columns existed (all NULL) must read as the defaults.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy', 'idea', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    rec = SqliteStrategyRepository(conn).get("legacy")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []


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
    # CANDIDATE via human: scaffolding to forward_tested, not exercising the agent shortlist gate.
    _transition(repo, "cross_sectional_momentum", Stage.BACKTESTED, Actor.AGENT)
    _transition(repo, "cross_sectional_momentum", Stage.CANDIDATE, Actor.HUMAN)
    _transition(repo, "cross_sectional_momentum", Stage.PAPER, Actor.AGENT)
    _transition(repo, "cross_sectional_momentum", Stage.FORWARD_TESTED, Actor.HUMAN,
                "test setup")

    rec = transition_strategy(
        repo,
        "cross_sectional_momentum",
        Stage.LIVE,
        Actor.HUMAN,
        approval_verifier=lambda *_args: True,
        # A passing certificate stub keeps this test on its named invariant — the injected
        # APPROVAL verifier seam — not the forward-certificate wall in front of it (#124).
        forward_certificate_verifier=lambda *_args: {"id": 1},
    )

    assert rec.stage is Stage.LIVE


def test_list_filters_by_stage(repo):
    repo.add("alpha")
    repo.add("beta")
    _transition(repo, "beta", Stage.BACKTESTED, Actor.AGENT)
    ideas = repo.list_strategies(Stage.IDEA)
    assert [r.name for r in ideas] == ["alpha"]


# --- holdout reservation: interval matching (#192) --------------------------

@pytest.fixture()
def repo_with_strategy(repo):
    return repo, repo.add("h").id


def _reserve(repo, sid, *, hs, he, frac=0.2, ps="2022-01-01", pe="2023-12-31",
             ds="demo", snap=None, allow_reuse=False):
    return repo.reserve_holdout(
        sid, data_source=ds, snapshot_id=snap, period_start=ps, period_end=pe,
        holdout_frac=frac, holdout_start=hs, holdout_end=he, allow_reuse=allow_reuse)


def test_overlapping_interval_blocks(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-09-01", he="2024-03-01")


def test_different_holdout_frac_same_interval_still_blocks(repo_with_strategy):
    # The #192 exploit: a different --holdout-frac must NOT escape the guard when the OOS bars
    # overlap. Identity is the interval, not the frac.
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", frac=0.2)
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", frac=0.4)


def test_non_overlapping_interval_allowed(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2021-06-01", he="2021-12-31", ps="2020-01-01", pe="2021-12-31")
    rid, reused = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    assert rid and reused is False


def test_allow_reuse_overrides_overlap(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    rid, reused = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", allow_reuse=True)
    assert rid and reused is True


def test_null_interval_row_fails_closed(repo_with_strategy):
    # An old-code/legacy row with a NULL interval must match unconditionally (fail closed).
    repo, sid = repo_with_strategy
    repo._conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at, committed_at, holdout_start, holdout_end)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, "demo", None, "2022-01-01", "2023-12-31", 0.2, "h", 0,
         "2022-01-01T00:00:00+00:00", "2022-02-01T00:00:00+00:00", None, None),
    )
    repo._conn.commit()
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")


def test_finalize_and_release_unchanged(repo_with_strategy):
    repo, sid = repo_with_strategy
    rid, _ = _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")
    repo.finalize_holdout_reservation(rid, config_hash="real-evidence-hash", strategy_id=sid)
    row = repo._conn.execute(
        "SELECT committed_at, config_hash FROM holdout_evaluations WHERE id = ?", (rid,)
    ).fetchone()
    assert row["committed_at"] and row["config_hash"] == "real-evidence-hash"

    rid2, _ = _reserve(repo, sid, hs="2024-06-01", he="2024-12-31",
                       ps="2024-01-01", pe="2024-12-31")
    repo.release_holdout_reservation(rid2)
    gone = repo._conn.execute(
        "SELECT 1 FROM holdout_evaluations WHERE id = ?", (rid2,)
    ).fetchone()
    assert gone is None


def test_reserve_inside_open_transaction_raises(repo_with_strategy):
    repo, sid = repo_with_strategy
    repo._conn.execute("BEGIN")  # simulate a caller holding an open transaction
    try:
        with pytest.raises(RuntimeError, match="open transaction"):
            _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")
    finally:
        repo._conn.rollback()


def test_finalize_twice_raises(repo_with_strategy):
    repo, sid = repo_with_strategy
    rid, _ = _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")
    repo.finalize_holdout_reservation(rid, config_hash="h1", strategy_id=sid)
    with pytest.raises(ValueError):
        repo.finalize_holdout_reservation(rid, config_hash="h2", strategy_id=sid)


def test_release_after_finalize_is_noop(repo_with_strategy):
    repo, sid = repo_with_strategy
    rid, _ = _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")
    repo.finalize_holdout_reservation(rid, config_hash="h1", strategy_id=sid)
    repo.release_holdout_reservation(rid)  # no-op, no raise
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")


def test_reserve_is_provenance_independent(repo_with_strategy):
    # #205: the OOS calendar window is the single-use unit REGARDLESS of provenance. A snapshot
    # burn now blocks a non-snapshot probe over the same interval (was: distinct provenance bucket).
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2022-06-01", he="2022-12-31", snap="snapA")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2022-06-01", he="2022-12-31", snap=None)
    # A DIFFERENT snapshot of an overlapping window is also blocked.
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2022-09-01", he="2023-03-31", snap="snapB")


def test_partial_overlap_cross_provenance_blocks(repo_with_strategy):
    # GATE-1 CRITICAL: a burn over snapshot S blocks a PARTIALLY-overlapping probe via a different
    # source. Whole-window content hashing could not catch this; the interval-overlap test does.
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-01-01", he="2023-12-31", snap="snapS")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2024-06-30", snap=None, ds="yfinance")


def test_inverted_interval_rejected(repo_with_strategy):
    # Defensive (GATE-1 r2): an inverted incoming interval would slip both the NULL branch and the
    # overlap test and fail OPEN. reserve_holdout rejects start > end.
    repo, sid = repo_with_strategy
    with pytest.raises(ValueError, match="invalid holdout interval"):
        _reserve(repo, sid, hs="2023-12-31", he="2023-01-01")


def test_release_then_reserve_same_interval_succeeds(repo_with_strategy):
    repo, sid = repo_with_strategy
    rid, _ = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    repo.release_holdout_reservation(rid)
    rid2, reused = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    assert rid2 and reused is False


def test_allow_reuse_past_orphaned_pending_reservation(repo_with_strategy):
    # A pending (uncommitted) reservation blocks fail-closed; allow_reuse overrides it too.
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")  # pending, never finalized (orphan)
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-09-01", he="2024-03-01")
    rid, reused = _reserve(repo, sid, hs="2023-09-01", he="2024-03-01", allow_reuse=True)
    assert rid and reused is True


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
    out = repo.apply_transition(rec, Stage.CANDIDATE, Actor.AGENT, "go", consume_gate_id=gid)
    assert out.stage == Stage.CANDIDATE
    # token consumed (single-use)
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None


def test_apply_transition_bad_token_rolls_back(repo):
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    rec = repo.get("alpha")
    with pytest.raises(TransitionError):
        repo.apply_transition(rec, Stage.CANDIDATE, Actor.AGENT, "go", consume_gate_id=999999)
    assert repo.get("alpha").stage == Stage.BACKTESTED  # stage unchanged (rolled back)
# ---------------------------------------------------------------------------
# Task 6: add() accepts metadata + derived_from validation
# ---------------------------------------------------------------------------

def test_add_with_metadata_roundtrips(tmp_path):
    from algua.contracts.registry_metadata import Author, HypothesisStatus
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(
        "child",
        family="mean-reversion",
        tags=["Slow", "slow", " carry "],
        author=Author.HUMAN,
        hypothesis_status=HypothesisStatus.SUPPORTED,
        description="a thing",
    )
    assert rec.family == "mean-reversion"
    assert rec.tags == ["carry", "slow"]
    assert rec.author == Author.HUMAN
    assert rec.hypothesis_status == HypothesisStatus.SUPPORTED
    assert rec.description == "a thing"


def test_add_derived_from_requires_existing_parent(tmp_path):
    import pytest

    from algua.registry.db import connect, migrate
    from algua.registry.repository import StrategyNotFound
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(StrategyNotFound):
        repo.add("orphan", derived_from="ghost")
    repo.add("parent")
    rec = repo.add("kid", derived_from="parent")
    assert rec.derived_from == "parent"


def test_add_derived_from_rejects_self(tmp_path):
    import pytest

    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(ValueError):
        repo.add("self", derived_from="self")


# ---------------------------------------------------------------------------
# Task 7: update_metadata repository method
# ---------------------------------------------------------------------------

def test_update_metadata_partial(tmp_path):
    from algua.contracts.registry_metadata import HypothesisStatus
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", family="mean-reversion", tags=["slow"])
    before = repo.get("a")
    rec = repo.update_metadata(
        "a", hypothesis_status=HypothesisStatus.SUPPORTED, add_tags=["carry"]
    )
    assert rec.hypothesis_status == HypothesisStatus.SUPPORTED
    assert rec.tags == ["carry", "slow"]
    assert rec.family == "mean-reversion"  # untouched
    assert rec.updated_at >= before.updated_at


def test_update_metadata_remove_tag(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", tags=["slow", "carry"])
    rec = repo.update_metadata("a", remove_tags=["slow"])
    assert rec.tags == ["carry"]


def test_update_metadata_derived_from_validation(tmp_path):
    import pytest

    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a")
    with pytest.raises(ValueError):
        repo.update_metadata("a", derived_from="a")


def test_update_metadata_unknown_parent_raises(tmp_path):
    import pytest

    from algua.registry.db import connect, migrate
    from algua.registry.repository import StrategyNotFound
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a")
    with pytest.raises(StrategyNotFound):
        repo.update_metadata("a", derived_from="ghost")


def test_get_unknown_strategy_error_is_self_identifying(tmp_path):
    """get() on a missing strategy renders a self-describing message, not the bare name (#271).

    The CLI surfaces str(exc) as the JSON error; a bare '<name>' is indistinguishable from
    any other bare-string error, so the message must name the failure mode.
    """
    import pytest

    from algua.registry.db import connect, migrate
    from algua.registry.repository import StrategyNotFound
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(StrategyNotFound, match="strategy not found: ghost"):
        repo.get("ghost")


# ---------------------------------------------------------------------------
# Task 9: list_strategies filter params
# ---------------------------------------------------------------------------

def _seed_pool(repo):
    from algua.contracts.registry_metadata import Author, HypothesisStatus
    repo.add("a", family="mean-reversion", tags=["slow"], author=Author.AGENT)
    repo.add("b", family="mean-reversion", tags=["slow", "carry"], author=Author.HUMAN,
             hypothesis_status=HypothesisStatus.SUPPORTED)
    repo.add("c", family="momentum", tags=["fast"], author=Author.AGENT)


def test_filter_by_family(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _seed_pool(repo)
    assert {r.name for r in repo.list_strategies(family="mean-reversion")} == {"a", "b"}


def test_filter_by_tag_is_all_of(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _seed_pool(repo)
    assert {r.name for r in repo.list_strategies(tags=["slow", "carry"])} == {"b"}


def test_filter_by_author_and_status_compose(tmp_path):
    from algua.contracts.registry_metadata import Author
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _seed_pool(repo)
    got = repo.list_strategies(author=Author.AGENT, family="mean-reversion")
    assert {r.name for r in got} == {"a"}


def test_filter_author_matches_null_legacy_row(tmp_path):
    # A legacy NULL-author row must match --author agent (COALESCE semantics).
    from algua.contracts.registry_metadata import Author
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('legacy','idea','2026-01-01','2026-01-01')")
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    assert {r.name for r in repo.list_strategies(author=Author.AGENT)} == {"legacy"}


def test_filter_by_stage_and_family(tmp_path):
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _seed_pool(repo)  # a,b are mean-reversion at idea; c is momentum at idea
    rec_a = repo.get("a")
    repo.apply_transition(rec_a, Stage.BACKTESTED, Actor.AGENT)
    # only b remains at idea within mean-reversion
    assert {r.name for r in repo.list_strategies(Stage.IDEA, family="mean-reversion")} == {"b"}


# ---------------------------------------------------------------------------
# Task 13: backfill_metadata — fills only NULL columns
# ---------------------------------------------------------------------------

def test_backfill_metadata_fills_only_nulls(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    # legacy NULL row (bypasses add() which writes defaults)
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('a','idea','2026-01-01','2026-01-01')")
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    repo.backfill_metadata("a", family="mean-reversion", hypothesis_status="supported")
    rec = repo.get("a")
    assert rec.family == "mean-reversion"
    assert rec.hypothesis_status.value == "supported"
    # second backfill must NOT overwrite a now-non-NULL value
    repo.backfill_metadata("a", family="momentum")
    assert repo.get("a").family == "mean-reversion"


def test_backfill_metadata_all_none_is_noop(tmp_path):
    # Calling backfill_metadata with all args None must be a clean no-op (no error,
    # no mutation): the returned record must equal the pre-call state.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", family="momentum", tags=["fast"])
    before = repo.get("a")
    after = repo.backfill_metadata("a")
    assert after.family == before.family
    assert after.tags == before.tags
    assert after.author == before.author
    assert after.hypothesis_status == before.hypothesis_status
    assert after.derived_from == before.derived_from
    assert after.description == before.description


# ---------------------------------------------------------------------------
# Fix 2: default_fill_metadata_nulls — lives in store, not CLI
# ---------------------------------------------------------------------------

def test_default_fill_metadata_nulls_fills_remaining_nulls(tmp_path):
    """default_fill_metadata_nulls must fill author/hypothesis_status/tags NULLs to defaults."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    # Raw-insert a legacy NULL row (bypasses add() which writes defaults).
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    repo.default_fill_metadata_nulls()
    rec = repo.get("legacy")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []


def test_default_fill_metadata_nulls_does_not_overwrite_existing(tmp_path):
    """default_fill_metadata_nulls must leave already-set values untouched."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", author=Author.HUMAN, hypothesis_status=HypothesisStatus.SUPPORTED,
             tags=["carry"])
    repo.default_fill_metadata_nulls()
    rec = repo.get("a")
    assert rec.author == Author.HUMAN
    assert rec.hypothesis_status == HypothesisStatus.SUPPORTED
    assert rec.tags == ["carry"]


# ---------------------------------------------------------------------------
# Fix 3: tag filter with malformed non-NULL JSON must not crash
# ---------------------------------------------------------------------------

def test_list_strategies_tag_filter_tolerates_malformed_tags(tmp_path):
    """list_strategies(tags=[...]) must not crash on a row with malformed non-NULL tags."""
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    # Raw-insert a row with malformed tags JSON.
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at, tags) VALUES "
        "('malformed','idea','2026-01-01','2026-01-01','not json')"
    )
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    # Must not raise; malformed row must simply be absent from results.
    results = repo.list_strategies(tags=["x"])
    assert all(r.name != "malformed" for r in results)


# ---------------------------------------------------------------------------
# Task 8 (#124): forward-gate token store methods + CAS stage updates
# ---------------------------------------------------------------------------

def _gate_row(*, passed=True, code="c0", config="cfg0", dep="dep0"):
    """record_forward_gate_evaluation's row kwargs minus actor/consumable — the shape
    record_forward_pass_and_promote takes as ``gate_row``."""
    return dict(
        passed=passed, n_forward_observations=70, min_forward_observations=63,
        session_coverage=0.95, realized_sharpe=0.8, holdout_sharpe=1.2, degradation_factor=0.5,
        sharpe_floor=0.3, realized_vol=0.1, min_forward_vol=0.02, realized_max_drawdown=0.05,
        max_forward_drawdown=0.25, first_tick_id=1, last_tick_id=70,
        first_tick_ts="2026-01-02T00:00:00+00:00", last_tick_ts="2026-06-01T00:00:00+00:00",
        max_staleness_sessions=5, n_reconcile_failures=0, n_concurrent_forward=1,
        account_id="acct", code_hash=code, config_hash=config, dependency_hash=dep,
        decision_json="{}")


def _record_forward(repo, sid, *, passed=True, actor="agent", code="c0", config="cfg0",
                    dep="dep0", consumable=True):
    return repo.record_forward_gate_evaluation(
        sid, **_gate_row(passed=passed, code=code, config=config, dep=dep), actor=actor,
        consumable=consumable)


def _now_iso():
    return datetime.now(UTC).isoformat()


def _find_forward(repo, sid, code="c0", config="cfg0", dep="dep0"):
    return repo.find_consumable_forward_gate_evaluation(
        sid, code, config, dep, now=_now_iso(), ttl_days=7)


def _to_paper(repo, name):
    repo.apply_transition(repo.get(name), Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.apply_transition(repo.get(name), Stage.CANDIDATE, Actor.HUMAN, "sl")
    repo.apply_transition(repo.get(name), Stage.PAPER, Actor.AGENT, "pp")
    return repo.get(name)


def test_record_forward_gate_evaluation_persists_pass_and_fail(repo):
    rec = repo.add("alpha")
    fid_pass = _record_forward(repo, rec.id)
    fid_fail = _record_forward(repo, rec.id, passed=False)
    rows = repo._conn.execute(
        "SELECT id, passed, consumed FROM forward_gate_evaluations WHERE strategy_id=?"
        " ORDER BY id", (rec.id,)).fetchall()
    assert [(r["id"], r["passed"], r["consumed"]) for r in rows] == [
        (fid_pass, 1, 0), (fid_fail, 0, 0)]


def test_find_consumable_forward_matches_agent_passing_identity(repo):
    rec = repo.add("alpha")
    fid = _record_forward(repo, rec.id)
    assert _find_forward(repo, rec.id) == fid


def test_record_forward_not_consumable_is_born_consumed_certificate(repo):
    # consumable=False writes the row already consumed: a CERTIFICATE for the live wall, never
    # a re-entry token a demoted strategy could bank (#124 GATE-2).
    rec = repo.add("alpha")
    fid = _record_forward(repo, rec.id, consumable=False)
    row = repo._conn.execute(
        "SELECT consumed FROM forward_gate_evaluations WHERE id=?", (fid,)).fetchone()
    assert row["consumed"] == 1
    assert _find_forward(repo, rec.id) is None
    # The certificate path ignores consumed: the live wall still sees the row.
    latest = repo.latest_forward_gate_row(rec.id, "c0", "cfg0", "dep0")
    assert latest is not None and latest["id"] == fid


def test_find_consumable_forward_rejects_each_disqualifier(repo):
    rec = repo.add("alpha")
    _record_forward(repo, rec.id, passed=False)          # failed row is not a token
    assert _find_forward(repo, rec.id) is None
    _record_forward(repo, rec.id, actor="human")         # human row is not a token
    assert _find_forward(repo, rec.id) is None
    fid = _record_forward(repo, rec.id)                  # consumed row is not a token
    repo._conn.execute("UPDATE forward_gate_evaluations SET consumed=1 WHERE id=?", (fid,))
    repo._conn.commit()
    assert _find_forward(repo, rec.id) is None
    fid = _record_forward(repo, rec.id)                  # fresh valid token, but...
    assert _find_forward(repo, rec.id, code="BAD") is None       # code drift
    assert _find_forward(repo, rec.id, config="BAD") is None     # config drift
    assert _find_forward(repo, rec.id, dep="BAD") is None        # dependency drift
    assert _find_forward(repo, rec.id, dep=None) is None         # NULL-dep probe fails closed
    assert _find_forward(repo, rec.id) == fid
    # TTL: a token older than the window is stale, never bankable
    stale = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    repo._conn.execute(
        "UPDATE forward_gate_evaluations SET created_at=? WHERE id=?", (stale, fid))
    repo._conn.commit()
    assert _find_forward(repo, rec.id) is None


def test_latest_forward_gate_row_newest_pass_or_fail(repo):
    rec = repo.add("alpha")
    assert repo.latest_forward_gate_row(rec.id, "c0", "cfg0", "dep0") is None
    fid_pass = _record_forward(repo, rec.id)
    row = repo.latest_forward_gate_row(rec.id, "c0", "cfg0", "dep0")
    assert row["id"] == fid_pass and row["passed"] == 1
    assert row["n_forward_observations"] == 70           # full-column dict, not just the id
    fid_fail = _record_forward(repo, rec.id, passed=False)
    # a newer FAILED re-evaluation beats the older pass — it must invalidate it
    row = repo.latest_forward_gate_row(rec.id, "c0", "cfg0", "dep0")
    assert row["id"] == fid_fail and row["passed"] == 0
    assert repo.latest_forward_gate_row(rec.id, "BAD", "cfg0", "dep0") is None  # identity drift
    assert repo.latest_forward_gate_row(rec.id, "c0", "cfg0", None) is None     # NULL-dep probe


def test_apply_transition_consumes_forward_token_atomically(repo):
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    fid = _record_forward(repo, rec.id)
    out = repo.apply_transition(
        rec, Stage.FORWARD_TESTED, Actor.AGENT, "fw", code_hash="c0", config_hash="cfg0",
        dependency_hash="dep0", consume_forward_gate_id=fid)
    assert out.stage is Stage.FORWARD_TESTED
    assert _find_forward(repo, rec.id) is None           # token consumed (single-use)
    # a second consume of the now-spent token fails and does NOT advance the stage
    repo.apply_transition(out, Stage.PAPER, Actor.AGENT, "back")
    rec = repo.get("alpha")
    with pytest.raises(TransitionError):
        repo.apply_transition(
            rec, Stage.FORWARD_TESTED, Actor.AGENT, "fw2", code_hash="c0", config_hash="cfg0",
            dependency_hash="dep0", consume_forward_gate_id=fid)
    assert repo.get("alpha").stage is Stage.PAPER


def test_apply_transition_rejects_both_consume_params(repo):
    rec = repo.add("alpha")
    with pytest.raises(ValueError):
        repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "x",
                              consume_gate_id=1, consume_forward_gate_id=1)


def test_forward_consume_rechecks_identity_at_consume_time(repo):
    """The consume UPDATE re-checks the FULL predicate — a drifted identity at consume time is
    refused even when the caller hands a real token id (no validate-then-consume gap)."""
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    fid = _record_forward(repo, rec.id)                  # token bound to c0/cfg0/dep0
    with pytest.raises(TransitionError):
        repo.apply_transition(
            rec, Stage.FORWARD_TESTED, Actor.AGENT, "fw", code_hash="DIFFERENT",
            config_hash="cfg0", dependency_hash="dep0", consume_forward_gate_id=fid)
    assert repo.get("alpha").stage is Stage.PAPER        # stage unchanged
    assert _find_forward(repo, rec.id) == fid            # token unconsumed


def test_forward_consume_refuses_human_and_foreign_token_ids(repo):
    """A hand-held id pointing at a HUMAN row or another strategy's token is refused at consume
    time — the WHERE re-checks actor and strategy_id, not just row existence."""
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    hid = _record_forward(repo, rec.id, actor="human")   # human row, matching identity
    with pytest.raises(TransitionError):
        repo.apply_transition(
            rec, Stage.FORWARD_TESTED, Actor.AGENT, "fw", code_hash="c0", config_hash="cfg0",
            dependency_hash="dep0", consume_forward_gate_id=hid)
    assert repo.get("alpha").stage is Stage.PAPER
    repo.add("beta")
    recb = _to_paper(repo, "beta")
    fid = _record_forward(repo, rec.id)                  # alpha's token...
    with pytest.raises(TransitionError):
        repo.apply_transition(                           # ...spent against beta
            recb, Stage.FORWARD_TESTED, Actor.AGENT, "fw", code_hash="c0", config_hash="cfg0",
            dependency_hash="dep0", consume_forward_gate_id=fid)
    assert repo.get("beta").stage is Stage.PAPER
    assert _find_forward(repo, rec.id) == fid            # alpha's token untouched


def test_cas_failure_rolls_back_a_successful_token_consume(tmp_path):
    """The token consume and the stage CAS live in ONE transaction: when another session wins the
    stage race AFTER this session's consume UPDATE already succeeded, the rollback must un-spend
    the token — losing a race may cost a retry, never the evidence."""
    db = tmp_path / "r.db"
    c1 = connect(db)
    migrate(c1)
    repo1 = SqliteStrategyRepository(c1)
    repo1.add("alpha")
    rec = _to_paper(repo1, "alpha")                      # this session's view: stage=paper
    fid = _record_forward(repo1, rec.id)
    repo2 = SqliteStrategyRepository(connect(db))
    repo2.apply_transition(repo2.get("alpha"), Stage.CANDIDATE, Actor.AGENT, "won the race")
    with pytest.raises(TransitionError, match="concurrent"):
        repo1.apply_transition(
            rec, Stage.FORWARD_TESTED, Actor.AGENT, "fw", code_hash="c0", config_hash="cfg0",
            dependency_hash="dep0", consume_forward_gate_id=fid)
    assert repo1.get("alpha").stage is Stage.CANDIDATE   # the winner's stage stands
    assert _find_forward(repo1, rec.id) == fid           # consume rolled back with the txn


def test_record_forward_pass_and_promote_atomic_happy_path(repo):
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    gate_id, out = repo.record_forward_pass_and_promote(
        rec, gate_row=_gate_row(), actor=Actor.AGENT, reason="fw")
    assert out.stage is Stage.FORWARD_TESTED
    assert repo.get("alpha").stage is Stage.FORWARD_TESTED
    row = repo._conn.execute(
        "SELECT passed, consumed, actor FROM forward_gate_evaluations WHERE id=?",
        (gate_id,)).fetchone()
    assert (row["passed"], row["consumed"], row["actor"]) == (1, 1, "agent")
    # Born-and-spent: never findable as a re-entry token...
    assert _find_forward(repo, rec.id) is None
    # ...while the live wall's certificate selection (which ignores consumed) still sees it.
    latest = repo.latest_forward_gate_row(rec.id, "c0", "cfg0", "dep0")
    assert latest is not None and latest["id"] == gate_id
    last = repo.list_transitions("alpha")[-1]
    assert (last["from_stage"], last["to_stage"], last["actor"], last["reason"]) == (
        "paper", "forward_tested", "agent", "fw")
    assert (last["code_hash"], last["config_hash"], last["dependency_hash"]) == (
        "c0", "cfg0", "dep0")


def test_record_forward_pass_and_promote_race_leaves_no_row_at_all(tmp_path):
    """THE GATE-2 banking window: losing the stage CAS must roll back the just-inserted passing
    row too. The loser leaves NO row — its decision is lost (re-run the gate), so no consumed=0
    pass can ever be banked for a post-demotion re-entry without a fresh gate run."""
    db = tmp_path / "r.db"
    c1 = connect(db)
    migrate(c1)
    repo1 = SqliteStrategyRepository(c1)
    repo1.add("alpha")
    rec = _to_paper(repo1, "alpha")                      # this session's view: stage=paper
    repo2 = SqliteStrategyRepository(connect(db))
    repo2.apply_transition(repo2.get("alpha"), Stage.CANDIDATE, Actor.AGENT, "won the race")
    with pytest.raises(TransitionError, match="concurrent"):
        repo1.record_forward_pass_and_promote(
            rec, gate_row=_gate_row(), actor=Actor.AGENT, reason="fw")
    assert repo1.get("alpha").stage is Stage.CANDIDATE   # the winner's stage stands
    assert c1.execute(
        "SELECT COUNT(*) FROM forward_gate_evaluations WHERE consumed=0 AND passed=1"
    ).fetchone()[0] == 0                                 # nothing banked...
    assert c1.execute(
        "SELECT COUNT(*) FROM forward_gate_evaluations").fetchone()[0] == 0  # ...no row at all


def test_record_forward_pass_and_promote_human_row_born_consumed(repo):
    """The HUMAN pass-from-paper row is born consumed too: identical observable effect (a human
    row was never consumable anyway — the actor='agent' token filter), one uniform semantics."""
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    gate_id, out = repo.record_forward_pass_and_promote(
        rec, gate_row=_gate_row(), actor=Actor.HUMAN, reason="fw")
    assert out.stage is Stage.FORWARD_TESTED
    row = repo._conn.execute(
        "SELECT consumed, actor FROM forward_gate_evaluations WHERE id=?", (gate_id,)).fetchone()
    assert (row["consumed"], row["actor"]) == (1, "human")
    last = repo.list_transitions("alpha")[-1]
    assert (last["to_stage"], last["actor"]) == ("forward_tested", "human")


def test_record_forward_pass_and_promote_refuses_failing_row(repo):
    repo.add("alpha")
    rec = _to_paper(repo, "alpha")
    with pytest.raises(ValueError, match="PASS path"):
        repo.record_forward_pass_and_promote(
            rec, gate_row=_gate_row(passed=False), actor=Actor.AGENT, reason="fw")
    assert repo.get("alpha").stage is Stage.PAPER
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM forward_gate_evaluations").fetchone()[0] == 0


def test_apply_transition_cas_detects_concurrent_stage_move(tmp_path):
    db = tmp_path / "r.db"
    c1 = connect(db)
    migrate(c1)
    repo1 = SqliteStrategyRepository(c1)
    repo1.add("alpha")
    repo2 = SqliteStrategyRepository(connect(db))
    stale = repo1.get("alpha")                           # this session's view: stage=idea
    repo2.apply_transition(repo2.get("alpha"), Stage.BACKTESTED, Actor.AGENT, "won the race")
    # the loser's error names the race (stage moved underneath), not an illegal transition
    with pytest.raises(TransitionError, match="concurrent"):
        repo1.apply_transition(stale, Stage.BACKTESTED, Actor.AGENT, "lost the race")
    assert repo1.get("alpha").stage is Stage.BACKTESTED
    n = c1.execute(
        "SELECT COUNT(*) c FROM stage_transitions WHERE to_stage='backtested'").fetchone()["c"]
    assert n == 1                                        # exactly one transition row, not two


def test_apply_transition_revokes_allocation_atomically(tmp_path):
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry import allocations
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    allocations.allocate(conn, rec.id, capital=10_000.0, actor="human", account_equity=50_000.0)
    assert allocations.active_allocation(conn, rec.id) is not None
    rec = repo.apply_transition(rec, Stage.DORMANT, Actor.AGENT, reason="bench",
                                revoke_allocation=True)
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, rec.id) is None


def test_apply_transition_revoke_rolls_back_with_stage_on_cas_failure(tmp_path):
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry import allocations
    from algua.registry.repository import StrategyRecord
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    allocations.allocate(conn, rec.id, capital=10_000.0, actor="human", account_equity=50_000.0)
    # Force the stage CAS to fail by passing a stale from-stage (rec says PAPER, DB says LIVE)
    stale = StrategyRecord(id=rec.id, name=rec.name, stage=Stage.PAPER,
                           created_at=rec.created_at, updated_at=rec.updated_at)
    with pytest.raises(TransitionError):
        repo.apply_transition(stale, Stage.DORMANT, Actor.AGENT, reason="bench",
                              revoke_allocation=True)
    assert allocations.active_allocation(conn, rec.id) is not None
    assert repo.get("s1").stage is Stage.LIVE


# ---------------------------------------------------------------------------
# Task 5 (#211): record_search_trial stats + pooled_trial_sharpe_var
# ---------------------------------------------------------------------------


def test_search_trials_records_and_pools_variance(repo):
    # two sweeps with different means -> pooled variance must exceed the mean of within-sweep vars
    repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                             trial_sharpe_mean=0.2, trial_sharpe_var_ann=0.04)
    repo.record_search_trial("s", 2, "{}", trial_sharpe_count=2,
                             trial_sharpe_mean=1.2, trial_sharpe_var_ann=0.04)
    pooled = repo.pooled_trial_sharpe_var("s")
    # exact pooled sample variance:
    # M = (3*0.2 + 2*1.2)/5 = 0.6
    # SSE = (2*0.04 + 3*(0.2-0.6)^2) + (1*0.04 + 2*(1.2-0.6)^2)
    #     = (0.08 + 0.48) + (0.04 + 0.72) = 1.32
    # pooled = 1.32/4 = 0.33
    assert pooled is not None
    assert abs(pooled - 0.33) < 1e-9


def test_pooled_variance_equal_means_matches_naive(repo):
    repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    repo.record_search_trial("s", 2, "{}", trial_sharpe_count=2,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.10)
    # equal means -> between-sweep term zero -> pooled = ((3-1)*0.04+(2-1)*0.10)/(5-1) = 0.045
    pooled = repo.pooled_trial_sharpe_var("s")
    assert pooled is not None
    assert abs(pooled - 0.045) < 1e-9


def test_pooled_variance_none_when_any_stat_missing(repo):
    repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                             trial_sharpe_mean=0.2, trial_sharpe_var_ann=0.04)
    repo.record_search_trial("s", 2, "{}")  # old-style row: NULL stats
    assert repo.pooled_trial_sharpe_var("s") is None


def test_pooled_variance_none_when_no_rows(repo):
    assert repo.pooled_trial_sharpe_var("nope") is None


# ---------------------------------------------------------------------------
# Task 3 (#220 Phase 2): fdr_stream_state — LORD++ alpha-wealth stream accessor
# ---------------------------------------------------------------------------


def _insert_fdr_row(
    repo, sid, *,
    fdr_binding,
    fdr_p_value,
    fdr_alpha_level,
    fdr_rejected,
    fdr_test_index,
    passed=0,
):
    """Insert a gate_evaluations row with FDR columns directly (tests only; Task 4 owns the
    production writer)."""
    repo._conn.execute(
        "INSERT INTO gate_evaluations"
        " (strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
        "  funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
        "  min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
        "  snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
        "  created_at, fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, passed, 9, 9, 9, 90, "measured", 1, 0, 80, 63,
         "c0", "cfg0", "dep0", "SyntheticProvider", None,
         "2022-01-01", "2023-12-31", 0.2, "agent", "{}",
         "2024-01-01T00:00:00+00:00",
         fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index),
    )
    repo._conn.commit()


def test_fdr_stream_empty(repo):
    result = repo.fdr_stream_state()
    assert result == FdrStreamState(t=0, discovery_indices=[])


def test_fdr_stream_non_binding_excluded(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=0, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=None)
    result = repo.fdr_stream_state()
    assert result == FdrStreamState(t=0, discovery_indices=[])


def test_fdr_stream_legacy_null_binding_excluded(repo):
    # A pre-v26 row has fdr_binding=NULL; it must be invisible to the stream.
    sid = repo.add("s").id
    repo._conn.execute(
        "INSERT INTO gate_evaluations"
        " (strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
        "  funnel_window_days, breadth_provenance, pit_ok, holdout_n_bars,"
        "  min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
        "  period_start, period_end, holdout_frac, actor, decision_json, created_at)"
        " VALUES (?,0,9,9,9,90,'measured',1,80,63,'c0','cfg0','dep0','SyntheticProvider',"
        "         '2022-01-01','2023-12-31',0.2,'agent','{}','2024-01-01T00:00:00+00:00')",
        (sid,),
    )
    repo._conn.commit()
    result = repo.fdr_stream_state()
    assert result == FdrStreamState(t=0, discovery_indices=[])


def test_fdr_stream_mixed_binding(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.06, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=1)
    _insert_fdr_row(repo, sid, fdr_binding=0, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=None)
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.02, fdr_alpha_level=0.04,
                    fdr_rejected=1, fdr_test_index=2)
    result = repo.fdr_stream_state()
    assert result is not None
    assert result.t == 2
    assert result.discovery_indices == [2]


def test_fdr_stream_discovery_extraction(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.06, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=1)
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.01, fdr_alpha_level=0.04,
                    fdr_rejected=1, fdr_test_index=2)
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.05, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=3)
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.001, fdr_alpha_level=0.02,
                    fdr_rejected=1, fdr_test_index=4)
    result = repo.fdr_stream_state()
    assert result is not None
    assert result.t == 4
    assert result.discovery_indices == [2, 4]


def test_fdr_stream_null_p_value_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=None, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=1)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_null_alpha_level_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=None,
                    fdr_rejected=0, fdr_test_index=1)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_null_rejected_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=None, fdr_test_index=1)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_bad_rejected_value_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=2, fdr_test_index=1)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_zero_test_index_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=0)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_negative_test_index_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=-1)
    assert repo.fdr_stream_state() is None


def test_fdr_stream_non_contiguous_indices_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.06, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=1)
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.02, fdr_alpha_level=0.04,
                    fdr_rejected=1, fdr_test_index=3)  # skips 2 → gap → corrupted
    assert repo.fdr_stream_state() is None


def test_fdr_stream_null_test_index_fails_closed(repo):
    sid = repo.add("s").id
    _insert_fdr_row(repo, sid, fdr_binding=1, fdr_p_value=0.03, fdr_alpha_level=0.04,
                    fdr_rejected=0, fdr_test_index=None)
    assert repo.fdr_stream_state() is None


# ---------------------------------------------------------------------------
# Task 4 (#220 Phase 2): record_gate_with_fdr_and_maybe_promote — atomic FDR write
# ---------------------------------------------------------------------------


def _make_gate_row(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "n_funnel": 9, "own_lifetime_combos": 9, "windowed_total_combos": 9,
        "funnel_window_days": 90, "breadth_provenance": "measured",
        "pit_ok": True, "pit_override": False, "holdout_n_bars": 80,
        "min_holdout_observations": 63, "code_hash": "c0", "config_hash": "cfg0",
        "dependency_hash": "dep0", "data_source": "SyntheticProvider",
        "snapshot_id": None, "period_start": "2022-01-01", "period_end": "2023-12-31",
        "holdout_frac": 0.2, "decision_json": json.dumps({"passed": passed}),
    }


# #339 — the funnel snapshot for a synthetic empty-search_trials, no-family DB: every field the
# in-lock CAS re-reads (windowed combos, family, the append-only fingerprint) evaluates to its
# zero/None state here, so this constant satisfies the CAS for the FDR-mechanism unit tests below.
# dsr_binding=False skips the own-combo/variance re-reads (those are exercised by the dedicated
# funnel-drift tests, which build snapshots that match a seeded DB).
_EMPTY_FUNNEL = FunnelSnapshot(
    strategy_name="s", funnel_window_days=90, dsr_binding=False,
    own_lifetime_combos=0, windowed_total_combos=0, family_id=None,
    family_lifetime_effective=0, dsr_trial_var_ann=None, funnel_floor_var_ann=None,
    funnel_floor_n_strategies=0, funnel_floor_n_total_rows=0,
    search_trials_count=0, search_trials_max_id=0,
)


def _at_backtested(repo, name: str = "s"):
    repo.add(name)
    return transition_strategy(repo, name, Stage.BACKTESTED, Actor.AGENT, "setup")


def _level_accept(t: int, taus: list[int]) -> float:
    return 1.0  # p ≤ 1.0 always → always a discovery


def _level_reject(t: int, taus: list[int]) -> float:
    return 0.0  # p ≤ 0.0 never → never a discovery


def test_fdr_gate_non_binding_passes_through_on_provisional_pass(repo):
    rec = _at_backtested(repo)
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=None,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert isinstance(outcome, FdrGateOutcome)
    assert outcome.final_passed is True
    assert outcome.fdr_binding is False
    assert outcome.fdr_test_index is None
    assert outcome.fdr_p_value is None
    assert outcome.fdr_alpha_level is None
    assert outcome.fdr_rejected is None
    # Non-binding: stream must remain empty.
    assert repo.fdr_stream_state() == FdrStreamState(t=0, discovery_indices=[])


def test_fdr_gate_non_binding_fails_on_provisional_fail(repo):
    rec = _at_backtested(repo)
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=False), p_value=None,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert outcome.final_passed is False
    assert outcome.updated_rec is None


def test_fdr_gate_refuses_promote_from_drifted_source_stage(repo):
    # #246: if the source stage drifted off BACKTESTED before the post-walk_forward re-read (e.g. a
    # concurrent transition to terminal RETIRED), the stage CAS would otherwise pin WHERE stage=
    # RETIRED and apply a forbidden RETIRED->CANDIDATE edge. The source-stage invariant is now
    # re-asserted INSIDE the locked critical section, so the gate refuses and the whole tx rolls
    # back (no promotion, no gate row). Before the fix this resurrected a terminal strategy.
    repo.add("s")
    transition_strategy(repo, "s", Stage.BACKTESTED, Actor.AGENT, "setup")
    drifted = transition_strategy(repo, "s", Stage.RETIRED, Actor.HUMAN, "retired")
    assert drifted.stage is Stage.RETIRED
    with pytest.raises(TransitionError, match="backtested"):
        repo.record_gate_with_fdr_and_maybe_promote(
            drifted, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=None,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert repo.get("s").stage is Stage.RETIRED            # not resurrected to candidate
    assert repo.connection.execute(                         # gate row rolled back with the tx
        "SELECT COUNT(*) FROM gate_evaluations").fetchone()[0] == 0


def test_fdr_gate_drift_rollback_does_not_advance_fdr_stream(repo):
    # #246 (Codex GATE-2): on the binding-FDR path, the drift assertion must roll back the inserted
    # binding gate row AND leave the LORD++ alpha-wealth stream position untouched — else a refused
    # promote would silently spend FDR alpha-wealth (corrupting the ledger) on a non-event.
    repo.add("s")
    transition_strategy(repo, "s", Stage.BACKTESTED, Actor.AGENT, "setup")
    drifted = transition_strategy(repo, "s", Stage.RETIRED, Actor.HUMAN, "retired")
    assert repo.fdr_stream_state() == FdrStreamState(t=0, discovery_indices=[])
    with pytest.raises(TransitionError, match="backtested"):
        repo.record_gate_with_fdr_and_maybe_promote(
            drifted, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.01,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")  # binding + would-pass
    assert repo.fdr_stream_state() == FdrStreamState(t=0, discovery_indices=[])  # stream untouched
    assert repo.connection.execute(
        "SELECT COUNT(*) FROM gate_evaluations").fetchone()[0] == 0


def test_fdr_gate_promotes_from_backtested_source(repo):
    # Guard against over-restriction: a genuine BACKTESTED source still promotes to candidate.
    rec = _at_backtested(repo)
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=None,
        level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert outcome.final_passed is True
    assert outcome.updated_rec is not None and outcome.updated_rec.stage is Stage.CANDIDATE
    assert repo.get("s").stage is Stage.CANDIDATE


def test_fdr_gate_binding_accept_promotes_when_provisional_passes(repo):
    rec = _at_backtested(repo)
    # p=0.03 ≤ level_accept(1, []) = 1.0 → FDR accepts → final_passed=True
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert outcome.final_passed is True
    assert outcome.fdr_binding is True
    assert outcome.fdr_rejected is True
    assert outcome.fdr_test_index == 1
    assert outcome.updated_rec is not None
    assert outcome.updated_rec.stage is Stage.CANDIDATE


def test_fdr_gate_binding_reject_blocks_promotion(repo):
    rec = _at_backtested(repo)
    # p=0.03 > level_reject(1, []) = 0.0 → FDR rejects → final_passed=False
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_reject, actor=Actor.AGENT)
    assert outcome.final_passed is False
    assert outcome.fdr_binding is True
    assert outcome.fdr_rejected is False
    assert outcome.fdr_test_index == 1
    assert outcome.updated_rec is None
    assert repo.get("s").stage is Stage.BACKTESTED


def test_fdr_gate_provisional_fail_skips_fdr_promotion(repo):
    rec = _at_backtested(repo)
    # provisional=False → final_passed=False regardless of FDR
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=False), p_value=0.03,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert outcome.final_passed is False
    assert outcome.updated_rec is None


def test_fdr_gate_db_passed_column_reflects_final_not_provisional(repo):
    rec = _at_backtested(repo)
    # provisional=True but FDR rejects → DB must store passed=0
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_reject, actor=Actor.AGENT)
    assert outcome.final_passed is False
    row = repo._conn.execute(
        "SELECT passed FROM gate_evaluations WHERE id=?", (outcome.gate_id,)
    ).fetchone()
    assert row["passed"] == 0  # final_passed, not provisional True


def test_fdr_gate_db_passed_column_true_on_accept(repo):
    rec = _at_backtested(repo)
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert outcome.final_passed is True
    row = repo._conn.execute(
        "SELECT passed FROM gate_evaluations WHERE id=?", (outcome.gate_id,)
    ).fetchone()
    assert row["passed"] == 1


def test_fdr_gate_decision_json_contains_fdr_evidence_check(repo):
    """decision_json stored in DB must include fdr_evidence in its checks list."""
    rec = _at_backtested(repo)
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_accept, actor=Actor.AGENT)
    row = repo._conn.execute(
        "SELECT decision_json FROM gate_evaluations WHERE id=?", (outcome.gate_id,)
    ).fetchone()
    import json
    stored = json.loads(row["decision_json"])
    fdr_checks = [c for c in stored.get("checks", []) if c.get("name") == "fdr_evidence"]
    assert len(fdr_checks) == 1
    check = fdr_checks[0]
    assert check["value"] == pytest.approx(0.03)
    assert check["op"] == "<="
    assert check["passed"] is True


def test_fdr_gate_stream_grows_for_binding_rows(repo):
    rec = _at_backtested(repo)
    repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_reject, actor=Actor.AGENT)
    stream = repo.fdr_stream_state()
    assert stream is not None
    assert stream.t == 1
    assert stream.discovery_indices == []


def test_fdr_gate_discovery_increments_test_index_and_replenishes(repo):
    rec = _at_backtested(repo)
    # First call: binding accept → discovery at t=1
    o1 = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=_level_accept, actor=Actor.AGENT)
    assert o1.fdr_test_index == 1
    assert o1.fdr_rejected is True
    # Second call with new strategy
    rec2 = _at_backtested(repo, "s2")
    # Use a level_fn that checks taus: if [1] present, it returns 0.5 (generous)
    recorded_calls: list[tuple[int, list[int]]] = []
    def level_fn_probe(t: int, taus: list[int]) -> float:
        recorded_calls.append((t, list(taus)))
        return 0.5
    o2 = repo.record_gate_with_fdr_and_maybe_promote(
        rec2, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
        level_fn=level_fn_probe, actor=Actor.AGENT)
    assert o2.fdr_test_index == 2
    # level_fn received t=2 and taus=[1] (the first discovery)
    assert len(recorded_calls) == 1
    assert recorded_calls[0] == (2, [1])


def test_fdr_gate_top_level_only_guard(repo):
    rec = _at_backtested(repo)
    # BEGIN IMMEDIATE sets in_transaction=True explicitly.
    repo._conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError, match="top level"):
            repo.record_gate_with_fdr_and_maybe_promote(
                rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(), p_value=0.03,
                level_fn=_level_accept, actor=Actor.AGENT)
    finally:
        repo._conn.rollback()


def test_fdr_gate_rollback_on_stage_cas_failure(repo, tmp_path):
    _at_backtested(repo)
    # Force a stage change on the same connection before the CAS fires, making stale invalid.
    stale = repo.get("s")  # record current state
    # manually move strategy to RETIRED so the CAS in record_gate_with_fdr fails
    repo._conn.execute(
        "UPDATE strategies SET stage='retired' WHERE name='s'"
    )
    repo._conn.commit()
    with pytest.raises(TransitionError):
        repo.record_gate_with_fdr_and_maybe_promote(
            stale, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True), p_value=0.03,
            level_fn=_level_accept, actor=Actor.AGENT)
    # FDR stream must be empty — the row was rolled back
    assert repo.fdr_stream_state() == FdrStreamState(t=0, discovery_indices=[])
    # No gate row was committed
    n = repo._conn.execute("SELECT COUNT(*) FROM gate_evaluations").fetchone()[0]
    assert n == 0


def test_fdr_gate_concurrent_distinct_t_values(tmp_path):
    """Two threads calling record_gate_with_fdr_and_maybe_promote concurrently against the same
    DB file must receive distinct fdr_test_index values (BEGIN IMMEDIATE serializes them)."""
    db = tmp_path / "c.db"

    def _setup() -> tuple[SqliteStrategyRepository, object]:
        c = connect(db)
        migrate(c)
        r = SqliteStrategyRepository(c)
        return r, None

    # Seed on main thread
    seed_conn = connect(db)
    migrate(seed_conn)
    seed_repo = SqliteStrategyRepository(seed_conn)
    seed_repo.add("s1")
    seed_repo.add("s2")
    transition_strategy(seed_repo, "s1", Stage.BACKTESTED, Actor.AGENT, "setup")
    transition_strategy(seed_repo, "s2", Stage.BACKTESTED, Actor.AGENT, "setup")
    seed_conn.close()

    outcomes: list[FdrGateOutcome | BaseException] = []
    lock = threading.Lock()

    def run_thread(name: str) -> None:
        c = connect(db)
        r = SqliteStrategyRepository(c)
        rec = r.get(name)
        try:
            outcome = r.record_gate_with_fdr_and_maybe_promote(
                rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=False), p_value=0.03,
                level_fn=_level_reject, actor=Actor.AGENT)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                outcomes.append(exc)
        else:
            with lock:
                outcomes.append(outcome)

    t1 = threading.Thread(target=run_thread, args=("s1",))
    t2 = threading.Thread(target=run_thread, args=("s2",))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert len(outcomes) == 2, f"expected 2 outcomes, got: {outcomes}"
    for o in outcomes:
        if isinstance(o, BaseException):
            raise o
    t_values = {o.fdr_test_index for o in outcomes if isinstance(o, FdrGateOutcome)}
    assert t_values == {1, 2}, f"expected distinct t=1,2 but got {t_values}"


# ---------------------------------------------------------------------------
# #339: funnel-snapshot CAS — a promotion decision computed against a funnel snapshot that drifts
# before commit must fail closed (never serialize a mixed-snapshot outcome).
# ---------------------------------------------------------------------------


def _live_funnel(repo, name: str = "s", *, dsr_binding: bool = False) -> FunnelSnapshot:
    """Build a FunnelSnapshot matching the repo's CURRENT funnel state (what run_gate captures)."""
    floor = repo.funnel_trial_sharpe_var(90)
    count, mx = repo.search_trials_fingerprint()
    fam = repo.strategy_family(name)
    return FunnelSnapshot(
        strategy_name=name, funnel_window_days=90, dsr_binding=dsr_binding,
        own_lifetime_combos=repo.total_search_combos(name),
        windowed_total_combos=repo.windowed_search_combos(90),
        family_id=fam,
        family_lifetime_effective=(repo.family_lifetime_combos(fam) if fam is not None else 0),
        dsr_trial_var_ann=(repo.pooled_trial_sharpe_var(name) if dsr_binding else None),
        funnel_floor_var_ann=(floor.var_ann if dsr_binding else None),
        funnel_floor_n_strategies=(floor.n_strategies if dsr_binding else 0),
        funnel_floor_n_total_rows=(floor.n_total_rows if dsr_binding else 0),
        search_trials_count=count, search_trials_max_id=mx,
    )


def test_funnel_cas_passes_when_snapshot_matches(repo):
    # A live snapshot that matches the committed funnel state promotes normally — the CAS is inert.
    _at_backtested(repo)
    repo.record_search_trial("s", 4, "{}")
    rec = repo.get("s")  # unchanged stage, refreshed record
    outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_live_funnel(repo, "s"), gate_row=_make_gate_row(passed=True), p_value=None,
        level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert outcome.final_passed is True


def test_funnel_cas_aborts_on_search_trials_insert(repo):
    # Snapshot captured, THEN a concurrent search_trials insert bumps the fingerprint (and windowed
    # combos + own combos): the commit must abort with FunnelDriftError, rolling back the whole tx.
    rec = _at_backtested(repo)
    stale = _live_funnel(repo, "s")
    repo.record_search_trial("other", 7, "{}")  # concurrent funnel change after the snapshot
    with pytest.raises(FunnelDriftError, match="search_trials"):
        repo.record_gate_with_fdr_and_maybe_promote(
            rec, funnel=stale, gate_row=_make_gate_row(passed=True), p_value=0.01,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    # Fail-closed: no gate row committed, stream untouched, strategy not promoted.
    assert repo._conn.execute("SELECT COUNT(*) FROM gate_evaluations").fetchone()[0] == 0
    assert repo.fdr_stream_state() == FdrStreamState(t=0, discovery_indices=[])
    assert repo.get("s").stage is Stage.BACKTESTED


def test_funnel_cas_aborts_on_windowed_breadth_drift(repo):
    # A snapshot whose windowed_total_combos is stale (lower than live) must abort even if the
    # fingerprint were to match — windowed breadth feeds n_funnel (the deflated bar) on every path.
    rec = _at_backtested(repo)
    snap = _live_funnel(repo, "s")._replace(windowed_total_combos=999999)
    with pytest.raises(FunnelDriftError, match="windowed_search_combos"):
        repo.record_gate_with_fdr_and_maybe_promote(
            rec, funnel=snap, gate_row=_make_gate_row(passed=True), p_value=None,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert repo.get("s").stage is Stage.BACKTESTED


def test_funnel_cas_aborts_on_variance_drift_when_binding(repo):
    # On the FDR-binding (measured) path the DSR variances feed p_value; a stale funnel-floor
    # variance must abort. Seed >=2 strategies so the funnel floor is non-None, then present a
    # snapshot with a perturbed floor variance.
    _at_backtested(repo)
    # MIN_FUNNEL_FLOOR_STRATEGIES=5 strategies with finite trial stats -> a non-None funnel floor.
    for nm in ("s", "t", "u", "v", "w"):
        repo.record_search_trial(
            nm, 3, "{}", trial_sharpe_count=3, trial_sharpe_mean=0.1, trial_sharpe_var_ann=0.2)
    rec = repo.get("s")
    live = _live_funnel(repo, "s", dsr_binding=True)
    assert live.funnel_floor_var_ann is not None  # 5 seeded strategies -> non-None floor
    # Perturb the stored floor variance so it no longer matches the live recompute at commit.
    drifted = live._replace(funnel_floor_var_ann=live.funnel_floor_var_ann + 1.0)
    with pytest.raises(FunnelDriftError, match="funnel_trial_sharpe_var"):
        repo.record_gate_with_fdr_and_maybe_promote(
            rec, funnel=drifted, gate_row=_make_gate_row(passed=True), p_value=0.01,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert repo.get("s").stage is Stage.BACKTESTED


def test_funnel_cas_aborts_on_family_drift(repo):
    # A stale family_id (decision computed before a concurrent family assignment) must abort.
    rec = _at_backtested(repo)
    snap = _live_funnel(repo, "s")._replace(family_id=4242)  # a family the strategy is not in
    with pytest.raises(FunnelDriftError, match="strategy_family"):
        repo.record_gate_with_fdr_and_maybe_promote(
            rec, funnel=snap, gate_row=_make_gate_row(passed=True), p_value=None,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    assert repo.get("s").stage is Stage.BACKTESTED


def test_funnel_cas_concurrent_stale_promote_aborts_order_independent(tmp_path):
    """Multi-connection proof of order-independence (#339): a promote whose funnel snapshot was
    computed BEFORE a concurrent breadth change commits must NOT serialize a stale-breadth outcome.
    It aborts with FunnelDriftError regardless of which connection commits first — the committed
    outcome is a pure function of the funnel snapshot, never of the wall-clock interleaving."""
    db = tmp_path / "f.db"
    seed = connect(db)
    migrate(seed)
    sr = SqliteStrategyRepository(seed)
    sr.add("s")
    transition_strategy(sr, "s", Stage.BACKTESTED, Actor.AGENT, "setup")
    seed.close()

    # Connection A captures a funnel snapshot, then connection B commits a breadth change.
    ca = connect(db)
    ra = SqliteStrategyRepository(ca)
    rec = ra.get("s")
    stale = _live_funnel(ra, "s")

    cb = connect(db)
    rb = SqliteStrategyRepository(cb)
    rb.record_search_trial("s", 5, "{}")  # funnel drift lands before A's commit

    with pytest.raises(FunnelDriftError):
        ra.record_gate_with_fdr_and_maybe_promote(
            rec, funnel=stale, gate_row=_make_gate_row(passed=True), p_value=0.01,
            level_fn=_level_accept, actor=Actor.AGENT, reason="promote")
    # A committed nothing; B's search_trials row is intact; s is still BACKTESTED (not promoted).
    assert ra.get("s").stage is Stage.BACKTESTED
    assert ra._conn.execute("SELECT COUNT(*) FROM gate_evaluations").fetchone()[0] == 0
    ca.close()
    cb.close()


def test_fdr_gate_agent_pass_is_born_consumed(repo):
    """Agent passing rows must be born consumed=1 so a back-step cannot replay the token."""
    rec = _at_backtested(repo)
    repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True),
        p_value=0.01, level_fn=_level_accept, actor=Actor.AGENT,
    )
    row = repo._conn.execute(
        "SELECT consumed FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["consumed"] == 1


def test_fdr_gate_agent_fail_and_human_pass_not_consumed(repo):
    """Non-passing agent rows and human rows must have consumed=0."""
    # Agent row that fails FDR (provisional pass but FDR rejects) → consumed=0
    rec = _at_backtested(repo)
    repo.record_gate_with_fdr_and_maybe_promote(
        rec, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True),
        p_value=0.01, level_fn=_level_reject, actor=Actor.AGENT,
    )
    row_agent_fail = repo._conn.execute(
        "SELECT consumed FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row_agent_fail["consumed"] == 0

    # Human passing row → also consumed=0 (human rows are never consumable tokens)
    rec2 = _at_backtested(repo, "s2")
    repo.record_gate_with_fdr_and_maybe_promote(
        rec2, funnel=_EMPTY_FUNNEL, gate_row=_make_gate_row(passed=True),
        p_value=0.01, level_fn=_level_accept, actor=Actor.HUMAN,
    )
    row_human = repo._conn.execute(
        "SELECT consumed FROM gate_evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row_human["consumed"] == 0
