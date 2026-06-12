from datetime import UTC, datetime, timedelta

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.db import connect, migrate
from algua.registry.repository import StrategyExists
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

def _record_forward(repo, sid, *, passed=True, actor="agent", code="c0", config="cfg0",
                    dep="dep0"):
    return repo.record_forward_gate_evaluation(
        sid, passed=passed, n_forward_observations=70, min_forward_observations=63,
        session_coverage=0.95, realized_sharpe=0.8, holdout_sharpe=1.2, degradation_factor=0.5,
        sharpe_floor=0.3, realized_vol=0.1, min_forward_vol=0.02, realized_max_drawdown=0.05,
        max_forward_drawdown=0.25, first_tick_id=1, last_tick_id=70,
        first_tick_ts="2026-01-02T00:00:00+00:00", last_tick_ts="2026-06-01T00:00:00+00:00",
        max_staleness_sessions=5, n_reconcile_failures=0, n_concurrent_forward=1,
        account_id="acct", code_hash=code, config_hash=config, dependency_hash=dep, actor=actor,
        decision_json="{}")


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
