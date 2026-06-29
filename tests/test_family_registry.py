"""TDD tests for Task 1 of #222: family registry + parentage DAG (schema 25→26).

All tests use an in-memory SQLite connection + db.migrate() so they are fast and
independent. The tests are written BEFORE the implementation (TDD).
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from algua.registry import db
from algua.registry.store import SqliteStrategyRepository


def _make_repo() -> SqliteStrategyRepository:
    """Create a fresh in-memory repo with the current schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign(
    repo: SqliteStrategyRepository,
    strategy_name: str,
    family_id: int,
    actor: str = "agent",
    verdict: str = "NEW",
    similarity_score: float = 0.0,
) -> None:
    repo.assign_strategy_to_family(
        strategy_name,
        family_id,
        actor,
        verdict=verdict,
        similarity_score=similarity_score,
        clustering_version="v1",
        clustering_config_json="{}",
        axis_json="{}",
        matched_family_id=None,
    )


# ---------------------------------------------------------------------------
# Schema / migration
# ---------------------------------------------------------------------------

def test_schema_version_is_31() -> None:
    repo = _make_repo()
    version = repo.connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == 31


def test_schema_migration_idempotent() -> None:
    """Running migrate() twice must not raise and must leave SCHEMA_VERSION == 31."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    db.migrate(conn)  # second call must be a no-op
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 31


def test_family_tables_exist() -> None:
    repo = _make_repo()
    conn = repo.connection
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"families", "family_members", "family_parents", "family_events"} <= tables


# ---------------------------------------------------------------------------
# create_family
# ---------------------------------------------------------------------------

def test_create_family_roundtrips() -> None:
    repo = _make_repo()
    fid = repo.create_family("momentum", actor="agent")
    assert isinstance(fid, int)
    # name unique constraint
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_family("momentum", actor="agent")
    # family_events row written
    rows = repo.connection.execute(
        "SELECT event_type, family_id FROM family_events WHERE event_type='family_created'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["family_id"] == fid


def test_create_family_with_created_by_strategy() -> None:
    repo = _make_repo()
    fid = repo.create_family("trend", actor="agent", created_by_strategy="strat_x")
    row = repo.connection.execute(
        "SELECT created_by_strategy FROM families WHERE id=?", (fid,)
    ).fetchone()
    assert row["created_by_strategy"] == "strat_x"


# ---------------------------------------------------------------------------
# assign_strategy_to_family
# ---------------------------------------------------------------------------

def test_assign_strategy_updates_active_and_keeps_old_row() -> None:
    repo = _make_repo()
    f1 = repo.create_family("family_a", actor="agent")
    f2 = repo.create_family("family_b", actor="agent")

    _assign(repo, "strat_a", f1)
    _assign(repo, "strat_a", f2)

    rows = repo.connection.execute(
        "SELECT family_id, removed_at FROM family_members WHERE strategy_name='strat_a'"
        " ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    # First row (old) must have removed_at set
    assert rows[0]["family_id"] == f1
    assert rows[0]["removed_at"] is not None
    # Second row (current) must have removed_at NULL
    assert rows[1]["family_id"] == f2
    assert rows[1]["removed_at"] is None


def test_strategy_family_returns_active_family() -> None:
    repo = _make_repo()
    f1 = repo.create_family("fam1", actor="agent")
    f2 = repo.create_family("fam2", actor="agent")

    assert repo.strategy_family("strat_b") is None

    _assign(repo, "strat_b", f1)
    assert repo.strategy_family("strat_b") == f1

    _assign(repo, "strat_b", f2)
    assert repo.strategy_family("strat_b") == f2


def test_assign_strategy_merge_verdict_event_type() -> None:
    repo = _make_repo()
    f1 = repo.create_family("fam_merge", actor="agent")
    repo.assign_strategy_to_family(
        "strat_m", f1, "agent",
        verdict="MERGE",
        similarity_score=0.9,
        clustering_version="v1",
        clustering_config_json="{}",
        axis_json="{}",
        matched_family_id=None,
    )
    row = repo.connection.execute(
        "SELECT event_type FROM family_events"
        " WHERE strategy_name='strat_m' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row["event_type"] == "strategy_merged"


# ---------------------------------------------------------------------------
# family_ancestry
# ---------------------------------------------------------------------------

def test_family_ancestry_empty() -> None:
    repo = _make_repo()
    fid = repo.create_family("lonely", actor="agent")
    assert repo.family_ancestry(fid) == []


def test_family_ancestry_linear() -> None:
    """A → B → C: ancestry(C) contains both B and A."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    c = repo.create_family("C", actor="agent")
    repo.add_parent_edge(b, a)
    repo.add_parent_edge(c, b)
    ancestors = repo.family_ancestry(c)
    assert set(ancestors) == {a, b}


def test_family_ancestry_multi_parent() -> None:
    """C has parents A and B; ancestry(C) contains both."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    c = repo.create_family("C", actor="agent")
    repo.add_parent_edge(c, a)
    repo.add_parent_edge(c, b)
    ancestors = repo.family_ancestry(c)
    assert set(ancestors) == {a, b}


def test_family_ancestry_visited_set_no_double_count() -> None:
    """Diamond: A→B→D, A→C→D; ancestry(A) has B, C, D each once."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    c = repo.create_family("C", actor="agent")
    d = repo.create_family("D", actor="agent")
    repo.add_parent_edge(b, d)
    repo.add_parent_edge(c, d)
    repo.add_parent_edge(a, b)
    repo.add_parent_edge(a, c)
    ancestors = repo.family_ancestry(a)
    assert ancestors.count(d) == 1
    assert set(ancestors) == {b, c, d}


# ---------------------------------------------------------------------------
# add_parent_edge
# ---------------------------------------------------------------------------

def test_add_parent_edge_cycle_rejected() -> None:
    """A→B; then B→A must raise ValueError (cycle)."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    repo.add_parent_edge(a, b)
    with pytest.raises(ValueError, match="cycle"):
        repo.add_parent_edge(b, a)


def test_add_parent_edge_self_cycle_rejected() -> None:
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    with pytest.raises(ValueError, match="cycle"):
        repo.add_parent_edge(a, a)


def test_add_parent_edge_unique_constraint() -> None:
    """add_parent_edge(A, B) twice; second raises (UNIQUE index)."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    repo.add_parent_edge(a, b)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_parent_edge(a, b)


def test_add_parent_edge_writes_event_row() -> None:
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    repo.add_parent_edge(a, b)
    rows = repo.connection.execute(
        "SELECT event_type FROM family_events WHERE event_type='parent_edge_added'"
    ).fetchall()
    assert len(rows) == 1


def test_add_parent_edge_atomic_race() -> None:
    """Two threads: one adds A→B, one adds B→A; exactly one raises ValueError.

    With a shared connection the GIL serializes Python execution, so one thread
    completes the full BEGIN IMMEDIATE…COMMIT before the other thread's BFS runs.
    The second thread then detects the cycle and raises ValueError.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    db.migrate(conn)
    repo = SqliteStrategyRepository(conn)

    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")

    errors: list[Exception] = []
    successes: list[str] = []
    barrier = threading.Barrier(2)

    def add_a_b() -> None:
        barrier.wait()
        try:
            repo.add_parent_edge(a, b)
            successes.append("a->b")
        except (ValueError, Exception) as exc:  # noqa: BLE001
            errors.append(exc)

    def add_b_a() -> None:
        barrier.wait()
        try:
            repo.add_parent_edge(b, a)
            successes.append("b->a")
        except (ValueError, Exception) as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=add_a_b)
    t2 = threading.Thread(target=add_b_a)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one must succeed, exactly one must fail.
    assert len(successes) == 1
    assert len(errors) == 1


def test_add_parent_edge_must_be_top_level() -> None:
    """Calling add_parent_edge inside an open transaction must raise RuntimeError."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    repo.connection.execute("BEGIN")
    with pytest.raises(RuntimeError):
        repo.add_parent_edge(a, b)
    repo.connection.rollback()


# ---------------------------------------------------------------------------
# family_lifetime_combos
# ---------------------------------------------------------------------------

def test_family_lifetime_combos_empty() -> None:
    repo = _make_repo()
    f = repo.create_family("empty_fam", actor="agent")
    assert repo.family_lifetime_combos(f) == 0


def test_family_lifetime_combos_own_only() -> None:
    repo = _make_repo()
    f = repo.create_family("fam_f", actor="agent")
    _assign(repo, "strat_a", f)
    repo.record_search_trial("strat_a", n_combos=50, grid_json="{}")
    assert repo.family_lifetime_combos(f) == 50


def test_family_lifetime_combos_includes_removed_members() -> None:
    """Strategies removed from the family (removed_at set) still count breadth."""
    repo = _make_repo()
    f1 = repo.create_family("fam1", actor="agent")
    f2 = repo.create_family("fam2", actor="agent")

    # strat_a starts in f1 (80 combos), then gets moved to f2 → removed_at set on f1 row
    _assign(repo, "strat_a", f1)
    repo.record_search_trial("strat_a", n_combos=80, grid_json="{}")
    _assign(repo, "strat_a", f2)  # triggers removed_at on f1 row

    # strat_b is still active in f1 (20 combos)
    _assign(repo, "strat_b", f1)
    repo.record_search_trial("strat_b", n_combos=20, grid_json="{}")

    # f1 breadth must include the removed strat_a (80) + active strat_b (20) = 100
    assert repo.family_lifetime_combos(f1) == 100


def test_family_lifetime_combos_with_ancestors() -> None:
    """Parent family contributes its strategies' combos to the child family."""
    repo = _make_repo()
    parent = repo.create_family("parent", actor="agent")
    child = repo.create_family("child", actor="agent")
    repo.add_parent_edge(child, parent)

    _assign(repo, "strat_p", parent)
    repo.record_search_trial("strat_p", n_combos=100, grid_json="{}")

    _assign(repo, "strat_c", child)
    repo.record_search_trial("strat_c", n_combos=30, grid_json="{}")

    assert repo.family_lifetime_combos(child) == 130


def test_family_lifetime_combos_deduped_on_diamond() -> None:
    """Diamond DAG: D→B→A, D→C→A; strat_a in A counted once, not twice."""
    repo = _make_repo()
    a = repo.create_family("A", actor="agent")
    b = repo.create_family("B", actor="agent")
    c = repo.create_family("C", actor="agent")
    d = repo.create_family("D", actor="agent")
    repo.add_parent_edge(b, a)
    repo.add_parent_edge(c, a)
    repo.add_parent_edge(d, b)
    repo.add_parent_edge(d, c)

    _assign(repo, "strat_a", a)
    repo.record_search_trial("strat_a", n_combos=100, grid_json="{}")

    # strat_a counted once — membership row in family A only
    assert repo.family_lifetime_combos(d) == 100


def test_family_lifetime_combos_no_double_count_on_strategy_move() -> None:
    """A strategy moved between two ancestor families must be counted once, not twice.

    Setup: child family C with parents A and B.
    strat_x starts in A (100 combos recorded), then is reassigned to B.
    Both A and B are ancestors of C, so strat_x now has TWO family_members rows:
    one in A (removed_at set) and one in B (active).  The old JOIN-based query would
    match strat_x twice and return 200; the DISTINCT subquery must return 100.
    """
    repo = _make_repo()
    fam_a = repo.create_family("fam_a", actor="human")
    fam_b = repo.create_family("fam_b", actor="human")
    fam_c = repo.create_family("fam_c", actor="human")
    repo.add_parent_edge(fam_c, fam_a)
    repo.add_parent_edge(fam_c, fam_b)

    # Assign strat_x to family A and record trials
    _assign(repo, "strat_x", fam_a, actor="human", verdict="MERGE", similarity_score=1.0)
    repo.record_search_trial("strat_x", n_combos=100, grid_json="{}")

    # Reassign strat_x to family B — creates second family_members row for strat_x
    _assign(repo, "strat_x", fam_b, actor="human", verdict="MERGE", similarity_score=1.0)

    # Both A and B are ancestors of C; strat_x has rows in both families.
    # Must count 100 (once), not 200 (double-counted).
    assert repo.family_lifetime_combos(fam_c) == 100


# ---------------------------------------------------------------------------
# lifetime_combos_for_families + family_names  (#228 Task 2)
# ---------------------------------------------------------------------------

def _seed_member_with_trials(
    repo: SqliteStrategyRepository,
    family_id: int,
    strategy_name: str,
    *,
    n_combos: int,
) -> None:
    """Assign strategy to family and record n_combos search trials."""
    repo.assign_strategy_to_family(
        strategy_name,
        family_id,
        "human",
        verdict="merge",
        similarity_score=0.9,
        clustering_version="v",
        clustering_config_json="{}",
        axis_json="{}",
    )
    repo.record_search_trial(strategy_name, n_combos=n_combos, grid_json="{}")


def test_lifetime_combos_for_families_dedups_shared_strategy() -> None:
    # family A and B both reference strategy "s_shared" via membership;
    # union breadth must count its trials exactly once.
    repo = _make_repo()
    fa = repo.create_family("fam_a", actor="human")
    fb = repo.create_family("fam_b", actor="human")
    _seed_member_with_trials(repo, fa, "s_a", n_combos=100)
    _seed_member_with_trials(repo, fb, "s_b", n_combos=200)
    # s_shared assigned to A then reassigned to B → appears in both (append-only)
    _seed_member_with_trials(repo, fa, "s_shared", n_combos=50)
    repo.assign_strategy_to_family(
        "s_shared", fb, "human", verdict="merge", similarity_score=0.9,
        clustering_version="v", clustering_config_json="{}", axis_json="{}")
    union = repo.lifetime_combos_for_families([fa, fb])
    assert union == 100 + 200 + 50  # s_shared counted once


def test_lifetime_combos_for_families_singleton_equals_family_lifetime_combos() -> None:
    repo = _make_repo()
    fa = repo.create_family("fam_solo", actor="human")
    _seed_member_with_trials(repo, fa, "s_a", n_combos=77)
    assert repo.lifetime_combos_for_families([fa]) == repo.family_lifetime_combos(fa)


def test_family_names_returns_id_to_name() -> None:
    repo = _make_repo()
    fa = repo.create_family("alpha", actor="human")
    names = repo.family_names()
    assert names[fa] == "alpha"
