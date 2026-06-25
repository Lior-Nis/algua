"""TDD tests for Task 3 of #222: windowed_family_combos breadth accessor.

All tests use an in-memory SQLite connection + db.migrate() so they are fast
and independent.  Tests are written BEFORE the implementation (TDD).
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from algua.registry import db
from algua.registry.store import SqliteStrategyRepository


def _make_repo() -> SqliteStrategyRepository:
    """Create a fresh in-memory repo with the current schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


def _assign(
    repo: SqliteStrategyRepository,
    strategy_name: str,
    family_id: int,
    actor: str = "agent",
) -> None:
    repo.assign_strategy_to_family(
        strategy_name,
        family_id,
        actor,
        verdict="NEW",
        similarity_score=0.0,
        clustering_version="v1",
        clustering_config_json="{}",
        axis_json="{}",
        matched_family_id=None,
    )


def _insert_trial_at(
    repo: SqliteStrategyRepository,
    strategy_name: str,
    n_combos: int,
    created_at: str,
) -> None:
    """Insert a search_trial row with a specific created_at (bypasses record_search_trial
    which always uses _now()). Used to test windowed filtering."""
    with repo.connection:
        repo.connection.execute(
            "INSERT INTO search_trials (strategy_name, n_combos, grid_json, created_at)"
            " VALUES (?,?,?,?)",
            (strategy_name, n_combos, "{}", created_at),
        )


def _ago(days: int) -> str:
    """Return an ISO-8601 UTC timestamp that is `days` days in the past."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# windowed_family_combos — basic filtering
# ---------------------------------------------------------------------------

def test_windowed_family_combos_filters_by_cutoff() -> None:
    """A trial 100 days ago is outside window_days=90; only the recent trial counts."""
    repo = _make_repo()
    f = repo.create_family("fam_filter", actor="agent")
    _assign(repo, "strat_a", f)

    # Outside the window (100 days ago)
    _insert_trial_at(repo, "strat_a", 200, _ago(100))
    # Inside the window (today)
    _insert_trial_at(repo, "strat_a", 50, _ago(0))

    result = repo.windowed_family_combos(f, window_days=90)
    assert result == 50


def test_windowed_family_combos_empty_family() -> None:
    """Family with no assigned strategies returns 0."""
    repo = _make_repo()
    f = repo.create_family("empty_fam", actor="agent")
    assert repo.windowed_family_combos(f, window_days=90) == 0


def test_windowed_family_combos_no_trials_in_window() -> None:
    """Family has a strategy but all trials are outside the window; result is 0."""
    repo = _make_repo()
    f = repo.create_family("fam_old", actor="agent")
    _assign(repo, "strat_b", f)
    _insert_trial_at(repo, "strat_b", 300, _ago(200))

    assert repo.windowed_family_combos(f, window_days=90) == 0


# ---------------------------------------------------------------------------
# windowed_family_combos — ancestor inclusion
# ---------------------------------------------------------------------------

def test_windowed_family_combos_ancestor_included() -> None:
    """Parent family P has a recent trial; child family C has no trials.
    windowed_family_combos(C, 90) includes the parent's recent trial.
    """
    repo = _make_repo()
    parent = repo.create_family("parent", actor="agent")
    child = repo.create_family("child", actor="agent")
    repo.add_parent_edge(child, parent)

    _assign(repo, "strat_p", parent)
    _insert_trial_at(repo, "strat_p", 75, _ago(10))  # inside window

    assert repo.windowed_family_combos(child, window_days=90) == 75


def test_windowed_family_combos_ancestor_old_trial_excluded() -> None:
    """Parent family P has a 200-day-old trial; child C has a recent trial.
    windowed_family_combos(C, 90) == child's recent trial only.
    """
    repo = _make_repo()
    parent = repo.create_family("parent_old", actor="agent")
    child = repo.create_family("child_new", actor="agent")
    repo.add_parent_edge(child, parent)

    _assign(repo, "strat_p", parent)
    _insert_trial_at(repo, "strat_p", 500, _ago(200))  # outside window

    _assign(repo, "strat_c", child)
    _insert_trial_at(repo, "strat_c", 40, _ago(5))  # inside window

    result = repo.windowed_family_combos(child, window_days=90)
    assert result == 40


# ---------------------------------------------------------------------------
# windowed vs lifetime comparison
# ---------------------------------------------------------------------------

def test_family_lifetime_vs_windowed() -> None:
    """lifetime includes an old trial; windowed does not.
    family_lifetime_combos(F) > windowed_family_combos(F, 90).
    """
    repo = _make_repo()
    f = repo.create_family("fam_compare", actor="agent")
    _assign(repo, "strat_q", f)

    _insert_trial_at(repo, "strat_q", 400, _ago(200))  # outside window but in lifetime
    _insert_trial_at(repo, "strat_q", 100, _ago(30))   # inside window

    lifetime = repo.family_lifetime_combos(f)
    windowed = repo.windowed_family_combos(f, window_days=90)

    assert lifetime == 500   # both trials
    assert windowed == 100   # only the recent one
    assert lifetime > windowed


# ---------------------------------------------------------------------------
# windowed_family_combos — double-count guard
# ---------------------------------------------------------------------------

def test_windowed_no_double_count_on_move() -> None:
    """Strategy moved between two ancestor families; windowed count is not doubled.

    Setup: child C with parents A and B.
    strat_x starts in A (recent 100-combo trial), then gets reassigned to B.
    strat_x now has TWO family_members rows (one in A with removed_at, one active in B).
    Both A and B are in the ancestor set of C.  Must count 100, not 200.
    """
    repo = _make_repo()
    fam_a = repo.create_family("fam_a", actor="human")
    fam_b = repo.create_family("fam_b", actor="human")
    fam_c = repo.create_family("fam_c", actor="human")
    repo.add_parent_edge(fam_c, fam_a)
    repo.add_parent_edge(fam_c, fam_b)

    _assign(repo, "strat_x", fam_a, actor="human")
    _insert_trial_at(repo, "strat_x", 100, _ago(5))  # inside window

    # Reassign — creates second family_members row for strat_x
    _assign(repo, "strat_x", fam_b, actor="human")

    result = repo.windowed_family_combos(fam_c, window_days=90)
    assert result == 100  # not 200
