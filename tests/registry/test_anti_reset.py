"""Task 6 of #222: anti-reset property tests (verification-only).

Verifies that family_lifetime_combos (and the 3-way effective_funnel_breadth) survive the
90-day windowed_family_combos expiry — the anti-reset property that makes family-scoped
breadth permanent, not decaying.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from algua.registry import db
from algua.registry.store import SqliteStrategyRepository
from algua.research.gates import effective_funnel_breadth


def _make_repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


def _assign(
    repo: SqliteStrategyRepository, strategy_name: str, family_id: int,
) -> None:
    repo.assign_strategy_to_family(
        strategy_name, family_id, actor="agent",
        verdict="NEW", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}",
    )


def _insert_trial_at(
    repo: SqliteStrategyRepository, name: str, n_combos: int, created_at: str,
) -> None:
    repo._conn.execute(
        "INSERT INTO search_trials (strategy_name, n_combos, grid_json, created_at)"
        " VALUES (?,?,?,?)",
        (name, n_combos, "{}", created_at),
    )


def _ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def test_lifetime_inheritance_survives_window_expiry() -> None:
    """Parent's 200-day-old trial is out of the 90-day window but still in lifetime.
    effective_funnel_breadth sees it via family_lifetime_combos (anti-reset)."""
    repo = _make_repo()
    parent = repo.create_family("parent", actor="human")
    child = repo.create_family("child", actor="human")
    repo.add_parent_edge(child, parent)

    _assign(repo, "strat_parent", parent)
    _insert_trial_at(repo, "strat_parent", 500, _ago(200))

    _assign(repo, "strat_child", child)
    _insert_trial_at(repo, "strat_child", 5, _ago(5))

    windowed = repo.windowed_family_combos(child, window_days=90)
    assert windowed == 5  # parent's old trial out of window

    lifetime = repo.family_lifetime_combos(child)
    assert lifetime == 505  # lifetime includes parent's 500

    assert effective_funnel_breadth(5, 5, lifetime) == 505  # anti-reset holds


def test_two_level_lifetime_inheritance() -> None:
    """A -> B -> C; each family has 100 combos. family_lifetime_combos(C) == 300."""
    repo = _make_repo()
    a = repo.create_family("A", actor="human")
    b = repo.create_family("B", actor="human")
    c = repo.create_family("C", actor="human")
    repo.add_parent_edge(b, a)
    repo.add_parent_edge(c, b)

    _assign(repo, "strat_a", a)
    repo.record_search_trial("strat_a", n_combos=100, grid_json="{}")
    _assign(repo, "strat_b", b)
    repo.record_search_trial("strat_b", n_combos=100, grid_json="{}")
    _assign(repo, "strat_c", c)
    repo.record_search_trial("strat_c", n_combos=100, grid_json="{}")

    assert repo.family_lifetime_combos(c) == 300


def test_diamond_ancestry_no_double_count_via_effective() -> None:
    """Diamond DAG through effective_funnel_breadth: A counted once despite two paths."""
    repo = _make_repo()
    a = repo.create_family("A", actor="human")
    b = repo.create_family("B", actor="human")
    c = repo.create_family("C", actor="human")
    d = repo.create_family("D", actor="human")
    repo.add_parent_edge(b, a)
    repo.add_parent_edge(c, a)
    repo.add_parent_edge(d, b)
    repo.add_parent_edge(d, c)

    _assign(repo, "strat_a", a)
    repo.record_search_trial("strat_a", n_combos=100, grid_json="{}")

    family_lt = repo.family_lifetime_combos(d)
    assert family_lt == 100  # not 200
    assert effective_funnel_breadth(0, 0, family_lt) == 100


def test_cycle_guard_prevents_infinite_bfs() -> None:
    """Manually corrupt family_parents to create a cycle; family_lifetime_combos must not hang."""
    repo = _make_repo()
    a = repo.create_family("A", actor="human")
    b = repo.create_family("B", actor="human")
    repo.add_parent_edge(b, a)

    # Corrupt: insert A->B edge bypassing the cycle guard
    repo._conn.execute(
        "INSERT INTO family_parents (child_family_id, parent_family_id) VALUES (?, ?)",
        (a, b),
    )

    _assign(repo, "strat_a", a)
    repo.record_search_trial("strat_a", n_combos=50, grid_json="{}")
    _assign(repo, "strat_b", b)
    repo.record_search_trial("strat_b", n_combos=50, grid_json="{}")

    # Must terminate (visited-set guard), not hang
    result = repo.family_lifetime_combos(a)
    assert result >= 50  # at least own family's combos
