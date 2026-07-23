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


# ---------------------------------------------------------------------------
# #524 (R9): family lifetime breadth PRIOR (families.seeded_prior_combos). A lifetime-only prior
# that flows through family_lifetime_combos (own + ancestor trials + own + ancestor seeds) but is
# NEVER counted in windowed/funnel-wide sums. Default 0 keeps all pre-#524 numbers unchanged.
# Under R9 the seed is written ONLY by the atomic pass-time mint (no create_family kwarg); tests
# set it via a direct families INSERT (allowed by the append-only triggers, which block only
# UPDATE/DELETE), mirroring what _mint_agent_novel_family persists at commit.
# ---------------------------------------------------------------------------

def _seed_family(repo: SqliteStrategyRepository, name: str, seed: int) -> int:
    """Insert a family row carrying ``seeded_prior_combos = seed`` + its family_created event,
    the way the atomic agent-NOVEL mint does. INSERT is trigger-permitted (append-only forbids
    only UPDATE/DELETE)."""
    now = datetime.now(UTC).isoformat()
    cur = repo._conn.execute(
        "INSERT INTO families(name, created_at, created_by_actor, created_by_strategy,"
        " seeded_prior_combos) VALUES (?,?,?,?,?)",
        (name, now, "agent", None, seed),
    )
    fid = int(cur.lastrowid)
    repo._conn.execute(
        "INSERT INTO family_events(event_type, family_id, actor, created_at)"
        " VALUES ('family_created', ?, 'agent', ?)",
        (fid, now),
    )
    repo._conn.commit()
    return fid


def test_seed_prior_counts_in_lifetime_with_no_trials() -> None:
    """A family seeded with 500 and no trials has family_lifetime_combos == 500."""
    repo = _make_repo()
    fid = _seed_family(repo, "seeded", 500)
    assert repo.family_lifetime_combos(fid) == 500


def test_seed_prior_adds_to_own_trials() -> None:
    """seed 500 + a 30-combo trial -> lifetime 530 (prior and trials both count)."""
    repo = _make_repo()
    fid = _seed_family(repo, "seeded_trials", 500)
    _assign(repo, "strat_seeded", fid)
    repo.record_search_trial("strat_seeded", n_combos=30, grid_json="{}")
    assert repo.family_lifetime_combos(fid) == 530


def test_seed_prior_is_lifetime_only_not_windowed() -> None:
    """The seed prior is a LIFETIME-only prior: it must NOT leak into windowed_family_combos."""
    repo = _make_repo()
    fid = _seed_family(repo, "seeded_win", 500)
    _assign(repo, "strat_seeded_win", fid)
    repo.record_search_trial("strat_seeded_win", n_combos=7, grid_json="{}")
    # windowed sees only the recent trial (seed excluded); lifetime includes the seed.
    assert repo.windowed_family_combos(fid, window_days=90) == 7
    assert repo.family_lifetime_combos(fid) == 507


def test_child_inherits_parent_seed_via_ancestry() -> None:
    """A child of a family seeded with 500 inherits the parent's seed via ancestry."""
    repo = _make_repo()
    parent = _seed_family(repo, "seed_parent", 500)
    child = repo.create_family("seed_child", actor="human")
    repo.add_parent_edge(child, parent)
    # No trials anywhere: the child's lifetime is entirely the inherited parent seed.
    assert repo.family_lifetime_combos(child) == 500


def test_seed_prior_deduped_across_diamond_ancestry() -> None:
    """A seed on a diamond apex is counted ONCE through both paths (set-dedup of family ids)."""
    repo = _make_repo()
    a = _seed_family(repo, "apex", 400)
    b = repo.create_family("Bd", actor="human")
    c = repo.create_family("Cd", actor="human")
    d = repo.create_family("Dd", actor="human")
    repo.add_parent_edge(b, a)
    repo.add_parent_edge(c, a)
    repo.add_parent_edge(d, b)
    repo.add_parent_edge(d, c)
    assert repo.family_lifetime_combos(d) == 400  # not 800


def test_default_create_family_has_zero_seed_unchanged_accounting() -> None:
    """create_family (the public helper) yields seed 0 and leaves lifetime accounting untouched
    (guards the existing PARENTAGE inheritance numbers against regression)."""
    repo = _make_repo()
    parent = repo.create_family("plain_parent", actor="human")
    child = repo.create_family("plain_child", actor="human")
    repo.add_parent_edge(child, parent)

    seed_row = repo._conn.execute(
        "SELECT seeded_prior_combos FROM families WHERE id=?", (parent,)
    ).fetchone()
    assert seed_row[0] == 0

    _assign(repo, "strat_pp", parent)
    repo.record_search_trial("strat_pp", n_combos=100, grid_json="{}")
    _assign(repo, "strat_pc", child)
    repo.record_search_trial("strat_pc", n_combos=5, grid_json="{}")
    # Purely trial-driven, exactly as before #524 (100 parent + 5 child, seeds are 0).
    assert repo.family_lifetime_combos(child) == 105
