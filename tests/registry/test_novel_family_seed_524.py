"""#524 (R9) — agent-NOVEL durable seed: schema v37, append-only triggers, the funnel-lifetime
accessor + graph fingerprint, and the mint authority bounds (rate cap + human-replenished lifetime
budget). Store-layer proof the deferred-pass-time-mint architecture is enforced by mechanism.

These tests exercise the store/repository seams directly; the atomic pass-time mint end-to-end
path (record_gate_with_fdr_and_maybe_promote) is covered by the promote CLI tests.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from algua.registry import db
from algua.registry.db import MAX_N_COMBOS
from algua.registry.repository import (
    AgentMintBudgetExhaustedError,
    AgentMintCapError,
    PendingNovelFamily,
)
from algua.registry.store import (
    AGENT_NOVEL_MINT_CAP,
    AGENT_NOVEL_MINT_LIFETIME_BUDGET,
    SqliteStrategyRepository,
)


def _make_repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


def _insert_agent_family(
    repo: SqliteStrategyRepository, name: str, *, created_at: str | None = None, seed: int = 1,
) -> int:
    """INSERT an agent-created family row directly (trigger-permitted append). Used to simulate
    prior mints without driving the whole promote path."""
    now = created_at or datetime.now(UTC).isoformat()
    cur = repo._conn.execute(
        "INSERT INTO families(name, created_at, created_by_actor, seeded_prior_combos)"
        " VALUES (?,?,?,?)",
        (name, now, "agent", seed),
    )
    repo._conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Schema v37: columns, grants table, user_version
# ---------------------------------------------------------------------------

def test_schema_version_is_37() -> None:
    repo = _make_repo()
    assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == 37


def test_families_has_seed_and_founder_columns() -> None:
    repo = _make_repo()
    cols = {r[1] for r in repo._conn.execute("PRAGMA table_info(families)")}
    assert "seeded_prior_combos" in cols
    assert "founder_gate_id" in cols


def test_family_members_has_persisted_profile_columns() -> None:
    repo = _make_repo()
    cols = {r[1] for r in repo._conn.execute("PRAGMA table_info(family_members)")}
    assert "member_code_hash" in cols
    assert "member_factors_json" in cols


def test_agent_mint_grants_table_exists_and_is_human_only() -> None:
    repo = _make_repo()
    # granted_by_actor CHECK ='human'
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "INSERT INTO agent_mint_grants(granted_at, granted_by_actor, grant_count)"
            " VALUES ('2026-01-01T00:00:00+00:00', 'agent', 1)")
    # grant_count CHECK >=1
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "INSERT INTO agent_mint_grants(granted_at, granted_by_actor, grant_count)"
            " VALUES ('2026-01-01T00:00:00+00:00', 'human', 0)")


def test_seeded_prior_combos_check_rejects_negative_on_fresh_db() -> None:
    repo = _make_repo()
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "INSERT INTO families(name, created_at, created_by_actor, seeded_prior_combos)"
            " VALUES ('bad', '2026-01-01T00:00:00+00:00', 'agent', -1)")


# ---------------------------------------------------------------------------
# n_combos type/bound hardening (record_search_trial + fresh CHECK)
# ---------------------------------------------------------------------------

def test_record_search_trial_rejects_non_int_and_out_of_range() -> None:
    repo = _make_repo()
    for bad in (0, -5, MAX_N_COMBOS + 1, 2.5, "10", True):
        with pytest.raises(ValueError):
            repo.record_search_trial("s", bad, "{}")  # type: ignore[arg-type]
    # a valid one succeeds
    repo.record_search_trial("s", 4, "{}")
    assert repo.total_search_combos("s") == 4


# ---------------------------------------------------------------------------
# funnel_lifetime_search_combos — WHERE-filtered, overflow-safe, TRUE lifetime
# ---------------------------------------------------------------------------

def test_funnel_lifetime_exceeds_windowed_and_excludes_corrupt_rows() -> None:
    repo = _make_repo()
    # An old (out-of-90d-window) trial + a recent one.
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('a', 500, '{}', ?)", (old,))
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('b', 7, '{}', ?)", (datetime.now(UTC).isoformat(),))
    repo._conn.commit()
    assert repo.windowed_search_combos(90) == 7          # old trial forgotten
    assert repo.funnel_lifetime_search_combos() == 507   # lifetime keeps it
    assert repo.agent_novel_mint_seed() == 507           # seed == accessor


def test_funnel_lifetime_excludes_mistyped_and_overlarge_rows() -> None:
    repo = _make_repo()
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('ok', 10, '{}', ?)", (datetime.now(UTC).isoformat(),))
    # Corrupt rows inserted around the fresh CHECK via a raw sqlite handle would violate the
    # CHECK; instead simulate a migrated-DB legacy row by dropping the constraint semantics:
    # a TEXT n_combos and an overlarge integer both fail the WHERE filter (contribute 0).
    repo._conn.execute("PRAGMA ignore_check_constraints=ON")
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('bad_text', 'abc', '{}', ?)", (datetime.now(UTC).isoformat(),))
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('bad_big', ?, '{}', ?)",
        (MAX_N_COMBOS + 1, datetime.now(UTC).isoformat()))
    repo._conn.commit()
    # Only the well-typed in-range row contributes — no overflow, no permanent DoS.
    assert repo.funnel_lifetime_search_combos() == 10
    audit = repo.agent_novel_mint_audit()
    assert audit["search_trials_corruption_count"] == 2


def test_family_lifetime_excludes_corrupt_member_rows_like_funnel_path() -> None:
    """lifetime_combos_for_families() must apply the SAME well-typed-row filter as the funnel/
    mint-seed path: a legacy corrupt search_trials row for a family member is EXCLUDED, not
    silently coerced to 0 by SUM, so the two lifetime-accounting paths cannot disagree."""
    repo = _make_repo()
    fid = repo.create_family("f", actor="human")
    repo.assign_strategy_to_family(
        "m", fid, actor="human", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}")
    repo.record_search_trial("m", 12, "{}")  # well-typed → counts
    # A migrated-DB legacy corrupt row (pre-#524, no CHECK) for the member: mistyped + overlarge.
    repo._conn.execute("PRAGMA ignore_check_constraints=ON")
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('m', 'abc', '{}', ?)", (datetime.now(UTC).isoformat(),))
    repo._conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES ('m', ?, '{}', ?)", (MAX_N_COMBOS + 1, datetime.now(UTC).isoformat()))
    repo._conn.commit()
    # Only the well-typed row (12) is taxed; the two corrupt rows are excluded (not coerced to 0).
    assert repo.family_lifetime_combos(fid) == 12


# ---------------------------------------------------------------------------
# family_graph_fingerprint — monotone digest over the classifier read-set
# ---------------------------------------------------------------------------

def test_fingerprint_changes_on_family_mint_member_and_returns() -> None:
    repo = _make_repo()
    fp0 = repo.family_graph_fingerprint()

    fid = repo.create_family("f", actor="human")
    fp1 = repo.family_graph_fingerprint()
    assert fp1 != fp0  # a new family bumps it

    repo.assign_strategy_to_family(
        "m", fid, actor="human", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}")
    fp2 = repo.family_graph_fingerprint()
    assert fp2 != fp1  # a member assignment bumps it

    repo.persist_backtest_returns(
        "m", "2024-01-01", "2024-02-01", pd.Series([0.1, 0.2]))
    fp3 = repo.family_graph_fingerprint()
    assert fp3 != fp2  # a member-returns refresh bumps it (return-correlation axis)


def test_fingerprint_changes_on_member_removal() -> None:
    repo = _make_repo()
    fid = repo.create_family("f", actor="human")
    repo.assign_strategy_to_family(
        "m", fid, actor="human", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}")
    fp_before = repo.family_graph_fingerprint()
    # Soft-remove the member (removed_at flip) — active-only COUNT decreases.
    repo._conn.execute(
        "UPDATE family_members SET removed_at=? WHERE strategy_name='m' AND removed_at IS NULL",
        (datetime.now(UTC).isoformat(),))
    repo._conn.commit()
    assert repo.family_graph_fingerprint() != fp_before


def test_fingerprint_stable_when_nothing_changes() -> None:
    repo = _make_repo()
    repo.create_family("f", actor="human")
    assert repo.family_graph_fingerprint() == repo.family_graph_fingerprint()


def test_fingerprint_changes_on_legacy_profile_materialisation() -> None:
    """A one-time NULL->value profile materialisation is the only permitted UPDATE that mutates a
    classifier-read column without changing COUNT/MAX(id)/active-count; the null-profile-count
    component makes it bump the fingerprint so a concurrent materialisation trips the mint CAS."""
    repo = _make_repo()
    fid = repo.create_family("f", actor="human")
    # A legacy-style member row with a NULL persisted profile.
    repo._conn.execute(
        "INSERT INTO family_members(family_id, strategy_name, joined_at, joined_by_actor,"
        " removed_at, member_code_hash, member_factors_json) VALUES (?,?,?,?,NULL,NULL,NULL)",
        (fid, "legacy_m", datetime.now(UTC).isoformat(), "human"))
    repo._conn.commit()
    fp_before = repo.family_graph_fingerprint()
    n = repo.materialise_legacy_member_profiles()  # NULL->value (module not loadable -> '' hash)
    assert n == 1
    assert repo.family_graph_fingerprint() != fp_before  # materialisation bumps the fingerprint


def test_mint_reraises_non_name_integrity_error_not_masked_as_name_clash() -> None:
    """The uuid regenerate-retry loop must ONLY swallow a families.name UNIQUE clash; a bad
    founder_gate_id FK (or any other constraint) must surface as an IntegrityError, never be
    masked 8x as 'could not obtain a unique family name'."""
    repo = _make_repo()
    repo.record_search_trial("x", 40, "{}")
    pending = PendingNovelFamily(
        slug_base="x_family", actor="agent", verdict="novel", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}",
        graph_fingerprint=repo.family_graph_fingerprint(),
        founder_code_hash="", founder_factors_json="[]")
    repo._conn.execute("BEGIN")
    try:
        # gate_id=999 has no gate_evaluations row → FK violation, must re-raise (not name-clash).
        with pytest.raises(sqlite3.IntegrityError):
            repo._mint_agent_novel_family(pending, "x", 999)
    finally:
        repo._conn.rollback()


# ---------------------------------------------------------------------------
# Append-only triggers — the classifier read-set is immutable in the engine
# ---------------------------------------------------------------------------

def test_families_update_and_delete_are_rejected() -> None:
    repo = _make_repo()
    fid = repo.create_family("f", actor="human")
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute("UPDATE families SET name='x' WHERE id=?", (fid,))
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute("DELETE FROM families WHERE id=?", (fid,))


def test_family_events_and_parents_and_returns_are_append_only() -> None:
    repo = _make_repo()
    a = repo.create_family("a", actor="human")
    b = repo.create_family("b", actor="human")
    repo.add_parent_edge(b, a)
    repo.persist_backtest_returns("s", "2024-01-01", "2024-02-01", pd.Series([0.1]))
    for tbl in ("family_events", "family_parents", "backtest_returns"):
        with pytest.raises(sqlite3.IntegrityError):
            repo._conn.execute(f"DELETE FROM {tbl}")
        with pytest.raises(sqlite3.IntegrityError):
            repo._conn.execute(f"UPDATE {tbl} SET id=id")


def test_family_members_delete_rejected_but_removed_at_flip_allowed() -> None:
    repo = _make_repo()
    fid = repo.create_family("f", actor="human")
    repo.assign_strategy_to_family(
        "m", fid, actor="human", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}")
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute("DELETE FROM family_members")
    # A forbidden field rewrite is rejected...
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute("UPDATE family_members SET strategy_name='x'")
    # ...but the removed_at tombstone flip is permitted.
    repo._conn.execute(
        "UPDATE family_members SET removed_at=? WHERE strategy_name='m'",
        (datetime.now(UTC).isoformat(),))
    repo._conn.commit()
    assert repo.strategy_family("m") is None


def test_all_append_only_triggers_present() -> None:
    repo = _make_repo()
    trigs = {
        r[0] for r in repo._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    for tbl in ("families", "family_parents", "family_events", "backtest_returns"):
        assert f"trg_{tbl}_append_only_upd" in trigs
        assert f"trg_{tbl}_append_only_del" in trigs
    assert "trg_family_members_no_delete" in trigs
    assert "trg_family_members_append_only_upd" in trigs


# ---------------------------------------------------------------------------
# REPLACE (implicit-delete) append-only guard — recursive_triggers pragma (R9-H1)
# ---------------------------------------------------------------------------

def _connect_repo(tmp_path: object) -> SqliteStrategyRepository:
    """A repository over a real DB opened through the app's db.connect() helper (which sets
    PRAGMA recursive_triggers=ON), so the implicit-delete path of REPLACE fires BEFORE DELETE
    triggers. The in-memory _make_repo() above deliberately does NOT go through connect()."""
    from pathlib import Path

    conn = db.connect(Path(str(tmp_path)) / "reg.db")
    db.migrate(conn)
    return SqliteStrategyRepository(conn)


def test_replace_on_existing_row_aborts_via_recursive_triggers(tmp_path: object) -> None:
    """The implicit row-delete SQLite runs to resolve a REPLACE conflict must fire the append-only
    BEFORE DELETE trigger. Without PRAGMA recursive_triggers=ON this silently succeeds — an in-place
    rewrite masquerading as an append. Covers families / family_members / backtest_returns."""
    repo = _connect_repo(tmp_path)
    fid = repo.create_family("f", actor="human")
    repo.assign_strategy_to_family(
        "m", fid, actor="human", verdict="NOVEL", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}")
    repo.persist_backtest_returns("s", "2024-01-01", "2024-02-01", pd.Series([0.1]))
    repo._conn.commit()
    mid = repo._conn.execute(
        "SELECT id FROM family_members WHERE strategy_name='m'").fetchone()[0]
    brid = repo._conn.execute(
        "SELECT id FROM backtest_returns WHERE strategy_name='s'").fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "REPLACE INTO families(id,name,created_at,created_by_actor,seeded_prior_combos)"
            " VALUES (?, 'g', '2026-01-02T00:00:00+00:00', 'agent', 9)", (fid,))
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "REPLACE INTO family_members(id,family_id,strategy_name,joined_at,joined_by_actor)"
            " VALUES (?, ?, 'm', '2026-01-02T00:00:00+00:00', 'agent')", (mid, fid))
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "REPLACE INTO backtest_returns(id,strategy_name,period_start,period_end,returns_json,"
            "created_at) VALUES (?, 's', '2024-01-01', '2024-02-01', X'00', "
            "'2026-01-02T00:00:00+00:00')", (brid,))


def test_raw_connection_without_connect_helper_bypasses_replace_guard(tmp_path: object) -> None:
    """Pins the NARROWED invariant claim: the REPLACE (implicit-delete) guard holds ONLY for
    connections opened via db.connect(). A raw sqlite3.connect() that skips the helper does not get
    PRAGMA recursive_triggers=ON, so a REPLACE conflict-delete bypasses the BEFORE DELETE trigger.
    (An EXPLICIT DELETE still aborts on such a handle — that is asserted separately.)"""
    from pathlib import Path

    path = Path(str(tmp_path)) / "raw.db"
    conn = db.connect(path)
    db.migrate(conn)
    repo = SqliteStrategyRepository(conn)
    fid = repo.create_family("f", actor="human")
    repo._conn.commit()
    conn.close()

    raw = sqlite3.connect(path)  # NOT db.connect() → recursive_triggers stays OFF (default)
    raw.execute("PRAGMA foreign_keys=ON;")
    # REPLACE succeeds — the append-only DELETE trigger is bypassed on this raw handle.
    raw.execute(
        "REPLACE INTO families(id,name,created_at,created_by_actor,seeded_prior_combos)"
        " VALUES (?, 'g', '2026-01-02T00:00:00+00:00', 'agent', 9)", (fid,))
    raw.commit()
    assert raw.execute("SELECT name FROM families WHERE id=?", (fid,)).fetchone()[0] == "g"
    # ...but an EXPLICIT DELETE still aborts even here (BEFORE DELETE fires unconditionally).
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM families WHERE id=?", (fid,))
    raw.close()


# ---------------------------------------------------------------------------
# Mint authority bounds: rate cap + human-replenished lifetime budget
# ---------------------------------------------------------------------------

def test_mint_bounds_pass_when_below_cap_and_budget() -> None:
    repo = _make_repo()
    repo.check_agent_novel_mint_bounds()  # no families → passes


def test_rate_cap_fails_closed_at_window_limit() -> None:
    repo = _make_repo()
    for i in range(AGENT_NOVEL_MINT_CAP):
        _insert_agent_family(repo, f"fam{i}")
    with pytest.raises(AgentMintCapError):
        repo.check_agent_novel_mint_bounds()


def test_rate_cap_ignores_out_of_window_families() -> None:
    repo = _make_repo()
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    for i in range(AGENT_NOVEL_MINT_CAP):
        _insert_agent_family(repo, f"old{i}", created_at=old)
    # All out of the 90d window → rate cap not tripped (budget still has room).
    repo.check_agent_novel_mint_bounds()


def test_rate_cap_does_not_count_human_families() -> None:
    repo = _make_repo()
    for i in range(AGENT_NOVEL_MINT_CAP + 2):
        repo.create_family(f"human{i}", actor="human")
    repo.check_agent_novel_mint_bounds()  # human families are not counted


def test_rate_cap_fails_closed_on_non_canonical_created_at() -> None:
    repo = _make_repo()
    # A naive (no tz) timestamp cannot be bucketed against the UTC cutoff → fail closed.
    repo._conn.execute(
        "INSERT INTO families(name, created_at, created_by_actor, seeded_prior_combos)"
        " VALUES ('naive', '2026-01-01 00:00:00', 'agent', 1)")
    repo._conn.commit()
    with pytest.raises(AgentMintCapError):
        repo.check_agent_novel_mint_bounds()


def test_lifetime_budget_fails_closed_when_exhausted_and_human_grant_replenishes() -> None:
    repo = _make_repo()
    # Fill the lifetime budget with out-of-window agent families (so the RATE cap is not the
    # one that trips — we want to prove the BUDGET is the binding bound).
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    for i in range(AGENT_NOVEL_MINT_LIFETIME_BUDGET):
        _insert_agent_family(repo, f"life{i}", created_at=old)
    with pytest.raises(AgentMintBudgetExhaustedError):
        repo.check_agent_novel_mint_bounds()
    # A human grant replenishes the budget.
    repo.grant_agent_novel_mints(5, actor="human", reason="epoch top-up")
    repo.check_agent_novel_mint_bounds()  # now passes


def test_agent_cannot_self_grant() -> None:
    repo = _make_repo()
    with pytest.raises(ValueError):
        repo.grant_agent_novel_mints(3, actor="agent")


def test_grant_rejects_non_positive_count() -> None:
    repo = _make_repo()
    with pytest.raises(ValueError):
        repo.grant_agent_novel_mints(0, actor="human")


# ---------------------------------------------------------------------------
# The mint seed guard: an all-corrupt/empty funnel fails closed
# ---------------------------------------------------------------------------

def test_mint_fails_closed_when_no_well_typed_trials() -> None:
    repo = _make_repo()
    fid_gate = None  # no gate row needed; the seed guard fires first
    pending = PendingNovelFamily(
        slug_base="x_family", actor="agent", verdict="novel", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}",
        graph_fingerprint=repo.family_graph_fingerprint(),
        founder_code_hash="", founder_factors_json="[]")
    # No search_trials at all → seed 0 → fail closed. Call the raw mint (needs an open tx to be
    # realistic, but the seed guard raises before any INSERT so a bare call surfaces the guard).
    repo._conn.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="strictly-positive"):
            repo._mint_agent_novel_family(pending, "x", fid_gate or 0)
    finally:
        repo._conn.rollback()


def _insert_min_gate_row(repo: SqliteStrategyRepository, strategy: str) -> int:
    """Insert a minimal strategies + gate_evaluations row so founder_gate_id has a valid FK
    target, and return the gate row id."""
    now = datetime.now(UTC).isoformat()
    sid = repo._conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES (?,?,?,?)",
        (strategy, "backtested", now, now)).lastrowid
    gid = repo._conn.execute(
        "INSERT INTO gate_evaluations(strategy_id, passed, n_funnel, own_lifetime_combos,"
        " windowed_total_combos, funnel_window_days, breadth_provenance, pit_ok,"
        " holdout_n_bars, min_holdout_observations, code_hash, config_hash, data_source,"
        " period_start, period_end, holdout_frac, actor, decision_json, created_at)"
        " VALUES (?,1,1,1,1,90,'measured',1,63,63,'c','c','demo','2024-01-01','2024-06-01',"
        " 0.2,'agent','{}',?)",
        (sid, now)).lastrowid
    repo._conn.commit()
    return int(gid)


def test_mint_succeeds_and_persists_seed_founder_and_profile() -> None:
    repo = _make_repo()
    repo.record_search_trial("x", 40, "{}")
    gid = _insert_min_gate_row(repo, "x")
    pending = PendingNovelFamily(
        slug_base="x_family", actor="agent", verdict="novel", similarity_score=0.0,
        clustering_version="v1", clustering_config_json="{}", axis_json="{}",
        graph_fingerprint=repo.family_graph_fingerprint(),
        founder_code_hash="abc123", founder_factors_json='["fx"]')
    repo._conn.execute("BEGIN")
    fid = repo._mint_agent_novel_family(pending, "x", gid)
    repo._conn.commit()
    row = repo._conn.execute(
        "SELECT seeded_prior_combos, founder_gate_id, created_by_actor, created_by_strategy"
        " FROM families WHERE id=?", (fid,)).fetchone()
    assert row["seeded_prior_combos"] == 40           # true lifetime seed captured at mint
    assert row["founder_gate_id"] == gid              # R9-M2 founder→gate audit link
    assert row["created_by_actor"] == "agent"
    assert row["created_by_strategy"] == "x"
    # The founder is a member with its classified profile DB-persisted from birth.
    mrow = repo._conn.execute(
        "SELECT member_code_hash, member_factors_json FROM family_members"
        " WHERE strategy_name='x' AND removed_at IS NULL").fetchone()
    assert mrow["member_code_hash"] == "abc123"
    assert mrow["member_factors_json"] == '["fx"]'
    # founder_gate_id queryable both directions.
    assert repo.family_lifetime_combos(fid) == 40 + 40  # seed + the founder's own real trial


def test_grant_agent_novel_mints_audit_math() -> None:
    repo = _make_repo()
    row_id = repo.grant_agent_novel_mints(10, actor="human", reason="r")
    assert row_id >= 1
    audit = repo.agent_novel_mint_audit()
    assert audit["lifetime_allowance"] == AGENT_NOVEL_MINT_LIFETIME_BUDGET + 10
    assert audit["lifetime_consumed"] == 0
    assert audit["lifetime_remaining"] == AGENT_NOVEL_MINT_LIFETIME_BUDGET + 10
