"""Tests for the ``attempt_token`` gate-row binding + read helper (#485, Task 0/5, R4).

Covers the additive nullable column, the partial unique index on non-null
``(strategy_id, attempt_token)``, and ``passing_gate_by_token`` — the authoritative read that binds
promote-outcome attribution to the merge-back attempt, not the ambient stage.
"""

from __future__ import annotations

import sqlite3

import pytest

from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

_COLS = (
    "strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos, funnel_window_days,"
    " breadth_provenance, pit_ok, holdout_n_bars, min_holdout_observations, code_hash, config_hash,"
    " data_source, period_start, period_end, holdout_frac, actor, decision_json, created_at,"
    " attempt_token"
)


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _insert_gate(conn, strategy_id, *, passed, actor, token):
    conn.execute(
        f"INSERT INTO gate_evaluations ({_COLS}) VALUES"
        " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (strategy_id, int(passed), 1, 1, 1, 365, "measured", 1, 100, 63, "c", "cfg",
         "Demo", "2022-01-01", "2022-12-31", 0.2, actor, "{}", "2022-01-01T00:00:00Z", token),
    )
    conn.commit()


def test_passing_gate_by_token_finds_our_row(tmp_path):
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=True, actor="agent", token="TOKEN_A")
    assert SqliteStrategyRepository(conn).passing_gate_by_token("s", "TOKEN_A") is not None


def test_concurrent_null_token_row_not_returned(tmp_path):
    # An external non-driver promote (NULL token, HIGHER id) must NOT be attributed to our attempt.
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=True, actor="agent", token="TOKEN_A")
    _insert_gate(conn, rec.id, passed=True, actor="agent", token=None)
    repo = SqliteStrategyRepository(conn)
    assert repo.passing_gate_by_token("s", "TOKEN_A") is not None
    # No row bears TOKEN_B, so the higher-id NULL row is never mistaken for it.
    assert repo.passing_gate_by_token("s", "TOKEN_B") is None


def test_relaxed_actor_row_not_returned(tmp_path):
    # A row bearing our token but a HUMAN (relaxed) actor is not the strict-agent attempt's row.
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=True, actor="human", token="TOKEN_A")
    assert SqliteStrategyRepository(conn).passing_gate_by_token("s", "TOKEN_A") is None


def test_failed_row_not_returned(tmp_path):
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=False, actor="agent", token="TOKEN_A")
    assert SqliteStrategyRepository(conn).passing_gate_by_token("s", "TOKEN_A") is None


def test_gate_exists_by_token_sees_failing_row(tmp_path):
    # HIGH-2 crash-idempotency read: unlike passing_gate_by_token, gate_exists_by_token returns True
    # for a FAILING row too (a failing promote still consumed the token + burned the holdout), so a
    # resume knows not to re-invoke.
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    repo = SqliteStrategyRepository(conn)
    assert repo.gate_exists_by_token("s", "TOKEN_A") is False
    _insert_gate(conn, rec.id, passed=False, actor="agent", token="TOKEN_A")
    assert repo.gate_exists_by_token("s", "TOKEN_A") is True     # failing row detected
    assert repo.passing_gate_by_token("s", "TOKEN_A") is None    # but not a PASS
    assert repo.gate_exists_by_token("s", "TOKEN_B") is False    # unrelated token absent


def test_duplicate_token_violates_unique_index(tmp_path):
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=True, actor="agent", token="TOKEN_A")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_gate(conn, rec.id, passed=False, actor="agent", token="TOKEN_A")


def test_null_tokens_do_not_collide(tmp_path):
    # The partial index ignores NULL rows, so many NULL-token rows coexist (backward-compatible).
    conn = _conn(tmp_path)
    rec = SqliteStrategyRepository(conn).add("s")
    _insert_gate(conn, rec.id, passed=True, actor="agent", token=None)
    _insert_gate(conn, rec.id, passed=True, actor="agent", token=None)  # no IntegrityError


def test_migration_adds_column_and_index_on_existing_db(tmp_path):
    # A DB migrated twice is idempotent and carries the column + partial unique index.
    conn = _conn(tmp_path)
    migrate(conn)  # re-run: idempotent
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gate_evaluations)")}
    assert "attempt_token" in cols
    idx = {r["name"] for r in conn.execute("PRAGMA index_list(gate_evaluations)")}
    assert "ux_gate_evaluations_attempt_token" in idx
