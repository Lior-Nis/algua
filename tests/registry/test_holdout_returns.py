"""Tests for holdout_returns table — schema 26→27 (#221 Slice 1).

Verifies the new table is created by migrate(), its exact column set, and
that the UNIQUE index on holdout_evaluation_id is present and enforced.
"""
from __future__ import annotations

from algua.registry.db import SCHEMA_VERSION, connect, migrate


def test_schema_version_is_27():
    assert SCHEMA_VERSION == 27


def test_holdout_returns_table_and_indexes_exist(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(holdout_returns)")}
    assert cols == {
        "id", "holdout_evaluation_id", "strategy_id", "holdout_start", "holdout_end",
        "n_bars", "returns_blob", "bar_dates_blob", "created_at",
    }
    idx = {r["name"] for r in conn.execute("PRAGMA index_list(holdout_returns)")}
    assert "ux_holdout_returns_eval" in idx          # UNIQUE(holdout_evaluation_id)
    # confirm UNIQUE is enforced
    uniq = [r for r in conn.execute("PRAGMA index_list(holdout_returns)")
            if r["name"] == "ux_holdout_returns_eval"]
    assert uniq and uniq[0]["unique"] == 1
