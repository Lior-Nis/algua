"""Tests for the 3-way effective_funnel_breadth (#222 Task 5)."""
from __future__ import annotations

import random

from algua.research.gates import effective_funnel_breadth


def test_effective_funnel_breadth_3way_max() -> None:
    assert effective_funnel_breadth(10, 5, 8) == 10
    assert effective_funnel_breadth(10, 5, 15) == 15
    assert effective_funnel_breadth(10, 15, 8) == 15


def test_effective_funnel_breadth_default_backward_compat() -> None:
    # 2-arg call still works (family=0 default)
    assert effective_funnel_breadth(10, 15) == 15
    assert effective_funnel_breadth(15, 10) == 15


def test_tighten_only_strong_property() -> None:
    # For any (own, windowed, family): 3-way >= 2-way always
    random.seed(42)
    for _ in range(100):
        own = random.randint(0, 1000)
        windowed = random.randint(0, 1000)
        family = random.randint(0, 1000)
        result_3 = effective_funnel_breadth(own, windowed, family)
        result_2 = effective_funnel_breadth(own, windowed)
        assert result_3 >= result_2


def test_crowded_family_raises_bar() -> None:
    # Family with 1000 combos dominates own=50, windowed=200
    assert effective_funnel_breadth(50, 200, 1000) == 1000


def test_empty_family_fallback_unchanged() -> None:
    # family=0 → same as 2-arg
    assert effective_funnel_breadth(50, 200, 0) == effective_funnel_breadth(50, 200)


def test_gate_evaluations_columns_exist() -> None:
    # After migration, gate_evaluations has family_id and family_lifetime_effective columns
    import sqlite3

    from algua.registry import db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(gate_evaluations)")}
    assert "family_id" in cols
    assert "family_lifetime_effective" in cols
