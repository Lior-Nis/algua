"""Tests for Task 7 (#222): backtest returns persistence + return-correlation axis."""
from __future__ import annotations

import math
import re
import sqlite3

import pandas as pd
import pytest

from algua.registry.db import migrate
from algua.research.clustering import (
    SimVerdict,
    _return_correlation_axis,
    clustering_version,
    family_similarity,
)

# ---------------------------------------------------------------------------
# Return-correlation axis (pure function)
# ---------------------------------------------------------------------------


def _make_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    """Helper: build a pd.Series with a daily DatetimeIndex."""
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_return_correlation_reference_value():
    """Perfectly correlated returns -> axis score ~1.0."""
    n = 70  # > 63 min overlap
    s1 = _make_series([float(i) for i in range(n)])
    s2 = _make_series([float(i) * 2.0 for i in range(n)])  # perfect positive correlation
    result = _return_correlation_axis(s1, s2)
    assert result is not None
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_return_correlation_anticorrelated_clamped():
    """Anticorrelated returns -> axis score 0.0 (clamped)."""
    n = 70
    s1 = _make_series([float(i) for i in range(n)])
    s2 = _make_series([float(-i) for i in range(n)])  # perfect negative
    result = _return_correlation_axis(s1, s2)
    assert result is not None
    assert result == 0.0


def test_return_correlation_below_min_overlap_omitted():
    """Only 30 shared dates (< 63) -> axis returns None."""
    n = 30
    s1 = _make_series([float(i) for i in range(n)])
    s2 = _make_series([float(i) for i in range(n)])
    result = _return_correlation_axis(s1, s2, min_overlap=63)
    assert result is None


def test_return_correlation_none_inputs():
    """None inputs -> None."""
    s = _make_series([1.0, 2.0, 3.0])
    assert _return_correlation_axis(None, s) is None
    assert _return_correlation_axis(s, None) is None
    assert _return_correlation_axis(None, None) is None


def test_return_correlation_non_finite_returns_none():
    """Non-finite correlation result -> None (fail-closed)."""
    n = 70
    # Constant series -> correlation is NaN
    s1 = _make_series([1.0] * n)
    s2 = _make_series([2.0] * n)
    result = _return_correlation_axis(s1, s2)
    assert result is None


# ---------------------------------------------------------------------------
# Backtest returns table roundtrip
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    """Create an in-memory SqliteStrategyRepository with migrated schema."""
    from algua.registry.store import SqliteStrategyRepository

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_backtest_returns_table_roundtrip(store):
    """persist_backtest_returns + load_backtest_returns roundtrips correctly."""
    returns = _make_series([0.01, -0.02, 0.03, 0.0, 0.015], start="2023-06-01")
    row_id = store.persist_backtest_returns(
        strategy_name="test_strategy",
        period_start="2023-06-01",
        period_end="2023-06-08",
        returns=returns,
    )
    assert isinstance(row_id, int)

    loaded = store.load_backtest_returns("test_strategy")
    assert loaded is not None
    assert len(loaded) == len(returns)
    # Roundtrip does not preserve freq (JSON has no concept of freq), so check values + dates
    pd.testing.assert_index_equal(loaded.index, returns.index, check_names=False)
    assert list(loaded.values) == pytest.approx(list(returns.values))


def test_load_backtest_returns_nonexistent(store):
    """load_backtest_returns for unknown strategy -> None."""
    assert store.load_backtest_returns("nonexistent") is None


def test_load_backtest_returns_most_recent(store):
    """Multiple persisted series -> loads the most recent one."""
    old = _make_series([0.01, 0.02], start="2023-01-01")
    store.persist_backtest_returns("strat", "2023-01-01", "2023-01-03", old)
    new = _make_series([0.05, 0.06, 0.07], start="2023-06-01")
    store.persist_backtest_returns("strat", "2023-06-01", "2023-06-04", new)
    loaded = store.load_backtest_returns("strat")
    assert loaded is not None
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# Clustering version changes with return axis activation
# ---------------------------------------------------------------------------


def test_clustering_version_changes_with_return_axis():
    """_AXIS_AVAILABILITY changed -> clustering_version() value changes.

    Since the axis is now True, we verify by temporarily flipping it back to False.
    """
    import algua.research.clustering as m

    current_version = clustering_version()
    old_avail = dict(m._AXIS_AVAILABILITY)
    m._AXIS_AVAILABILITY = {**old_avail, "return_correlation": False}
    try:
        old_version = clustering_version()
    finally:
        m._AXIS_AVAILABILITY = old_avail
    assert current_version != old_version, (
        "clustering_version should change when return_correlation availability changes"
    )


def test_clustering_version_is_stable():
    """Calling clustering_version() multiple times yields the same value."""
    v1 = clustering_version()
    v2 = clustering_version()
    assert v1 == v2
    assert re.match(r"^[0-9a-f]{32}$", v1)


# ---------------------------------------------------------------------------
# family_similarity with return-correlation axis active
# ---------------------------------------------------------------------------


def test_family_similarity_with_return_lookup_increases_score():
    """Providing correlated returns via returns_lookup should increase the similarity score
    vs. not providing them (where the return axis contributes 0.0)."""
    n = 70
    strat_ret = _make_series([float(i) for i in range(n)])
    member_ret = _make_series([float(i) * 2.0 for i in range(n)])

    members = [{"code_hash": "abc", "factors": {"f1"}, "name": "member_a"}]

    _, score_no_ret = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1"},
        family_members=members,
    )
    _, score_with_ret = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1"},
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "member_a": member_ret},
    )
    # With perfectly correlated returns: return_score=1.0, adds 0.20*1.0 = 0.20
    assert score_with_ret > score_no_ret
    # code=1.0, factor=1.0, return=1.0 -> 0.50+0.30+0.20 = 1.00
    assert math.isclose(score_with_ret, 1.0, rel_tol=1e-9)


def test_family_similarity_return_lookup_insufficient_overlap():
    """If returns have < 63 shared dates, return_correlation contributes 0.0."""
    n = 30  # below threshold
    strat_ret = _make_series([float(i) for i in range(n)])
    member_ret = _make_series([float(i) for i in range(n)])

    members = [{"code_hash": "abc", "factors": {"f1"}, "name": "member_a"}]

    _, score_no_ret = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1"},
        family_members=members,
    )
    _, score_with_ret = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1"},
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "member_a": member_ret},
    )
    # Insufficient overlap -> return_score stays 0.0
    assert math.isclose(score_with_ret, score_no_ret, rel_tol=1e-9)


def test_family_similarity_return_lookup_missing_member_name():
    """Member without 'name' key -> return axis contributes 0.0 for that member."""
    n = 70
    strat_ret = _make_series([float(i) for i in range(n)])
    member_ret = _make_series([float(i) for i in range(n)])

    # Member has no 'name' key
    members = [{"code_hash": "abc", "factors": {"f1"}}]

    _, score = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1"},
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "member_a": member_ret},
    )
    # return axis not computed (no name) -> contributes 0.0
    # code=1.0, factor=1.0, return=0.0 -> 0.50 + 0.30 = 0.80
    assert math.isclose(score, 0.80, rel_tol=1e-9)


def test_merge_now_reachable_with_all_axes():
    """With return_correlation active, MERGE_THRESHOLD (0.85) is reachable:
    code=1.0, factor=1.0, return=1.0 -> 0.50+0.30+0.20 = 1.00 >= 0.85."""
    n = 70
    strat_ret = _make_series([float(i) for i in range(n)])
    member_ret = _make_series([float(i) * 3.0 for i in range(n)])

    members = [{"code_hash": "abc", "factors": {"f1", "f2"}, "name": "mem"}]

    verdict, score = family_similarity(
        strategy_code_hash="abc",
        strategy_factors={"f1", "f2"},
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "mem": member_ret},
    )
    assert math.isclose(score, 1.0, rel_tol=1e-9)
    assert verdict == SimVerdict.MERGE
