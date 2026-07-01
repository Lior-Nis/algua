"""Tests for Task 7 (#222): backtest returns persistence + return-correlation axis."""
from __future__ import annotations

import json
import math
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from algua.registry.db import migrate
from algua.research.clustering import (
    MERGE_THRESHOLD,
    PARENTAGE_THRESHOLD,
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


def test_load_backtest_returns_same_timestamp_id_tiebreak(store):
    """Two rows with identical created_at -> the higher id (later insert) wins (#257).

    created_at alone is not a deterministic ordering key (microsecond timestamps can
    collide for back-to-back runs); the id DESC tiebreak makes 'latest' well-defined.
    """
    ts = "2023-06-01T00:00:00+00:00"
    first = json.dumps([["2023-06-01", 0.01], ["2023-06-02", 0.02]])
    second = json.dumps([["2023-06-01", 0.05], ["2023-06-02", 0.06], ["2023-06-03", 0.07]])
    # Insert both rows with the SAME created_at so only the id can break the tie.
    store._conn.execute(
        "INSERT INTO backtest_returns"
        " (strategy_name, period_start, period_end, returns_json, created_at)"
        " VALUES (?,?,?,?,?)",
        ("strat", "2023-06-01", "2023-06-03", first, ts),
    )
    store._conn.execute(
        "INSERT INTO backtest_returns"
        " (strategy_name, period_start, period_end, returns_json, created_at)"
        " VALUES (?,?,?,?,?)",
        ("strat", "2023-06-01", "2023-06-04", second, ts),
    )
    store._conn.commit()
    loaded = store.load_backtest_returns("strat")
    assert loaded is not None
    # The later-inserted (higher id) row deterministically wins the tie.
    assert len(loaded) == 3
    assert list(loaded.values) == pytest.approx([0.05, 0.06, 0.07])


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


# ---------------------------------------------------------------------------
# #338: standalone return-correlation escalation
#
# Return-correlation is the gaming-resistant axis (code can be rewritten, factors
# re-declared). It must be able to BIND on its own — a rewritten + re-labelled clone
# that trades identically must not escape into NOVEL (fresh family, zero inherited
# breadth). Escalation is max(blend, ret), applied ONLY in family_similarity.
# ---------------------------------------------------------------------------


def _correlated_series(target_noise: float, n: int = 200, seed: int = 42):
    """Two return series whose Pearson corr ~= 1/sqrt(1+target_noise**2)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    s1 = pd.Series(base, index=idx, dtype=float)
    s2 = pd.Series(base + target_noise * noise, index=idx, dtype=float)
    return s1, s2


def test_standalone_return_corr_binds_merge_despite_code_factor_mismatch():
    """Identical trading (corr ~1.0) but rewritten code + disjoint factors -> MERGE.

    Old behaviour: blend = 0.20*1.0 = 0.20 -> NOVEL (the #338 evasion). With standalone
    escalation, max(0.20, ~1.0) -> MERGE, folding the clone back into the incumbent family.
    """
    n = 70
    strat_ret = _make_series([float(i) for i in range(n)])
    member_ret = _make_series([float(i) * 5.0 for i in range(n)])  # perfect positive corr

    members = [{"code_hash": "rewritten_hash", "factors": {"x", "y"}, "name": "mem"}]
    verdict, score = family_similarity(
        strategy_code_hash="original_hash",   # code axis = 0.0
        strategy_factors={"a", "b"},           # factor axis = 0.0 (disjoint)
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "mem": member_ret},
    )
    assert math.isclose(score, 1.0, rel_tol=1e-9)
    assert verdict == SimVerdict.MERGE


def test_standalone_return_corr_binds_parentage_moderate():
    """Moderate return corr (~0.71) with code/factor mismatch -> PARENTAGE (not NOVEL)."""
    strat_ret, member_ret = _correlated_series(target_noise=1.0)  # corr ~ 0.707
    members = [{"code_hash": "hb", "factors": {"x"}, "name": "mem"}]
    verdict, score = family_similarity(
        strategy_code_hash="ha",               # mismatch
        strategy_factors={"a"},                 # disjoint
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "mem": member_ret},
    )
    assert 0.55 < score < 0.82, score
    assert PARENTAGE_THRESHOLD <= score < MERGE_THRESHOLD
    assert verdict == SimVerdict.PARENTAGE


def test_standalone_low_return_corr_stays_novel():
    """Weak return corr (~0.32) with code/factor mismatch -> still NOVEL (does not bind)."""
    strat_ret, member_ret = _correlated_series(target_noise=3.0)  # corr ~ 0.316
    members = [{"code_hash": "hb", "factors": {"x"}, "name": "mem"}]
    verdict, score = family_similarity(
        strategy_code_hash="ha",
        strategy_factors={"a"},
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "mem": member_ret},
    )
    assert score < PARENTAGE_THRESHOLD, score
    assert verdict == SimVerdict.NOVEL


def test_pairwise_axes_stays_pure_no_standalone_escalation():
    """pairwise_axes must NOT apply the escalation — it stays the pure linear blend so
    family_audit's provenance-gated return logic is unaffected. Identical returns + code/
    factor mismatch -> blended == WEIGHT_RETURN_CORRELATION (0.20), NOT max()-escalated.
    """
    from algua.research.clustering import WEIGHT_RETURN_CORRELATION, pairwise_axes

    n = 70
    s = _make_series([float(i) for i in range(n)])
    blended, axes = pairwise_axes("ha", {"a"}, s, "hb", {"b"}, s)  # corr 1.0, code/factor 0
    assert axes["return"] is not None and math.isclose(axes["return"], 1.0, abs_tol=1e-9)
    assert math.isclose(blended, WEIGHT_RETURN_CORRELATION, rel_tol=1e-9)  # 0.20, not escalated


def test_standalone_escalation_is_monotone_never_lowers():
    """When the linear blend already exceeds the raw return axis, max() picks the blend —
    escalation is a no-op and never LOWERS the score (forward-only tightening). With a
    code+factor match the blend is 0.80 + 0.20*ret, which strictly exceeds ret, so the
    standalone floor cannot pull the score down.
    """
    from algua.research.clustering import _return_correlation_axis

    strat_ret, member_ret = _correlated_series(target_noise=1.0)
    ret = _return_correlation_axis(strat_ret, member_ret)
    assert ret is not None
    blend = 0.50 + 0.30 + 0.20 * ret  # code=1.0, factor=1.0, + return term
    assert blend > ret  # blend dominates -> max() must leave the score == blend

    members = [{"code_hash": "same", "factors": {"a", "b"}, "name": "mem"}]
    _verdict, score = family_similarity(
        strategy_code_hash="same",              # code = 1.0
        strategy_factors={"a", "b"},             # factor = 1.0
        family_members=members,
        returns_lookup={"__strategy__": strat_ret, "mem": member_ret},
    )
    assert math.isclose(score, blend, rel_tol=1e-9)  # escalation did not lower the blend


def test_clustering_version_changes_when_escalation_toggled(monkeypatch):
    """The standalone-escalation flag is part of the config digest (records invalidate)."""
    import algua.research.clustering as m

    current = clustering_version()
    monkeypatch.setattr(m, "_RETURN_STANDALONE_ESCALATION", False)
    assert clustering_version() != current
