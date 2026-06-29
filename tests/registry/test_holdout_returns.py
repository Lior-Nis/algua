"""Tests for holdout_returns table — schema 26→27 (#221 Slice 1).

Verifies the new table is created by migrate(), its exact column set, and
that the UNIQUE index on holdout_evaluation_id is present and enforced.
Also covers the store write path: record_holdout_returns + finalize strategy_id hardening.
"""
from __future__ import annotations

import numpy as np
import pytest

from algua.registry.db import SCHEMA_VERSION, connect, migrate
from algua.registry.store import SqliteStrategyRepository


def test_schema_version_is_31():
    assert SCHEMA_VERSION == 31


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


# ---------------------------------------------------------------------------
# Fixtures for store write-path tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def _repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _reserve(repo, sid, *, hs, he, frac=0.2, ps="2020-01-01", pe="2020-12-31",
             ds="demo", snap=None, allow_reuse=False):
    return repo.reserve_holdout(
        sid, data_source=ds, snapshot_id=snap, period_start=ps, period_end=pe,
        holdout_frac=frac, holdout_start=hs, holdout_end=he, allow_reuse=allow_reuse)


@pytest.fixture()
def repo_with_burn(_repo):
    """A committed holdout burn: (repo, strategy_id, holdout_evaluation_id, (h_start, h_end))."""
    sid = _repo.add("alpha").id
    h_start, h_end = "2020-12-29", "2020-12-31"
    rid, _ = _reserve(_repo, sid, hs=h_start, he=h_end)
    _repo.finalize_holdout_reservation(rid, config_hash="testhash", strategy_id=sid)
    return _repo, sid, rid, (h_start, h_end)


@pytest.fixture()
def repo_pending_reservation(_repo):
    """A pending (committed_at IS NULL) reservation: (repo, strategy_id, reservation_id)."""
    sid = _repo.add("beta").id
    rid, _ = _reserve(_repo, sid, hs="2020-12-29", he="2020-12-31")
    return _repo, sid, rid


@pytest.fixture()
def other_strategy_id(_repo):
    """A second registered strategy's id — never touches the burn."""
    return _repo.add("other").id


# ---------------------------------------------------------------------------
# store write-path tests
# ---------------------------------------------------------------------------

def test_record_and_read_back_round_trip(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn  # committed burn fixture
    rets = [0.01, -0.02, 0.005]
    dates = ["2020-12-29", "2020-12-30", "2020-12-31"]
    rid = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                      returns=rets, bar_dates=dates)
    assert rid > 0
    row = repo._conn.execute(
        "SELECT n_bars, returns_blob, bar_dates_blob FROM holdout_returns WHERE id=?", (rid,)
    ).fetchone()
    assert row["n_bars"] == 3
    assert list(np.frombuffer(row["returns_blob"], dtype="<f8")) == pytest.approx(rets)
    assert row["bar_dates_blob"].decode("utf-8").split("\n") == dates


def test_length_mismatch_raises(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.1, 0.2], bar_dates=["2020-12-31"])


def test_strategy_id_mismatch_raises(repo_with_burn, other_strategy_id):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError, match="does not match"):
        repo.record_holdout_returns(
            hid, other_strategy_id, holdout_start=h_start, holdout_end=h_end,
            returns=[0.1], bar_dates=["2020-12-31"])


def test_unique_prevents_double_write(repo_with_burn):
    """A second write with DIFFERENT returns raises ValueError (conflicting double-write)."""
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                returns=[0.1], bar_dates=["2020-12-31"])
    with pytest.raises(ValueError, match="different content"):
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.2], bar_dates=["2020-12-31"])


def test_idempotent_rewrite_returns_existing_id(repo_with_burn):
    """Identical content written twice: second call returns the SAME row id, ONE row in DB."""
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    rets = [0.01, -0.02, 0.005]
    dates = ["2020-12-29", "2020-12-30", "2020-12-31"]
    rid1 = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                       returns=rets, bar_dates=dates)
    rid2 = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                       returns=rets, bar_dates=dates)
    assert rid1 == rid2, "second identical write must return the same row id"
    count = repo._conn.execute(
        "SELECT COUNT(*) FROM holdout_returns WHERE holdout_evaluation_id=?", (hid,)
    ).fetchone()[0]
    assert count == 1, "exactly one row must exist after idempotent double-write"


def test_reconciliation_after_missing_gate_row(repo_with_burn):
    """Committed burn with returns row already present → re-calling record_holdout_returns
    with identical args succeeds and returns the existing id (reconciliation path)."""
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    rets = [0.03, 0.04]
    dates = ["2020-12-29", "2020-12-30"]
    # Simulate: returns written, gate row absent (will be written later by re-run).
    rid1 = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                       returns=rets, bar_dates=dates)
    # Re-run calls record_holdout_returns again with identical args — must not raise.
    rid2 = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                       returns=rets, bar_dates=dates)
    assert rid1 == rid2


def test_finalize_requires_matching_strategy_id(repo_pending_reservation, other_strategy_id):
    repo, sid, rid = repo_pending_reservation  # a PENDING (committed_at IS NULL) reservation
    with pytest.raises(ValueError):            # wrong strategy_id -> rowcount 0 -> raise
        repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=other_strategy_id)
    # correct strategy_id commits the burn
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)
    committed = repo._conn.execute(
        "SELECT committed_at FROM holdout_evaluations WHERE id=?", (rid,)).fetchone()
    assert committed["committed_at"] is not None


# ---------------------------------------------------------------------------
# Fixtures for overlapping_holdout_return_streams access-control tests
# ---------------------------------------------------------------------------

def _make_burn_and_returns(repo, name, *, h_start, h_end, returns, dates):
    """Helper: register strategy, create committed burn, write return vector. Returns (sid, hid)."""
    sid = repo.add(name).id
    rid, _ = _reserve(repo, sid, hs=h_start, he=h_end)
    repo.finalize_holdout_reservation(rid, config_hash="h", strategy_id=sid)
    repo.record_holdout_returns(rid, sid, holdout_start=h_start, holdout_end=h_end,
                                returns=returns, bar_dates=dates)
    return sid, rid


@pytest.fixture()
def repo_with_burn_and_returns(_repo):
    """Single strategy with a return vector — used by singleton-funnel test."""
    sid, hid = _make_burn_and_returns(
        _repo, "alpha",
        h_start="2020-12-29", h_end="2020-12-31",
        returns=[0.01, -0.02, 0.005],
        dates=["2020-12-29", "2020-12-30", "2020-12-31"],
    )
    interval = ("2020-12-29", "2020-12-31")
    window = 365
    return _repo, sid, interval, window


@pytest.fixture()
def repo_with_two_strategy_burns(_repo):
    """Two strategies A and B with overlapping holdout intervals and return vectors."""
    a_id, _ = _make_burn_and_returns(
        _repo, "alpha",
        h_start="2020-12-01", h_end="2020-12-31",
        returns=[0.01, -0.02],
        dates=["2020-12-01", "2020-12-31"],
    )
    b_id, _ = _make_burn_and_returns(
        _repo, "bravo",
        h_start="2020-12-15", h_end="2021-01-15",
        returns=[0.03, 0.04, 0.05],
        dates=["2020-12-15", "2020-12-31", "2021-01-15"],
    )
    interval = ("2020-12-01", "2020-12-31")
    window = 365
    return _repo, a_id, b_id, interval, window


@pytest.fixture()
def repo_with_two_strategy_burns_disjoint(_repo):
    """Two strategies where B's OOS interval does NOT overlap A's query interval."""
    a_id, _ = _make_burn_and_returns(
        _repo, "alpha",
        h_start="2020-01-01", h_end="2020-06-30",
        returns=[0.01, 0.02],
        dates=["2020-01-01", "2020-06-30"],
    )
    # B's interval is entirely after A's — disjoint
    b_id, _ = _make_burn_and_returns(
        _repo, "bravo",
        h_start="2020-07-01", h_end="2020-12-31",
        returns=[0.03, 0.04],
        dates=["2020-07-01", "2020-12-31"],
    )
    a_interval = ("2020-01-01", "2020-06-30")
    window = 365
    return _repo, a_id, b_id, a_interval, window


# ---------------------------------------------------------------------------
# Access-control tests for overlapping_holdout_return_streams
# ---------------------------------------------------------------------------

def test_sibling_read_excludes_own_vector(repo_with_two_strategy_burns):
    # strategy A and sibling B both have overlapping-interval holdout_returns rows.
    repo, a_id, b_id, interval, window = repo_with_two_strategy_burns
    streams = repo.overlapping_holdout_return_streams(a_id, interval[0], interval[1], window)
    # B's vector is returned; A's own is NOT.
    assert len(streams) == 1
    assert all(isinstance(v, tuple) and len(v) == 2 for v in streams)
    # Roundtrip fidelity: the returned payload must equal B's actual returns/dates.
    b_returns = [0.03, 0.04, 0.05]
    b_dates = ["2020-12-15", "2020-12-31", "2021-01-15"]
    ret_vec, ret_dates = streams[0]
    assert ret_vec == pytest.approx(b_returns)
    assert ret_dates == b_dates


def test_singleton_funnel_returns_empty(repo_with_burn_and_returns):
    # only the requesting strategy has a vector -> it is its own sibling -> empty.
    repo, sid, interval, window = repo_with_burn_and_returns
    assert repo.overlapping_holdout_return_streams(sid, interval[0], interval[1], window) == []


def test_disjoint_interval_excluded(repo_with_two_strategy_burns_disjoint):
    repo, a_id, b_id, a_interval, window = repo_with_two_strategy_burns_disjoint
    # B's OOS interval does not overlap A's -> not a sibling for this query.
    assert repo.overlapping_holdout_return_streams(a_id, a_interval[0], a_interval[1], window) == []


def test_out_of_window_excluded(_repo):
    """A sibling whose burn he.created_at is older than window_days is not returned."""
    a_id, a_hid = _make_burn_and_returns(
        _repo, "alpha",
        h_start="2020-12-01", h_end="2020-12-31",
        returns=[0.01, -0.02],
        dates=["2020-12-01", "2020-12-31"],
    )
    b_id, b_hid = _make_burn_and_returns(
        _repo, "bravo",
        h_start="2020-12-15", h_end="2021-01-15",
        returns=[0.03, 0.04, 0.05],
        dates=["2020-12-15", "2020-12-31", "2021-01-15"],
    )
    # Backdate B's holdout_evaluation created_at to 400 days ago (outside a 365-day window)
    from datetime import UTC, datetime, timedelta
    old_ts = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    _repo._conn.execute(
        "UPDATE holdout_evaluations SET created_at=? WHERE id=?", (old_ts, b_hid)
    )
    _repo._conn.commit()

    interval = ("2020-12-01", "2020-12-31")
    streams = _repo.overlapping_holdout_return_streams(a_id, interval[0], interval[1], 365)
    assert streams == []


# ---------------------------------------------------------------------------
# GATE-2 findings: new guard tests
# ---------------------------------------------------------------------------

def test_empty_vector_raises(repo_with_burn):
    """Finding 5: record_holdout_returns with zero-length vector/dates raises ValueError."""
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[], bar_dates=[])


def test_pending_reservation_raises(repo_pending_reservation):
    """Finding 2: record_holdout_returns against a PENDING (uncommitted) reservation raises."""
    repo, sid, rid = repo_pending_reservation
    with pytest.raises(ValueError, match="not a committed burn"):
        repo.record_holdout_returns(rid, sid, holdout_start="2020-12-29", holdout_end="2020-12-31",
                                    returns=[0.1], bar_dates=["2020-12-31"])


def test_corrupt_bar_dates_blob_raises(repo_with_two_strategy_burns):
    """Finding 1: overlapping_holdout_return_streams raises ValueError when bar_dates_blob
    length != n_bars (corrupt row)."""
    repo, a_id, b_id, interval, window = repo_with_two_strategy_burns
    # Corrupt B's bar_dates_blob to a wrong-length value (only one date instead of three).
    repo._conn.execute(
        "UPDATE holdout_returns SET bar_dates_blob=? WHERE strategy_id=?",
        (b"only-one-date", b_id),
    )
    repo._conn.commit()
    with pytest.raises(ValueError, match="bar_dates_blob"):
        repo.overlapping_holdout_return_streams(a_id, interval[0], interval[1], window)
