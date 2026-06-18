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
    assert list(np.frombuffer(row["returns_blob"], dtype=np.float64)) == pytest.approx(rets)
    assert row["bar_dates_blob"].decode("utf-8").split("\n") == dates


def test_length_mismatch_raises(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.1, 0.2], bar_dates=["2020-12-31"])


def test_strategy_id_mismatch_raises(repo_with_burn, other_strategy_id):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(
            hid, other_strategy_id, holdout_start=h_start, holdout_end=h_end,
            returns=[0.1], bar_dates=["2020-12-31"])


def test_unique_prevents_double_write(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                returns=[0.1], bar_dates=["2020-12-31"])
    with pytest.raises(ValueError):  # UNIQUE(holdout_evaluation_id) -> second write rejected
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.2], bar_dates=["2020-12-31"])


def test_finalize_requires_matching_strategy_id(repo_pending_reservation, other_strategy_id):
    repo, sid, rid = repo_pending_reservation  # a PENDING (committed_at IS NULL) reservation
    with pytest.raises(ValueError):            # wrong strategy_id -> rowcount 0 -> raise
        repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=other_strategy_id)
    # correct strategy_id commits the burn
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)
    committed = repo._conn.execute(
        "SELECT committed_at FROM holdout_evaluations WHERE id=?", (rid,)).fetchone()
    assert committed["committed_at"] is not None
