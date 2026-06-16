"""Tests for the factor_evaluations ledger repository methods (#219, slice E of #140).

Mirrors test_shortlist_gate.py's _repo(tmp_path) fixture pattern.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

UTC = UTC


def _repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _row(
    *,
    factor_name: str = "momentum",
    import_path: str = "algua.features.momentum:score",
    code_hash: str = "abc123",
    hypothesis_hash: str = "hyp1",
    period_start: str = "2023-01-01",
    period_end: str = "2023-12-31",
    horizon: int = 1,
    params_json: str = '{"lookback": 60}',
    construction: str = "equal_weight",
    construction_params_json: str = "{}",
    n_obs: int | None = 30,
    mean_ic: float | None = 0.04,
    ic_ir: float | None = 0.8,
    t_stat: float | None = 4.4,
    ic_skew: float | None = 0.1,
    ic_kurtosis: float | None = 3.2,
    n_dependents: int = 2,
    data_source: str = "demo",
    snapshot_id: str | None = None,
    actor: str = "agent",
    created_at: str | None = None,
) -> dict:
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    return dict(
        factor_name=factor_name,
        import_path=import_path,
        code_hash=code_hash,
        hypothesis_hash=hypothesis_hash,
        period_start=period_start,
        period_end=period_end,
        horizon=horizon,
        params_json=params_json,
        construction=construction,
        construction_params_json=construction_params_json,
        n_obs=n_obs,
        mean_ic=mean_ic,
        ic_ir=ic_ir,
        t_stat=t_stat,
        ic_skew=ic_skew,
        ic_kurtosis=ic_kurtosis,
        n_dependents=n_dependents,
        data_source=data_source,
        snapshot_id=snapshot_id,
        actor=actor,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# record_factor_evaluation
# ---------------------------------------------------------------------------

def test_record_returns_positive_row_id(tmp_path):
    repo = _repo(tmp_path)
    row_id = repo.record_factor_evaluation(**_row())
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_correction_cols_are_null_until_finalize(tmp_path):
    """Correction columns are NULL immediately after record; fail-closed."""
    repo = _repo(tmp_path)
    row_id = repo.record_factor_evaluation(**_row())
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    db_row = conn.execute(
        "SELECT n_hypotheses, dsr_confidence, significant FROM factor_evaluations WHERE id=?",
        (row_id,),
    ).fetchone()
    assert db_row["n_hypotheses"] is None
    assert db_row["dsr_confidence"] is None
    assert db_row["significant"] is None


# ---------------------------------------------------------------------------
# factor_hypothesis_breadth
# ---------------------------------------------------------------------------

def test_breadth_includes_just_inserted_row(tmp_path):
    """own_lifetime and windowed_total both reflect the just-inserted hypothesis."""
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1"))
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 1
    assert windowed == 1


def test_same_hypothesis_hash_does_not_inflate_count(tmp_path):
    """Inserting the same hypothesis_hash twice counts as 1 distinct hypothesis."""
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1"))
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1"))  # same hash, re-run
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 1
    assert windowed == 1


def test_different_hypothesis_hash_inflates_own_count(tmp_path):
    """Two distinct hashes for the same factor → own_lifetime = 2."""
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1"))
    repo.record_factor_evaluation(**_row(hypothesis_hash="h2"))
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 2
    assert windowed == 2


def test_different_factor_contributes_to_windowed_but_not_own(tmp_path):
    """A different factor's hash increments windowed_total but not own_lifetime."""
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(factor_name="momentum", hypothesis_hash="h1"))
    repo.record_factor_evaluation(**_row(factor_name="reversion", hypothesis_hash="h2"))
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 1        # only momentum's hash
    assert windowed == 2   # momentum + reversion both in window


def test_old_hypothesis_excluded_from_windowed(tmp_path):
    """A hypothesis_hash inserted before the window cutoff is excluded from windowed_total."""
    repo = _repo(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=95)).isoformat()
    repo.record_factor_evaluation(**_row(hypothesis_hash="old", created_at=old_ts))
    repo.record_factor_evaluation(**_row(hypothesis_hash="recent"))  # within 90 days
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 2        # own_lifetime is ALL TIME (not windowed)
    assert windowed == 1   # only recent is within 90 days


def test_old_hypothesis_included_in_own_lifetime(tmp_path):
    """own_lifetime counts all-time distinct hashes for this factor, not windowed."""
    repo = _repo(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    repo.record_factor_evaluation(**_row(hypothesis_hash="ancient", created_at=old_ts))
    own, windowed = repo.factor_hypothesis_breadth("momentum", window_days=90)
    assert own == 1
    assert windowed == 0   # outside window


# ---------------------------------------------------------------------------
# windowed_factor_irs
# ---------------------------------------------------------------------------

def test_windowed_irs_empty_on_fresh_db(tmp_path):
    repo = _repo(tmp_path)
    assert repo.windowed_factor_irs(window_days=90) == []


def test_windowed_irs_returns_finite_irs(tmp_path):
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1", ic_ir=0.8))
    repo.record_factor_evaluation(**_row(factor_name="rev", hypothesis_hash="h2", ic_ir=1.2))
    irs = sorted(repo.windowed_factor_irs(window_days=90))
    assert irs == pytest.approx(sorted([0.8, 1.2]))


def test_windowed_irs_deduplicates_to_latest_per_hash(tmp_path):
    """Two rows with the same hypothesis_hash → only latest IR counted."""
    repo = _repo(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1", ic_ir=0.5, created_at=old_ts))
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1", ic_ir=0.9))  # newer
    irs = repo.windowed_factor_irs(window_days=90)
    assert irs == [pytest.approx(0.9)]  # only latest


def test_windowed_irs_excludes_none_ir(tmp_path):
    """A NULL ic_ir (underpowered eval) is excluded from the IR list."""
    repo = _repo(tmp_path)
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1", ic_ir=None))
    irs = repo.windowed_factor_irs(window_days=90)
    assert irs == []


def test_windowed_irs_excludes_old_rows(tmp_path):
    """Rows outside the window are excluded."""
    repo = _repo(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    repo.record_factor_evaluation(**_row(hypothesis_hash="h1", ic_ir=1.5, created_at=old_ts))
    irs = repo.windowed_factor_irs(window_days=90)
    assert irs == []


# ---------------------------------------------------------------------------
# finalize_factor_evaluation
# ---------------------------------------------------------------------------

def test_finalize_writes_correction_columns(tmp_path):
    repo = _repo(tmp_path)
    row_id = repo.record_factor_evaluation(**_row())
    repo.finalize_factor_evaluation(row_id, n_hypotheses=3, dsr_confidence=0.97, significant=True)
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    db_row = conn.execute(
        "SELECT n_hypotheses, dsr_confidence, significant FROM factor_evaluations WHERE id=?",
        (row_id,),
    ).fetchone()
    assert db_row["n_hypotheses"] == 3
    assert abs(db_row["dsr_confidence"] - 0.97) < 1e-9
    assert db_row["significant"] == 1


def test_finalize_significant_false_stored_as_zero(tmp_path):
    repo = _repo(tmp_path)
    row_id = repo.record_factor_evaluation(**_row())
    repo.finalize_factor_evaluation(row_id, n_hypotheses=5, dsr_confidence=0.3, significant=False)
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    db_row = conn.execute(
        "SELECT significant FROM factor_evaluations WHERE id=?", (row_id,)
    ).fetchone()
    assert db_row["significant"] == 0


def test_finalize_dsr_confidence_none_stored_as_null(tmp_path):
    """When DSR didn't bind, dsr_confidence=None → stored as NULL."""
    repo = _repo(tmp_path)
    row_id = repo.record_factor_evaluation(**_row())
    repo.finalize_factor_evaluation(row_id, n_hypotheses=1, dsr_confidence=None, significant=True)
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    db_row = conn.execute(
        "SELECT dsr_confidence FROM factor_evaluations WHERE id=?", (row_id,)
    ).fetchone()
    assert db_row["dsr_confidence"] is None
