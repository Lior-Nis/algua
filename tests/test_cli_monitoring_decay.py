"""CLI tests for `algua monitoring decay` (issue #391).

End-to-end over a seeded registry DB: a real strategy, a passing forward-test certificate for its
current identity, and admissible `lane='live'` ticks. Verifies the advisory verdict, that it fails
closed (unknown / insufficient_data) when data is missing, and that a decay finding still exits 0.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from typer.testing import CliRunner

from algua.calendar.market_calendar import MarketCalendar
from algua.cli.main import app
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import MIN_FORWARD_OBSERVATIONS

runner = CliRunner()
STRAT = "cross_sectional_momentum"


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _register():
    assert runner.invoke(app, ["registry", "add", STRAT]).exit_code == 0


def _seed_cert(conn, strategy_id, ident, *, passed=True, holdout=1.0,
               created_at) -> int:
    repo = SqliteStrategyRepository(conn)
    rid = repo.record_forward_gate_evaluation(
        strategy_id, passed=passed, n_forward_observations=80, min_forward_observations=63,
        session_coverage=0.95, realized_sharpe=1.2, holdout_sharpe=holdout, degradation_factor=0.5,
        sharpe_floor=0.3, realized_vol=0.1, min_forward_vol=0.02, realized_max_drawdown=0.1,
        max_forward_drawdown=0.25, first_tick_id=1, last_tick_id=None, first_tick_ts=None,
        last_tick_ts=None, max_staleness_sessions=5, n_reconcile_failures=0,
        n_concurrent_forward=1, account_id="acct", code_hash=ident.code_hash,
        config_hash=ident.config_hash, dependency_hash=ident.dependency_hash, actor="agent",
        decision_json="{}", consumable=False)
    conn.execute("UPDATE forward_gate_evaluations SET created_at=? WHERE id=?",
                 (created_at, rid))
    conn.commit()
    return rid


def _seed_live_ticks(conn, strategy_id, ident, *, sharpe_target, n, start_after=None):
    """Insert n admissible live ticks (broker clock, matching identity, one per recent session).

    Equity is a compounding walk whose per-session returns give ~`sharpe_target` annualized. Each
    tick uses a distinct recent trading session for both tick_ts and decision_ts (lag 0). Returns
    the seeded session dates (ascending) so the caller can pin the certificate relative to them.
    """
    cal = MarketCalendar()
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.01, n)
    std = float(np.std(noise, ddof=1))
    mean = sharpe_target * std / (252.0 ** 0.5)
    rets = noise - float(np.mean(noise)) + mean

    # Build a list of n consecutive trading sessions ending "today".
    today = datetime.now(UTC).date()
    sess = cal.session_on_or_before(today)
    sessions = [sess]
    while len(sessions) < n:
        sess = cal.previous_session(sess)
        sessions.append(sess)
    sessions.reverse()
    _ = start_after  # ticks are all recent; the certificate is pinned just before sessions[0].
    equity = 100_000.0
    for i, day in enumerate(sessions):
        equity *= (1.0 + float(rets[i]))
        ts = datetime.combine(day, datetime.min.time(), UTC).replace(hour=20).isoformat()
        conn.execute(
            "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, positions,"
            " n_submitted, reconcile_ok, lane, strategy_id, clock_source, code_hash,"
            " config_hash, dependency_hash, account_id)"
            " VALUES (?, ?, ?, ?, '{}', 0, 1, 'live', ?, 'broker', ?, ?, ?, 'acct')",
            (STRAT, ts, ts, equity, strategy_id, ident.code_hash, ident.config_hash,
             ident.dependency_hash))
    conn.commit()
    return sessions


def _cert_ts_before(sessions):
    """Certificate instant one trading session before the first seeded live tick, so the whole
    live window is post-certification and session coverage reads ~1.0."""
    prev = MarketCalendar().previous_session(sessions[0])
    return datetime.combine(prev, datetime.min.time(), UTC).replace(hour=20).isoformat()


def _open():
    conn = connect(_dbpath())
    migrate(conn)
    return conn


def _dbpath():
    import os
    from pathlib import Path
    return Path(os.environ["ALGUA_DB_PATH"])


def test_healthy_live_is_ok():
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    sessions = _seed_live_ticks(conn, sid, ident, sharpe_target=1.5,
                                n=MIN_FORWARD_OBSERVATIONS + 20, start_after=None)
    _seed_cert(conn, sid, ident, holdout=1.0, created_at=_cert_ts_before(sessions))
    conn.close()

    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["ok"] is True
    assert payload["strategy"] == STRAT
    assert payload["verdict"] == "ok"
    assert payload["advisory"] is True
    assert payload["session_coverage"] >= 0.9  # dense contiguous window since the certificate
    assert payload["certified_baseline"]["holdout_sharpe"] == 1.0


def test_decayed_live_is_warn_exit_zero():
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    sessions = _seed_live_ticks(conn, sid, ident, sharpe_target=0.2,
                                n=MIN_FORWARD_OBSERVATIONS + 20, start_after=None)
    _seed_cert(conn, sid, ident, holdout=2.0, created_at=_cert_ts_before(sessions))  # bar = 1.0
    conn.close()

    result = runner.invoke(app, ["monitoring", "decay", STRAT])
    payload = _json(result)  # exit 0 even on a decay finding
    assert payload["verdict"] == "decay_warn"


def test_no_certificate_is_unknown():
    _register()
    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["verdict"] == "unknown"
    assert payload["recert_needed"] is True


def test_too_few_live_ticks_is_insufficient():
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    sessions = _seed_live_ticks(conn, sid, ident, sharpe_target=1.5, n=5, start_after=None)
    _seed_cert(conn, sid, ident, holdout=1.0, created_at=_cert_ts_before(sessions))
    conn.close()

    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["verdict"] == "insufficient_data"


def test_long_post_cert_gap_is_insufficient_not_ok():
    # Dense recent ticks, but the certificate is ~1 year back with no ticks for most of the
    # window: coverage must read sparse and fail closed to insufficient_data, NOT a false ok.
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    _seed_live_ticks(conn, sid, ident, sharpe_target=1.5,
                     n=MIN_FORWARD_OBSERVATIONS + 20, start_after=None)
    old_cert = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    _seed_cert(conn, sid, ident, holdout=1.0, created_at=old_cert)
    conn.close()

    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["session_coverage"] < 0.9
    assert payload["verdict"] == "insufficient_data"


def test_unparseable_certificate_created_at_is_unknown():
    # A passing certificate with a malformed created_at is UNUSABLE: no parseable window boundary
    # means an unbounded live window, so it must fail closed to unknown, never a false ok.
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    _seed_live_ticks(conn, sid, ident, sharpe_target=1.5,
                     n=MIN_FORWARD_OBSERVATIONS + 20, start_after=None)
    _seed_cert(conn, sid, ident, holdout=1.0, created_at="not-a-timestamp")
    conn.close()

    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["verdict"] == "unknown"


def test_newest_failed_certificate_invalidates_prior_pass():
    _register()
    ident = compute_artifact_hashes(STRAT)
    conn = _open()
    sid = SqliteStrategyRepository(conn).get(STRAT).id
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    new = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    _seed_cert(conn, sid, ident, passed=True, holdout=1.0, created_at=old)
    _seed_cert(conn, sid, ident, passed=False, holdout=1.0, created_at=new)
    _seed_live_ticks(conn, sid, ident, sharpe_target=1.5, n=MIN_FORWARD_OBSERVATIONS + 20,
                     start_after=None)
    conn.close()

    payload = _json(runner.invoke(app, ["monitoring", "decay", STRAT]))
    assert payload["verdict"] == "unknown"  # a newer FAIL invalidates the older pass


def test_rejects_unknown_strategy():
    result = runner.invoke(app, ["monitoring", "decay", "nope"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False
