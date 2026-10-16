"""The PAPER lane's whole-account loss breaker (#647).

Live has had the book breaker since #390; paper (the rehearsal lane) had none. These tests pin that
`paper run-all` evaluates the same breaker, engages the same persistent halt, flattens the whole
account, leaves no strategy ticked, and keeps its high-water mark separate from live's.
"""
from __future__ import annotations

import json
from contextlib import closing

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate
from algua.risk import global_halt
from algua.risk.book_cycle import evaluate_book_loss_breaker
from algua.risk.book_equity import clear_book_peak, get_book_peak, update_book_peak
from tests.test_paper_run_all import (  # noqa: F401  (fixtures are used by name)
    _END,
    _S1,
    _SNAP,
    _START,
    _isolated,
    _RunAllBroker,
    _second_strategy,
    _seed_allocation,
    _success_result,
    _to_paper,
)

runner = CliRunner()


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


# --- the peak is per lane ---------------------------------------------------------------------

def test_paper_and_live_peaks_are_independent(tmp_path):
    conn = _conn(tmp_path)
    update_book_peak(conn, 100_000.0, "live")
    update_book_peak(conn, 5_000.0, "paper")
    assert get_book_peak(conn, "live") == 100_000.0
    assert get_book_peak(conn, "paper") == 5_000.0
    clear_book_peak(conn, "paper")
    assert get_book_peak(conn, "paper") is None
    assert get_book_peak(conn, "live") == 100_000.0


def test_unknown_lane_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        get_book_peak(_conn(tmp_path), "papr")


def test_breaker_ratchets_only_its_own_lanes_peak(tmp_path):
    from tests.test_live_book_breaker import _Broker

    conn = _conn(tmp_path)
    assert evaluate_book_loss_breaker(conn, _Broker(5_000.0, 5_000.0), "paper") is None
    assert get_book_peak(conn, "paper") == 5_000.0
    assert get_book_peak(conn, "live") is None


# --- the cycle ------------------------------------------------------------------------------

def _run_all():
    return runner.invoke(
        app, ["paper", "run-all", "--snapshot", _SNAP, "--start", _START, "--end", _END])


def _arm(monkeypatch, broker):
    _to_paper(_S1)
    _seed_allocation(_S1)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())
    ticks: list[int] = []
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick",
                        lambda *a, **k: (ticks.append(1), _success_result())[1])
    return ticks


def _audit_actions():
    with closing(connect(get_settings().db_path)) as conn:
        return [r["action"] for r in conn.execute("SELECT action FROM audit_log")]


def test_daily_loss_halts_flattens_and_ticks_nothing(monkeypatch):
    s = get_settings()
    last = 100_000.0
    broker = _RunAllBroker(equity=last * (1.0 - s.book_max_daily_loss) - 1.0, last_equity=last)
    ticks = _arm(monkeypatch, broker)

    result = _run_all()

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["book_breach"]["kind"] == "book_daily_loss"
    assert payload["global_halt"] == "set"
    assert payload["liquidation_submitted"] is True
    assert broker.closed_all == 1
    assert ticks == []
    with closing(connect(get_settings().db_path)) as conn:
        assert global_halt.is_engaged(conn)
    assert "paper_book_circuit_breaker" in _audit_actions()


def test_drawdown_vs_paper_peak_trips(monkeypatch):
    s = get_settings()
    broker = _RunAllBroker(equity=200_000.0 * (1.0 - s.book_max_drawdown) - 1.0,
                           last_equity=200_000.0 * (1.0 - s.book_max_drawdown) - 1.0)
    ticks = _arm(monkeypatch, broker)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_book_peak(conn, 200_000.0, "paper")

    result = _run_all()

    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["book_breach"]["kind"] == "book_drawdown"
    assert ticks == []


def test_live_peak_does_not_trip_paper(monkeypatch):
    """A big LIVE high-water mark must not make a small paper account look drawn down."""
    broker = _RunAllBroker(equity=5_000.0)
    ticks = _arm(monkeypatch, broker)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_book_peak(conn, 1_000_000.0, "live")

    result = _run_all()

    assert result.exit_code == 0, result.stdout
    assert ticks == [1]
    assert broker.closed_all == 0


def test_flatten_failure_keeps_halt_and_reports(monkeypatch):
    s = get_settings()
    broker = _RunAllBroker(equity=100_000.0 * (1.0 - s.book_max_daily_loss) - 1.0,
                           last_equity=100_000.0)

    def _boom():
        raise RuntimeError("venue down")

    broker.close_all_positions = _boom  # type: ignore[method-assign]
    _arm(monkeypatch, broker)

    result = _run_all()

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["liquidation_submitted"] is False
    assert payload["flatten_error"] == "venue down"
    with closing(connect(get_settings().db_path)) as conn:
        assert global_halt.is_engaged(conn)


def test_unusable_baseline_fails_closed(monkeypatch):
    broker = _RunAllBroker(equity=100_000.0, last_equity=0.0)
    ticks = _arm(monkeypatch, broker)

    result = _run_all()

    assert result.exit_code == 1
    assert json.loads(result.stdout)["book_breach"]["kind"] == "book_baseline_unusable"
    assert ticks == []


def test_healthy_book_ticks_and_ratchets_paper_peak(monkeypatch):
    broker = _RunAllBroker(equity=100_000.0)
    ticks = _arm(monkeypatch, broker)

    result = _run_all()

    assert result.exit_code == 0, result.stdout
    assert ticks == [1]
    with closing(connect(get_settings().db_path)) as conn:
        assert get_book_peak(conn, "paper") == 100_000.0
        assert get_book_peak(conn, "live") is None


def test_paper_resume_all_rebases_the_paper_peak(monkeypatch):
    """After a flatten-to-cash the paper breaker must not re-trip against the pre-loss peak."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        update_book_peak(conn, 200_000.0, "paper")
        global_halt.engage(conn, reason="x", actor="system")

    assert runner.invoke(app, ["paper", "resume-all"]).exit_code == 0

    with closing(connect(get_settings().db_path)) as conn:
        assert get_book_peak(conn, "paper") is None


# --- error precedence is the same in both lanes (#647) -------------------------------------------

@pytest.mark.parametrize("lane", ["paper", "live"])
def test_bad_max_drawdown_is_reported_before_the_snapshot_exclusivity(lane):
    # BOTH wrong: no --snapshot/--refresh choice AND an out-of-range --max-drawdown. The same
    # invocation must name the same error in both lanes.
    result = runner.invoke(app, [lane, "run-all", "--max-drawdown", "7"])
    assert result.exit_code != 0
    assert "--max-drawdown must be in (0, 1]" in result.stdout
