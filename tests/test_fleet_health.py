"""Fleet-wide health aggregation + liveness/staleness (#400, folds in #399's hazard)."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from algua.calendar.market_calendar import MarketCalendar
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.fleet_health import STALE_AFTER_SESSIONS, fleet_status, strategy_health
from algua.execution.order_state import record_tick_snapshot, update_peak_equity
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch

runner = CliRunner()


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _register(conn, name, stage=Stage.PAPER):
    repo = SqliteStrategyRepository(conn)
    repo.add(name)
    # Drive to the requested stage directly at the DB level for a hermetic unit test.
    if stage is not Stage.IDEA:
        conn.execute("UPDATE strategies SET stage = ? WHERE name = ?", (stage.value, name))
        conn.commit()
    return repo.get(name)


def _tick(conn, rec, *, tick_ts, equity=100_000.0, peak=100_000.0, reconcile_ok=True):
    update_peak_equity(conn, rec.name, peak)
    record_tick_snapshot(
        conn, rec.name, tick_ts=tick_ts, decision_ts=tick_ts, equity=equity, peak_equity=peak,
        positions={}, n_submitted=0, reconcile_ok=reconcile_ok, lane="paper", strategy_id=rec.id,
        code_hash="c", config_hash="cfg", dependency_hash="d", account_id="acct", cash=equity,
        clock_source="broker")


def _now():
    return datetime(2023, 6, 15, 20, 0, tzinfo=UTC)


def test_fresh_tick_is_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_ok")
        _tick(conn, rec, tick_ts=now.isoformat())  # same session -> 0 stale
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "ok"
    assert h["staleness_sessions"] == 0


def test_dead_loop_stale_not_ok(monkeypatch, tmp_path):
    """#399: a strategy that ticked days ago then went silent must NOT read 'ok'."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    old = (now - timedelta(days=30)).isoformat()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_dead")
        _tick(conn, rec, tick_ts=old)
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "stale"
    assert h["staleness_sessions"] > STALE_AFTER_SESSIONS


def test_never_ticked_is_idle_not_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_idle")
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "idle"
    assert h["staleness_sessions"] is None


def test_unparseable_tick_fails_closed_to_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_bad")
        # tz-naive tick_ts (a raw-write fabrication) -> _parse_utc returns None -> fail closed.
        _tick(conn, rec, tick_ts="2023-06-15T20:00:00")
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "stale"
    assert h["staleness_sessions"] > STALE_AFTER_SESSIONS


def test_future_tick_fails_closed_to_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    future = (now + timedelta(days=10)).isoformat()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_future")
        _tick(conn, rec, tick_ts=future)
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "stale"


def test_reconcile_drift_beats_stale(monkeypatch, tmp_path):
    """A fresh but drifting tick is 'drift'; drift outranks stale/ok in precedence."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_drift")
        _tick(conn, rec, tick_ts=now.isoformat(), reconcile_ok=False)
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "drift"


def test_kill_switch_and_global_halt_win(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_killed")
        # even a drifting fresh tick is dominated by 'halted'
        _tick(conn, rec, tick_ts=now.isoformat(), reconcile_ok=False)
        kill_switch.trip(conn, rec.name, reason="manual", actor="agent")
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
        assert h["health"] == "halted"
        assert h["kill_switch"]["tripped"] is True
        # global halt engages 'halted' for a strategy that is otherwise ok
        rec2 = _register(conn, "s_gh")
        _tick(conn, rec2, tick_ts=now.isoformat())
        gh = strategy_health(conn, rec2, MarketCalendar(), halted_globally=True, now=now)
    assert gh["health"] == "halted"
    assert gh["kill_switch"]["global_halt"] is True


def test_fleet_status_ranks_worst_first(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        ok = _register(conn, "a_ok")
        _tick(conn, ok, tick_ts=now.isoformat())
        _register(conn, "b_idle")  # never ticked
        stale = _register(conn, "c_stale")
        _tick(conn, stale, tick_ts=(now - timedelta(days=30)).isoformat())
        drift = _register(conn, "d_drift")
        _tick(conn, drift, tick_ts=now.isoformat(), reconcile_ok=False)
        killed = _register(conn, "e_killed")
        _tick(conn, killed, tick_ts=now.isoformat())
        kill_switch.trip(conn, killed.name, reason="x", actor="agent")
        rows = fleet_status(conn, MarketCalendar(), now=now)
    assert [r["health"] for r in rows] == ["halted", "drift", "stale", "idle", "ok"]
    # covers ALL stages/strategies in one call
    assert {r["strategy"] for r in rows} == {"a_ok", "b_idle", "c_stale", "d_drift", "e_killed"}


def test_fleet_status_global_halt_marks_all_halted(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        a = _register(conn, "a")
        _tick(conn, a, tick_ts=now.isoformat())
        b = _register(conn, "b")
        _tick(conn, b, tick_ts=now.isoformat())
        global_halt.engage(conn, reason="panic", actor="agent")
        rows = fleet_status(conn, MarketCalendar(), now=now)
    assert all(r["health"] == "halted" for r in rows)


def test_fleet_status_cli_emits_bare_array(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "cli_s")
        _tick(conn, rec, tick_ts=datetime.now(UTC).isoformat())
    result = runner.invoke(app, ["fleet", "status"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload[0]["strategy"] == "cli_s"
    assert "health" in payload[0] and "staleness_sessions" in payload[0]


def test_fleet_status_cli_empty_fleet(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    result = runner.invoke(app, ["fleet", "status"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == []
