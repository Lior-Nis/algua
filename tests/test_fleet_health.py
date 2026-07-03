"""Fleet-wide health aggregation + liveness/staleness (#400, folds in #399's hazard)."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from algua.calendar.market_calendar import MarketCalendar
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.fleet_health import (
    OPERATIONAL_STAGES,
    STALE_AFTER_SESSIONS,
    fleet_alert,
    fleet_status,
    strategy_health,
)
from algua.execution.order_state import record_tick_snapshot, update_peak_equity
from algua.registry.db import connect, migrate
from algua.registry.gating import load_gated_strategy
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


def test_corrupt_tick_row_fails_closed_not_crash(monkeypatch, tmp_path):
    """A strategy with a corrupt persisted tick (unreadable positions JSON) must fail closed to
    'stale' with a last_tick_error — never crash and never read 'idle'/'ok' (one bad row must not
    take down the whole fleet view)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        rec = _register(conn, "s_corrupt")
        _tick(conn, rec, tick_ts=now.isoformat())
        # scribble non-JSON into the positions column of the newest row
        conn.execute("UPDATE tick_snapshots SET positions = ? WHERE strategy = ?",
                     ("{not json", rec.name))
        conn.commit()
        h = strategy_health(conn, rec, MarketCalendar(), halted_globally=False, now=now)
    assert h["health"] == "stale"
    assert h["last_tick_error"] is not None
    assert h["staleness_sessions"] > STALE_AFTER_SESSIONS


def test_fleet_status_survives_one_corrupt_strategy(monkeypatch, tmp_path):
    """fleet_status returns the full array even when one strategy's tick row is corrupt."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    now = _now()
    with closing(_conn()) as conn:
        good = _register(conn, "z_good")
        _tick(conn, good, tick_ts=now.isoformat())
        bad = _register(conn, "a_bad")
        _tick(conn, bad, tick_ts=now.isoformat())
        conn.execute("UPDATE tick_snapshots SET positions = ? WHERE strategy = ?",
                     ("{oops", bad.name))
        conn.commit()
        rows = fleet_status(conn, MarketCalendar(), now=now)
    by_name = {r["strategy"]: r for r in rows}
    assert by_name["z_good"]["health"] == "ok"
    assert by_name["a_bad"]["health"] == "stale"
    assert len(rows) == 2  # the aggregate did not collapse on the bad row


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


# ---------------------------------------------------------------------------
# #399: fleet_alert() + `fleet health` liveness/heartbeat GATE
# ---------------------------------------------------------------------------


def _row(strategy, stage, health, **extra):
    """A minimal fleet_status-shaped row for the pure fleet_alert() unit tests."""
    return {"strategy": strategy, "stage": stage, "health": health, **extra}


def test_alert_operational_stale_alerts():
    rows = [_row("s", "paper", "stale")]
    assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_operational_idle_alerts():
    """A live/paper/forward_tested strategy that NEVER ticked is a loop that never started."""
    for stage in ("live", "paper", "forward_tested"):
        rows = [_row("s", stage, "idle")]
        assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_operational_drift_and_halted_alert():
    for health in ("drift", "halted"):
        rows = [_row("s", "paper", health)]
        assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_operational_ok_does_not_alert():
    rows = [_row("s", "live", "ok")]
    assert fleet_alert(rows, halted_globally=False) == []


def test_alert_nonoperational_stale_does_not_alert():
    """The load-bearing false-positive guard: a RETIRED strategy whose last tick is ancient reads
    'stale' forever, but it is not run by a loop, so it must NOT wedge the watchdog red."""
    for stage in ("retired", "dormant", "idea", "backtested", "candidate"):
        rows = [_row("s", stage, "stale")]
        assert fleet_alert(rows, halted_globally=False) == []


def test_alert_nonoperational_idle_does_not_alert():
    rows = [_row("s", "retired", "idle")]
    assert fleet_alert(rows, halted_globally=False) == []


def test_alert_nonoperational_killswitch_halted_does_not_alert():
    """A per-strategy kill-switch left tripped on a retired strategy is 'halted' but must not alert
    (only an account-wide global halt alerts regardless of stage)."""
    rows = [_row("s", "retired", "halted")]
    assert fleet_alert(rows, halted_globally=False) == []


def test_alert_global_halt_alerts_every_row_regardless_of_stage():
    rows = [_row("r", "retired", "ok"), _row("p", "paper", "ok")]
    assert len(fleet_alert(rows, halted_globally=True)) == 2


def test_alert_corrupt_health_fails_closed():
    rows = [_row("s", "paper", "bogus_verdict")]
    assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_corrupt_stage_fails_closed():
    rows = [_row("s", "not_a_stage", "ok")]
    assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_missing_keys_fail_closed():
    rows = [{"strategy": "s"}]  # no health / no stage
    assert fleet_alert(rows, halted_globally=False) == rows


def test_alert_ranks_worst_first():
    rows = [
        _row("z", "paper", "idle"),
        _row("a", "paper", "halted"),
        _row("m", "paper", "stale"),
        _row("b", "paper", "drift"),
    ]
    out = fleet_alert(rows, halted_globally=False)
    assert [r["health"] for r in out] == ["halted", "drift", "stale", "idle"]


def test_operational_stages_match_gating(tmp_path, monkeypatch):
    """DRIFT GUARD: OPERATIONAL_STAGES' paper-lane members must be EXACTLY the stages the real
    tick surface (`load_gated_strategy`) accepts, so the gate can never silently disagree with the
    loop it watches. Probe every Stage through the gate and compare."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "g.db"))
    accepted: set[str] = set()
    # load_gated_strategy loads a REAL module by name, so every probe must use the same loadable
    # strategy name; a fresh per-stage DB keeps the rows isolated without deleting FK children.
    for stage in Stage:
        db = tmp_path / f"g_{stage.value}.db"
        monkeypatch.setenv("ALGUA_DB_PATH", str(db))
        with closing(_conn()) as conn:
            name = "cross_sectional_momentum"
            SqliteStrategyRepository(conn).add(name)
            conn.execute("UPDATE strategies SET stage = ? WHERE name = ?", (stage.value, name))
            conn.commit()
            try:
                load_gated_strategy(conn, name, "probe")
                accepted.add(stage.value)
            except ValueError:
                pass  # stage rejected by the gate
    # The paper tick surface accepts exactly PAPER + FORWARD_TESTED. LIVE is ticked by the live
    # lane (live run-all), so OPERATIONAL_STAGES = paper-accepted ∪ {live}.
    assert accepted == {Stage.PAPER.value, Stage.FORWARD_TESTED.value}
    assert OPERATIONAL_STAGES == accepted | {Stage.LIVE.value}


# --- CLI exit-code gate ---


def test_fleet_health_cli_ok_exit_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "h_ok", stage=Stage.PAPER)
        _tick(conn, rec, tick_ts=datetime.now(UTC).isoformat())
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["alerting"] == []
    assert payload["global_halt"] is False
    assert payload["summary"]["total"] == 1
    assert payload["stale_after_sessions"] == STALE_AFTER_SESSIONS
    assert payload["operational_stages"] == sorted(OPERATIONAL_STAGES)
    assert len(payload["rows"]) == 1


def test_fleet_health_cli_dead_paper_loop_exits_one(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "h_dead", stage=Stage.PAPER)
        _tick(conn, rec, tick_ts=(datetime.now(UTC) - timedelta(days=60)).isoformat())
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert [r["strategy"] for r in payload["alerting"]] == ["h_dead"]
    assert payload["alerting"][0]["health"] == "stale"


def test_fleet_health_cli_never_started_live_loop_exits_one(monkeypatch, tmp_path):
    """A LIVE strategy with no tick at all is a loop that never started -> exit 1 (#399 F1/F6)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        _register(conn, "h_live_idle", stage=Stage.LIVE)  # never ticked
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["alerting"][0]["health"] == "idle"


def test_fleet_health_cli_retired_stale_is_healthy(monkeypatch, tmp_path):
    """A retired strategy with an ancient tick must NOT trip the gate (false-positive guard)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "h_retired", stage=Stage.RETIRED)
        _tick(conn, rec, tick_ts=(datetime.now(UTC) - timedelta(days=365)).isoformat())
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True


def test_fleet_health_cli_global_halt_with_only_retired_exits_one(monkeypatch, tmp_path):
    """Global halt is account state: it trips the gate even when only a non-operational strategy
    exists (#399 F4)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "h_ret", stage=Stage.RETIRED)
        _tick(conn, rec, tick_ts=datetime.now(UTC).isoformat())
        global_halt.engage(conn, reason="panic", actor="agent")
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["global_halt"] is True


def test_fleet_health_cli_global_halt_empty_fleet_exits_one(monkeypatch, tmp_path):
    """Global halt with ZERO strategies still trips the gate (#399 F4 zero-rows)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        global_halt.engage(conn, reason="panic", actor="agent")
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["global_halt"] is True
    assert payload["summary"]["total"] == 0


def test_fleet_health_cli_empty_fleet_is_healthy(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True


def test_fleet_health_cli_status_engine_crash_fails_closed(monkeypatch, tmp_path):
    """If the status engine itself raises, @json_errors must still emit {ok:false} + exit 1
    (#399 iter-2 #3 — a crash is never a silent healthy)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))

    def _boom(*a, **k):
        raise RuntimeError("corrupt fleet state")

    monkeypatch.setattr("algua.cli.fleet_cmd.fleet_status", _boom)
    result = runner.invoke(app, ["fleet", "health"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False


def test_fleet_health_cadence_is_market_sessions_not_wallclock(monkeypatch, tmp_path):
    """A tick across a long weekend/holiday gap that is < STALE_AFTER_SESSIONS *sessions* old (but
    many wall-clock days) must NOT alert — proving the gate counts market sessions, not wall-clock
    (#399 F9). 2023-11-22 (Wed) -> 2023-11-27 (Mon): Thu 23rd = Thanksgiving holiday, Fri 24th is
    a session, weekend closed -> only ~2 completed sessions across 5 calendar days."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    cal = MarketCalendar()
    tick = datetime(2023, 11, 22, 20, 0, tzinfo=UTC)
    now = datetime(2023, 11, 27, 20, 0, tzinfo=UTC)
    wall_days = (now - tick).days
    sessions = cal.sessions_between(tick.date(), now.date())
    assert wall_days >= 5 and sessions <= STALE_AFTER_SESSIONS  # many days, few sessions
    with closing(_conn()) as conn:
        rec = _register(conn, "h_holiday", stage=Stage.PAPER)
        _tick(conn, rec, tick_ts=tick.isoformat())
        rows = fleet_status(conn, cal, now=now)
        assert rows[0]["health"] == "ok"
        assert fleet_alert(rows, halted_globally=False) == []
