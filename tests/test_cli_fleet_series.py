"""`fleet series` — read-only per-lane tick-snapshot time series (hermetic CLI tests)."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.order_state import record_tick_snapshot
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

runner = CliRunner()

T0 = datetime(2023, 6, 15, 20, 0, tzinfo=UTC)


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _register(conn, name):
    return SqliteStrategyRepository(conn).add(name)


def _tick(conn, rec, *, tick_ts, lane="paper", equity=100_000.0):
    record_tick_snapshot(
        conn, rec.name, tick_ts=tick_ts, decision_ts=tick_ts, equity=equity,
        peak_equity=equity, positions={}, n_submitted=0, reconcile_ok=True, lane=lane,
        strategy_id=rec.id, code_hash="c", config_hash="cfg", dependency_hash="d",
        account_id="acct", cash=equity, clock_source="broker")


def _raw_row(conn, *, strategy, tick_ts, strategy_id=None, lane=None):
    """Insert a tick_snapshots row the sanctioned writer would refuse (legacy/fabricated shapes)."""
    conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, equity, positions, n_submitted, "
        "reconcile_ok, lane, strategy_id) VALUES (?,?,?,?,?,?,?,?)",
        (strategy, tick_ts, 1.0, "{}", 0, 1, lane, strategy_id))
    conn.commit()


def _invoke(*args):
    result = runner.invoke(app, ["fleet", "series", *args])
    return result, json.loads(result.stdout)


def test_event_time_ordering_not_insertion_order(monkeypatch, tmp_path):
    """Rows inserted out of chronological order (id order disagrees with tick_ts order) must come
    back sorted by parsed event time, not by rowid."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    ts = [(T0 + timedelta(days=d)).isoformat() for d in range(3)]
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        for t in (ts[2], ts[0], ts[1]):  # deliberately shuffled insertion
            _tick(conn, rec, tick_ts=t)
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert payload["ok"] is True
    assert [r["tick_ts"] for r in payload["series"]["paper"]] == ts


def test_lane_segmentation(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick(conn, rec, tick_ts=T0.isoformat(), lane="paper", equity=1.0)
        _tick(conn, rec, tick_ts=T0.isoformat(), lane="live", equity=2.0)
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert [r["equity"] for r in payload["series"]["paper"]] == [1.0]
    assert [r["equity"] for r in payload["series"]["live"]] == [2.0]


def test_lane_filter_null_vs_empty(monkeypatch, tmp_path):
    """--lane paper: the unrequested lane is null (not queried), a requested lane with no rows is
    an empty list, and the filter is echoed as lane_filter."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick(conn, rec, tick_ts=T0.isoformat(), lane="live")  # no paper rows at all
    result, payload = _invoke("s", "--lane", "paper")
    assert result.exit_code == 0, result.stdout
    assert payload["lane_filter"] == "paper"
    assert payload["series"]["live"] is None
    assert payload["truncated"]["live"] is None
    assert payload["series"]["paper"] == []
    assert payload["truncated"]["paper"] is False


def test_legacy_rows_excluded_and_counted(monkeypatch, tmp_path):
    """A pre-v21 row (strategy_id NULL) is only findable by name: never a series row, always a
    count."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick(conn, rec, tick_ts=T0.isoformat())
        _raw_row(conn, strategy=rec.name, tick_ts=T0.isoformat())  # strategy_id/lane NULL
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert payload["n_legacy_excluded"] == 1
    assert len(payload["series"]["paper"]) == 1


def test_invalid_lane_row_skipped_and_counted(monkeypatch, tmp_path):
    """Lane discipline is writer-enforced only — a raw-write 'bogus' lane must be counted, never
    mixed into a lane or crash the read."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick(conn, rec, tick_ts=T0.isoformat())
        _raw_row(conn, strategy=rec.name, tick_ts=T0.isoformat(),
                 strategy_id=rec.id, lane="bogus")
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert payload["n_invalid_lane"] == 1
    assert len(payload["series"]["paper"]) == 1
    assert payload["series"]["live"] == []


def test_filter_then_limit(monkeypatch, tmp_path):
    """--since filters BEFORE the newest-N cut: the result is the newest N of the FILTERED
    interval, and truncated reflects the filtered count exceeding the limit."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    ts = [(T0 + timedelta(days=d)).isoformat() for d in range(5)]
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        for t in ts:
            _tick(conn, rec, tick_ts=t)
    result, payload = _invoke("s", "--since", ts[1], "--limit", "2")
    assert result.exit_code == 0, result.stdout
    # filtered interval = ts[1:] (4 rows, inclusive bound); newest 2 of it, ascending
    assert [r["tick_ts"] for r in payload["series"]["paper"]] == [ts[3], ts[4]]
    assert payload["truncated"]["paper"] is True


def test_unparseable_tick_ts_skipped_and_counted(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick(conn, rec, tick_ts=T0.isoformat())
        _tick(conn, rec, tick_ts="not-a-timestamp")
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert payload["n_unparseable"] == 1
    assert len(payload["series"]["paper"]) == 1


def test_truncated_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    ts = [(T0 + timedelta(days=d)).isoformat() for d in range(3)]
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        for t in ts:
            _tick(conn, rec, tick_ts=t)
    result, payload = _invoke("s", "--limit", "2")
    assert payload["truncated"]["paper"] is True
    assert [r["tick_ts"] for r in payload["series"]["paper"]] == [ts[1], ts[2]]
    result, payload = _invoke("s", "--limit", "3")
    assert payload["truncated"]["paper"] is False
    assert len(payload["series"]["paper"]) == 3


def test_unknown_strategy_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    result, payload = _invoke("nope")
    assert result.exit_code == 1
    assert payload["ok"] is False
    assert payload["code"] == "not_found"


def test_registered_strategy_zero_ticks(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        _register(conn, "s")
    result, payload = _invoke("s")
    assert result.exit_code == 0, result.stdout
    assert payload["ok"] is True
    assert payload["series"] == {"paper": [], "live": []}
    assert payload["truncated"] == {"paper": False, "live": False}
    assert payload["n_legacy_excluded"] == 0
    assert payload["lane_filter"] is None


def test_invalid_lane_filter_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        _register(conn, "s")
    result, payload = _invoke("s", "--lane", "bogus")
    assert result.exit_code == 1
    assert payload["ok"] is False
    assert payload["code"] == "invalid_input"


def test_limit_bounds_via_console_entry_point(monkeypatch, tmp_path, capsys):
    """--limit 0 violates the option's min=1 range: through the REAL console entry point
    (algua.cli.main.main) the Click usage error must render as the JSON error envelope, never a
    Rich usage text / stack trace."""
    from algua.cli.main import main

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with pytest.raises(SystemExit) as excinfo:
        main(["fleet", "series", "s", "--limit", "0"])
    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "usage_error"
