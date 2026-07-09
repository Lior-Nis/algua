"""Tests for the always-on operator scheduling core (#486)."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, date, datetime

import pytest

from algua.calendar.market_calendar import MarketCalendar
from algua.operator.schedule import (
    CalendarOutOfBounds,
    MarkerCorrupt,
    OperatorLockHeld,
    SessionMarker,
    operator_run_lock,
    session_gate,
    target_session,
)

_MARKER = "operator_sessions.json"


def _cal() -> MarketCalendar:
    return MarketCalendar()


def _record(m: SessionMarker, job: str, session: date, command=None) -> None:
    m.record(
        job,
        session,
        command=command or ["algua", "paper", "run-all", "--snapshot", "S"],
        rc=0,
        host="h",
        pid=1,
    )


# --- target_session ---------------------------------------------------------------------------


def test_target_session_after_close() -> None:
    # 2023-06-01 21:30Z is after the 20:00Z close of the June 1 session.
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    assert target_session(now, _cal()) == date(2023, 6, 1)


def test_target_session_during_session_returns_previous() -> None:
    now = datetime(2023, 6, 1, 14, 0, tzinfo=UTC)
    assert target_session(now, _cal()) == date(2023, 5, 31)


def test_target_session_saturday_returns_friday() -> None:
    now = datetime(2023, 6, 3, 7, 0, tzinfo=UTC)
    assert target_session(now, _cal()) == date(2023, 6, 2)


def test_target_session_early_close_half_day() -> None:
    # 2023-11-24 (day after Thanksgiving) is a 13:00-ET (18:00Z) early close.
    cal = _cal()
    # Just after the ACTUAL early close -> that half-day session.
    after = datetime(2023, 11, 24, 18, 30, tzinfo=UTC)
    assert target_session(after, cal) == date(2023, 11, 24)
    # Before the early close but after a fixed 16:00-ET would-be close -> prior session (proves the
    # gate uses the real early close, not a fixed 21:00Z).
    before = datetime(2023, 11, 24, 17, 30, tzinfo=UTC)
    assert target_session(before, cal) == date(2023, 11, 22)


def test_target_session_before_first_session_returns_none() -> None:
    # 1800 is before the calendar's first minute -> benign bring-up, None (not an anomaly).
    now = datetime(1800, 1, 1, tzinfo=UTC)
    assert target_session(now, _cal()) is None


def test_target_session_past_horizon_raises_calendar_out_of_bounds() -> None:
    # The XNYS calendar's horizon ends ~2027; a 2099 instant has run past it.
    now = datetime(2099, 1, 1, tzinfo=UTC)
    with pytest.raises(CalendarOutOfBounds):
        target_session(now, _cal())


def test_target_session_naive_treated_as_utc() -> None:
    aware = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    naive = datetime(2023, 6, 1, 21, 30)
    cal = _cal()
    assert target_session(naive, cal) == target_session(aware, cal) == date(2023, 6, 1)


# --- SessionMarker ----------------------------------------------------------------------------


def test_marker_round_trips_enriched_entry(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    cmd = ["algua", "paper", "run-all", "--snapshot", "SNAP"]
    m.record("paper", date(2023, 6, 1), command=cmd, rc=0, host="box", pid=42)
    assert m.last_session("paper") == date(2023, 6, 1)
    entry = json.loads((tmp_path / _MARKER).read_text())["paper"]
    assert entry["session"] == "2023-06-01"
    assert entry["command"] == cmd  # full argv, not a head prefix
    assert entry["rc"] == 0
    assert "recorded_at" in entry


def test_marker_missing_file_returns_none(tmp_path) -> None:
    assert SessionMarker(tmp_path).last_session("paper") is None


def test_marker_absent_job_returns_none(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    _record(m, "paper", date(2023, 6, 1))
    assert m.last_session("research") is None


def test_marker_isolates_jobs(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    _record(m, "research", date(2023, 5, 30))
    _record(m, "paper", date(2023, 6, 1))
    assert m.last_session("research") == date(2023, 5, 30)
    assert m.last_session("paper") == date(2023, 6, 1)


def test_marker_same_job_overwrites(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    _record(m, "paper", date(2023, 6, 1))
    _record(m, "paper", date(2023, 6, 2))
    assert m.last_session("paper") == date(2023, 6, 2)


def test_marker_legacy_bare_string_still_parses(tmp_path) -> None:
    (tmp_path / _MARKER).write_text(json.dumps({"paper": "2023-06-01"}))
    assert SessionMarker(tmp_path).last_session("paper") == date(2023, 6, 1)


def test_marker_corrupt_json_raises(tmp_path) -> None:
    (tmp_path / _MARKER).write_text("{not valid json")
    with pytest.raises(MarkerCorrupt):
        SessionMarker(tmp_path).last_session("paper")


def test_marker_corrupt_non_iso_session_raises(tmp_path) -> None:
    (tmp_path / _MARKER).write_text(json.dumps({"paper": {"session": "not-a-date"}}))
    with pytest.raises(MarkerCorrupt):
        SessionMarker(tmp_path).last_session("paper")


def test_marker_corrupt_non_object_root_raises(tmp_path) -> None:
    (tmp_path / _MARKER).write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(MarkerCorrupt):
        SessionMarker(tmp_path).last_session("paper")


def test_marker_record_fsyncs_tmp_and_dir(tmp_path, monkeypatch) -> None:
    fsynced: list[int] = []
    real_fsync = os.fsync

    def _spy(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    _record(SessionMarker(tmp_path), "paper", date(2023, 6, 1))
    # Both the temp file fd and the directory fd are fsync'd (durability, round-2 #4).
    assert len(fsynced) >= 2


def test_marker_concurrent_records_preserve_both_entries(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    barrier = threading.Barrier(2)

    def _rec(job: str, day: date) -> None:
        barrier.wait()
        _record(m, job, day)

    t1 = threading.Thread(target=_rec, args=("paper", date(2023, 6, 1)))
    t2 = threading.Thread(target=_rec, args=("research", date(2023, 5, 30)))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # The marker lock serializes the RMW so neither clobbers the other.
    assert m.last_session("paper") == date(2023, 6, 1)
    assert m.last_session("research") == date(2023, 5, 30)


# --- operator_run_lock ------------------------------------------------------------------------


def test_run_lock_writes_holder_metadata(tmp_path) -> None:
    lock = tmp_path / "operator.lock"
    with operator_run_lock(lock, job="paper", host="box", pid=99):
        holder = json.loads(lock.read_text())
        assert holder["job"] == "paper"
        assert holder["pid"] == 99
        assert holder["host"] == "box"
        assert "started_at" in holder


def test_run_lock_truncates_body_on_release(tmp_path) -> None:
    lock = tmp_path / "operator.lock"
    with operator_run_lock(lock, job="paper", host="box", pid=99):
        pass
    assert lock.read_text().strip() == ""


def test_run_lock_contention_raises_with_holder(tmp_path) -> None:
    lock = tmp_path / "operator.lock"
    with operator_run_lock(lock, job="paper", host="box", pid=99):
        with pytest.raises(OperatorLockHeld) as exc:
            with operator_run_lock(lock, job="paper", host="box", pid=100):
                pass
    assert exc.value.holder is not None
    assert exc.value.holder["pid"] == 99


def test_run_lock_garbled_holder_body_yields_none(tmp_path) -> None:
    lock = tmp_path / "operator.lock"
    import fcntl

    handle = open(lock, "a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.write("garbled not json")
    handle.flush()
    try:
        with pytest.raises(OperatorLockHeld) as exc:
            with operator_run_lock(lock, job="paper", host="box", pid=1):
                pass
        assert exc.value.holder is None
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


# --- session_gate -----------------------------------------------------------------------------


def test_session_gate_fresh_marker_is_due(tmp_path) -> None:
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), SessionMarker(tmp_path))
    assert d.due is True
    assert d.session == date(2023, 6, 1)
    assert d.reason == "due"
    assert d.skipped_sessions == 0


def test_session_gate_already_ran(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    _record(m, "paper", date(2023, 6, 1))
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), m)
    assert d.due is False
    assert d.reason == "already_ran"


def test_session_gate_no_session(tmp_path) -> None:
    now = datetime(1800, 1, 1, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), SessionMarker(tmp_path))
    assert d.due is False
    assert d.session is None
    assert d.reason == "no_session"


def test_session_gate_calendar_out_of_bounds(tmp_path) -> None:
    now = datetime(2099, 1, 1, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), SessionMarker(tmp_path))
    assert d.due is False
    assert d.reason == "calendar_out_of_bounds"


def test_session_gate_marker_corrupt(tmp_path) -> None:
    (tmp_path / _MARKER).write_text("{corrupt")
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), SessionMarker(tmp_path))
    assert d.due is False
    assert d.reason == "marker_corrupt"
    assert d.session == date(2023, 6, 1)


def test_session_gate_skipped_sessions_on_one_session_gap(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    # Target is 2023-06-01; last recorded is 2023-05-30 -> one skipped session (5/31).
    _record(m, "paper", date(2023, 5, 30))
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), m)
    assert d.due is True
    assert d.skipped_sessions == 1


def test_session_gate_no_skip_in_daily_cadence(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    _record(m, "paper", date(2023, 5, 31))
    now = datetime(2023, 6, 1, 21, 30, tzinfo=UTC)
    d = session_gate("paper", now, _cal(), m)
    assert d.due is True
    assert d.skipped_sessions == 0
