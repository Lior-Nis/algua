"""Machine-liveness rollup for the autonomous loops (`algua ops status`).

The condition these exist to catch: on 2026-08-15 the research loop had been dead for a day and
the monitor was green, because a dead loop produces no strategy rows to be unhealthy about.
"""

import json
from datetime import UTC, datetime, timedelta

from algua.operator.loop_health import (
    MERGEBACK_STALE_AFTER_S,
    PAPER_STALE_AFTER_SESSIONS,
    RESEARCH_STALE_AFTER_S,
    loop_status,
)

NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


class _Calendar:
    """Stubbed session count — enough to prove the paper loop counts SESSIONS, not seconds."""

    def __init__(self, sessions: int = 0) -> None:
        self.sessions = sessions

    def sessions_between_instants(self, a: datetime, b: datetime) -> int:
        return self.sessions


def _digest(tmp_path, records):
    path = tmp_path / "research-runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return tmp_path


def _run(*, stamp="20260815-113000", exit_code=0, rate_limited=False, timed_out=False, **extra):
    return {"stamp": stamp, "exit_code": exit_code, "rate_limited": rate_limited,
            "timed_out": timed_out, "outcome": "completed", **extra}


def _status(tmp_path, *, sessions=0, now=NOW):
    return loop_status(tmp_path, _Calendar(sessions), now=now)


# --- research loop ---


def test_rate_limited_is_its_own_health_not_a_generic_failure(tmp_path) -> None:
    """An exhausted provider quota needs a different operator response than a crashing loop."""
    _digest(tmp_path, [_run(finished_at=NOW.isoformat(), rate_limited=True, exit_code=1)])
    research = _status(tmp_path)["loops"]["research"]
    assert research["health"] == "rate_limited"
    assert research["detail"] == "provider usage limit reached"


def test_non_zero_exit_is_failing(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=NOW.isoformat(), exit_code=1)])
    assert _status(tmp_path)["loops"]["research"]["health"] == "failing"


def test_timeout_is_failing_even_with_a_zero_exit_code(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=NOW.isoformat(), exit_code=0, timed_out=True)])
    research = _status(tmp_path)["loops"]["research"]
    assert research["health"] == "failing"
    assert research["detail"] == "timed out"


def test_a_clean_recent_run_is_ok(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=(NOW - timedelta(minutes=20)).isoformat())])
    assert _status(tmp_path)["loops"]["research"]["health"] == "ok"


def test_a_clean_but_ancient_run_is_stale(tmp_path) -> None:
    old = NOW - timedelta(seconds=RESEARCH_STALE_AFTER_S + 60)
    _digest(tmp_path, [_run(finished_at=old.isoformat())])
    assert _status(tmp_path)["loops"]["research"]["health"] == "stale"


def test_consecutive_failures_and_last_ok_are_counted_across_the_window(tmp_path) -> None:
    _digest(tmp_path, [
        _run(finished_at=(NOW - timedelta(hours=5)).isoformat(), exit_code=0),
        _run(finished_at=(NOW - timedelta(hours=3)).isoformat(), exit_code=1),
        _run(finished_at=(NOW - timedelta(hours=1)).isoformat(), exit_code=1),
    ])
    research = _status(tmp_path)["loops"]["research"]
    assert research["consecutive_failures"] == 2
    assert research["last_ok_at"] == (NOW - timedelta(hours=5)).isoformat()


def test_bare_stamp_is_read_as_LOCAL_time_not_utc(tmp_path) -> None:
    """run-research-loop.sh stamps with a bare `date` (no -u). Reading it as UTC shifts every run
    by the host offset and, east of Greenwich, lands the newest run in the FUTURE — which would
    then read as permanently fresh."""
    local_now = NOW.astimezone()  # same instant, host-local wall clock
    _digest(tmp_path, [_run(stamp=local_now.strftime("%Y%m%d-%H%M%S"))])
    research = _status(tmp_path)["loops"]["research"]
    assert research["last_run_at"] == NOW.isoformat()
    assert research["health"] == "ok"


def test_a_future_stamp_fails_closed_instead_of_reading_as_fresh(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=(NOW + timedelta(hours=2)).isoformat())])
    research = _status(tmp_path)["loops"]["research"]
    assert research["health"] == "unknown"


def test_missing_digest_is_unknown_never_ok(tmp_path) -> None:
    assert _status(tmp_path)["loops"]["research"]["health"] == "unknown"


def test_a_torn_final_line_does_not_blind_the_rollup(tmp_path) -> None:
    """The digest is appended by a shell driver; a run killed mid-append leaves a partial line."""
    path = tmp_path / "research-runs.jsonl"
    path.write_text(
        json.dumps(_run(finished_at=NOW.isoformat())) + "\n{\"stamp\": \"2026", encoding="utf-8"
    )
    assert _status(tmp_path)["loops"]["research"]["health"] == "ok"


# --- paper loop (session-gated) ---


def _marker(tmp_path, entry):
    (tmp_path / "operator_sessions.json").write_text(json.dumps({"paper": entry}), encoding="utf-8")


def test_paper_cadence_is_counted_in_sessions_not_wall_clock(tmp_path) -> None:
    """A weekend gap is not a fault — the loop is SUPPOSED to be silent when the market is shut."""
    long_ago = (NOW - timedelta(days=4)).isoformat()
    _marker(tmp_path, {"rc": 0, "session": "2026-08-11", "recorded_at": long_ago})
    assert _status(tmp_path, sessions=0)["loops"]["paper"]["health"] == "ok"
    assert _status(tmp_path, sessions=PAPER_STALE_AFTER_SESSIONS + 1)["loops"]["paper"][
        "health"
    ] == "stale"


def test_paper_non_zero_rc_is_failing(tmp_path) -> None:
    _marker(tmp_path, {"rc": 2, "session": "2026-08-14", "recorded_at": NOW.isoformat()})
    assert _status(tmp_path)["loops"]["paper"]["health"] == "failing"


def test_paper_marker_absent_is_idle_not_an_alert(tmp_path) -> None:
    """A never-run operator on a fresh install is a quiet system, not a broken one."""
    paper = _status(tmp_path)["loops"]["paper"]
    assert paper["health"] == "idle"
    assert "paper" not in _status(tmp_path)["alerting"]


# --- merge-back queue ---


def _queue(tmp_path, items):
    (tmp_path / "mergeback-queue.json").write_text(json.dumps({"items": items}), encoding="utf-8")


def test_absent_queue_file_is_idle(tmp_path) -> None:
    assert _status(tmp_path)["loops"]["mergeback"]["health"] == "idle"


def test_a_deep_but_fresh_queue_is_ok(tmp_path) -> None:
    """Depth alone is fine — the drainer runs every half hour."""
    fresh = (NOW - timedelta(minutes=5)).isoformat()
    _queue(tmp_path, {f"i{n}": {"attempts": 1, "enqueued_at": fresh} for n in range(20)})
    mergeback = _status(tmp_path)["loops"]["mergeback"]
    assert mergeback["health"] == "ok"
    assert mergeback["queue_depth"] == 20


def test_an_item_that_outlived_several_drain_cycles_is_stale(tmp_path) -> None:
    old = (NOW - timedelta(seconds=MERGEBACK_STALE_AFTER_S + 60)).isoformat()
    _queue(tmp_path, {"stuck": {"attempts": 1, "enqueued_at": old}})
    assert _status(tmp_path)["loops"]["mergeback"]["health"] == "stale"


def test_a_corrupt_queue_file_degrades_only_its_own_loop(tmp_path) -> None:
    (tmp_path / "mergeback-queue.json").write_text("{not json", encoding="utf-8")
    _digest(tmp_path, [_run(finished_at=NOW.isoformat())])
    status = _status(tmp_path)
    assert status["loops"]["mergeback"]["health"] == "unknown"
    assert status["loops"]["research"]["health"] == "ok"  # unaffected


# --- overall verdict ---


def test_ok_requires_every_loop_to_be_non_alerting(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=NOW.isoformat())])
    _marker(tmp_path, {"rc": 0, "session": "2026-08-14", "recorded_at": NOW.isoformat()})
    status = _status(tmp_path)
    assert status["ok"] is True
    assert status["alerting"] == []


def test_alerting_is_ordered_worst_first(tmp_path) -> None:
    _digest(tmp_path, [_run(finished_at=NOW.isoformat(), exit_code=1)])  # failing (worst)
    _marker(tmp_path, {"rc": 0, "session": "2026-08-14",
                       "recorded_at": (NOW - timedelta(days=4)).isoformat()})  # stale
    status = _status(tmp_path, sessions=PAPER_STALE_AFTER_SESSIONS + 1)
    assert status["ok"] is False
    assert status["alerting"] == ["research", "paper"]
