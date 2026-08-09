"""Slice-E notify diff: pure diff_fleet/mark_notified rules + the loop's stale-skip."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from backend import poller
from backend.poller import diff_fleet, mark_notified

NOW = datetime(2026, 8, 9, 12, 0, 0, tzinfo=UTC)


def _row(
    name: str,
    stage: str = "live",
    health: str = "ok",
    reason: str | None = None,
    global_halt: bool = False,
) -> dict[str, Any]:
    # kill_switch mirrors algua/execution/fleet_health.py: {tripped, reason, global_halt}.
    return {
        "strategy": name,
        "stage": stage,
        "health": health,
        "kill_switch": {
            "tripped": reason is not None,
            "reason": reason,
            "global_halt": global_halt,
        },
    }


def _fleet(
    rows: list[dict[str, Any]],
    *,
    global_halt: bool = False,
    alerting: int | None = None,
) -> dict[str, Any]:
    if alerting is None:
        alerting = sum(
            1
            for r in rows
            if r["health"] != "ok" and r["stage"] in ("live", "paper", "forward_tested")
        )
    return {
        "ok": alerting == 0 and not global_halt,
        "global_halt": global_halt,
        "alerting": [],
        "summary": {"total": len(rows), "alerting": alerting, "by_health": {}},
        "rows": rows,
    }


def _diff(
    fleet: dict[str, Any], state: dict[str, Any] | None, now: datetime = NOW
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return diff_fleet(fleet, state, now=now)


def _settle(fleet: dict[str, Any], now: datetime = NOW) -> dict[str, Any]:
    """Baseline + mark everything notified — a fully-acknowledged steady state."""
    _, state = _diff(fleet, None, now)
    notifications, state = _diff(fleet, state, now)
    for n in notifications:
        mark_notified(state, n, now.isoformat())
    return state


# --- baseline (R2) ---


def test_baseline_on_missing_state_notifies_nothing_even_when_unhealthy() -> None:
    fleet = _fleet([_row("momo", health="halted")], global_halt=True)
    notifications, state = _diff(fleet, None)
    assert notifications == []
    assert state["strategies"]["momo"]["observed_health"] == "halted"
    assert state["strategies"]["momo"]["notified"] is None
    assert state["global_halt"] == {"observed": True, "notified_at": None}
    assert state["unhealthy"] is True


# --- worsening (R#: severity map, LOWER = worse) ---


def test_worsening_notifies_with_payload_and_high_urgency_for_halted() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    fleet = _fleet([_row("momo", health="halted", reason="drawdown breach")])
    notifications, new_state = _diff(fleet, state)
    assert [n["kind"] for n in notifications] == ["strategy"]
    n = notifications[0]
    assert n["urgency"] == "high"
    assert n["payload"] == {
        "title": "momo halted",
        "body": "live · halted — drawdown breach",
        "tag": "strategy:momo",
        "url": "/s/momo",
        "renotify": True,
    }
    assert new_state["strategies"]["momo"]["observed_health"] == "halted"


def test_worsening_to_non_halted_health_is_normal_urgency() -> None:
    _, state = _diff(_fleet([_row("momo", stage="paper")]), None)
    notifications, _ = _diff(_fleet([_row("momo", stage="paper", health="stale")]), state)
    assert [n["kind"] for n in notifications] == ["strategy"]
    assert notifications[0]["urgency"] == "normal"
    assert notifications[0]["payload"]["body"] == "paper · stale"


def test_non_operational_stage_never_notifies() -> None:
    _, state = _diff(_fleet([_row("momo", stage="backtested")], alerting=0), None)
    fleet = _fleet([_row("momo", stage="backtested", health="halted")], alerting=0)
    notifications, new_state = _diff(fleet, state)
    assert notifications == []
    # Still OBSERVED, so a later promotion diffs from truth.
    assert new_state["strategies"]["momo"]["observed_health"] == "halted"


# --- cooldown + recovery reset (R3) ---


def test_flapping_between_bad_healths_within_cooldown_is_silent() -> None:
    """The Gate-2 flapping pin: stale<->drift alternation must not notify every cycle."""
    _, state = _diff(_fleet([_row("momo")]), None)  # baseline: ok
    notifications, state = _diff(_fleet([_row("momo", health="stale")]), state)
    assert [n["kind"] for n in notifications] == ["strategy"]
    mark_notified(state, notifications[0], NOW.isoformat())
    for hours, health in enumerate(("drift", "stale", "drift", "stale"), start=1):
        t = NOW + timedelta(hours=hours)
        notifications, state = _diff(_fleet([_row("momo", health=health)]), state, t)
        assert notifications == [], f"flap to {health} at +{hours}h notified"


def test_halted_bypass_pierces_the_cooldown_exactly_once() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    notifications, state = _diff(_fleet([_row("momo", health="stale")]), state)
    mark_notified(state, notifications[0], NOW.isoformat())
    # A lateral bad flap inside the cooldown is silent...
    notifications, state = _diff(
        _fleet([_row("momo", health="drift")]), state, NOW + timedelta(hours=1)
    )
    assert notifications == []
    # ...but the transition INTO halted is materially worse: it bypasses once.
    t_halt = NOW + timedelta(hours=2)
    notifications, state = _diff(_fleet([_row("momo", health="halted")]), state, t_halt)
    assert [n["kind"] for n in notifications] == ["strategy"]
    assert notifications[0]["health"] == "halted"
    mark_notified(state, notifications[0], t_halt.isoformat())
    # After the accepted bypass the cooldown covers halted too: a halted->stale
    # ->halted flap inside the window stays silent.
    notifications, state = _diff(
        _fleet([_row("momo", health="stale")]), state, NOW + timedelta(hours=3)
    )
    assert notifications == []
    notifications, state = _diff(
        _fleet([_row("momo", health="halted")]), state, NOW + timedelta(hours=4)
    )
    assert notifications == []


def test_recovery_to_ok_resets_the_cooldown() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    notifications, state = _diff(_fleet([_row("momo", health="stale")]), state)
    mark_notified(state, notifications[0], NOW.isoformat())
    notifications, state = _diff(_fleet([_row("momo")]), state, NOW + timedelta(hours=1))
    assert [n["kind"] for n in notifications] == ["fleet_clear"]  # recovery itself
    mark_notified(state, notifications[0], (NOW + timedelta(hours=1)).isoformat())
    assert state["strategies"]["momo"]["last_bad_notified_at"] is None
    # Well inside the original 24h window, but the recovery reset the cooldown.
    notifications, _ = _diff(
        _fleet([_row("momo", health="stale")]), state, NOW + timedelta(hours=2)
    )
    assert [n["kind"] for n in notifications] == ["strategy"]


def test_old_schema_state_missing_last_bad_notified_at_reads_as_null() -> None:
    """Migration tolerance: a pre-cooldown state file must not wedge or crash the diff."""
    state = {
        "strategies": {
            "momo": {
                "observed_health": "ok",
                "observed_at": NOW.isoformat(),
                "notified": None,
            }
        },
        "global_halt": {"observed": False, "notified_at": None},
        "unhealthy": False,
    }
    notifications, new_state = _diff(_fleet([_row("momo", health="stale")]), state)
    assert [n["kind"] for n in notifications] == ["strategy"]
    assert new_state["strategies"]["momo"]["last_bad_notified_at"] is None


def test_same_health_within_24h_dedups() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    notifications, _ = _diff(
        _fleet([_row("momo", health="halted")]), state, NOW + timedelta(hours=23)
    )
    assert notifications == []


def test_same_health_after_24h_renotifies() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    notifications, _ = _diff(
        _fleet([_row("momo", health="halted")]), state, NOW + timedelta(hours=25)
    )
    assert [n["kind"] for n in notifications] == ["strategy"]


def test_improvement_resets_dedup_and_does_not_notify() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    # halted -> ok is an improvement: no STRATEGY notification, notified marker
    # cleared. (The fleet as a whole recovered, so the R6 fleet_clear fires —
    # the only notification an improvement may produce.)
    notifications, new_state = _diff(_fleet([_row("momo")]), state)
    assert [n["kind"] for n in notifications] == ["fleet_clear"]
    assert new_state["strategies"]["momo"]["notified"] is None


def test_redegradation_after_recovery_notifies_again_immediately() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    _, state = _diff(_fleet([_row("momo")]), state)  # recovered
    notifications, _ = _diff(_fleet([_row("momo", health="halted")]), state)
    # Within 24h of the first alert, but the recovery reset the dedup: still fires.
    assert [n["kind"] for n in notifications] == ["strategy"]


def test_zero_accepted_sends_leaves_notified_unchanged_and_retries_next_cycle() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    fleet = _fleet([_row("momo", health="halted")])
    notifications, state = _diff(fleet, state)
    assert len(notifications) == 1
    # send_to_all returned 0 -> NO mark_notified. Next cycle retries.
    assert state["strategies"]["momo"]["notified"] is None
    retry, _ = _diff(fleet, state)
    assert [n["kind"] for n in retry] == ["strategy"]


# --- global halt ---


def test_global_halt_flip_notifies_high_urgency_once() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    fleet = _fleet([_row("momo")], global_halt=True)
    notifications, state = _diff(fleet, state)
    halt = [n for n in notifications if n["kind"] == "global_halt"]
    assert len(halt) == 1
    assert halt[0]["urgency"] == "high"
    assert halt[0]["payload"]["tag"] == "global:halt"
    assert halt[0]["payload"]["url"] == "/"
    mark_notified(state, halt[0], NOW.isoformat())
    # Still halted next cycle: no repeat.
    notifications, state = _diff(fleet, state)
    assert [n for n in notifications if n["kind"] == "global_halt"] == []


def test_account_halt_fanout_collapses_to_one_global_notification() -> None:
    """The Gate-2 fan-out pin: an account halt marks EVERY row halted; only global:halt fires."""
    _, state = _diff(_fleet([_row(f"s{i}") for i in range(3)]), None)
    halted_rows = [_row(f"s{i}", health="halted", global_halt=True) for i in range(3)]
    notifications, state = _diff(_fleet(halted_rows, global_halt=True), state)
    assert [n["kind"] for n in notifications] == ["global_halt"]
    assert notifications[0]["payload"]["tag"] == "global:halt"
    # Rows are still OBSERVED so the eventual un-halt diffs from truth.
    assert all(state["strategies"][f"s{i}"]["observed_health"] == "halted" for i in range(3))


def test_independent_kill_switch_trip_still_notifies_per_strategy() -> None:
    _, state = _diff(_fleet([_row("momo")]), None)
    # tripped=True with its own reason, global_halt=False: an independent trip.
    fleet = _fleet([_row("momo", health="halted", reason="drawdown breach")])
    notifications, _ = _diff(fleet, state)
    assert [n["kind"] for n in notifications] == ["strategy"]
    assert "drawdown breach" in notifications[0]["payload"]["body"]


def test_global_halt_clear_resets_marker_so_next_engage_notifies() -> None:
    state = _settle(_fleet([_row("momo")], global_halt=True))
    assert state["global_halt"]["notified_at"] is not None
    _, state = _diff(_fleet([_row("momo")]), state)  # halt cleared
    assert state["global_halt"] == {"observed": False, "notified_at": None}
    notifications, _ = _diff(_fleet([_row("momo")], global_halt=True), state)
    assert [n["kind"] for n in notifications if n["kind"] == "global_halt"] == ["global_halt"]


# --- fleet recovered (R6) ---


def test_fleet_recovered_emits_exactly_once() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    assert state["unhealthy"] is True
    healthy = _fleet([_row("momo")])
    notifications, state = _diff(healthy, state)
    clear = [n for n in notifications if n["kind"] == "fleet_clear"]
    assert len(clear) == 1
    assert clear[0]["payload"]["tag"] == "global:clear"
    assert clear[0]["urgency"] == "normal"
    mark_notified(state, clear[0], NOW.isoformat())
    assert state["unhealthy"] is False
    notifications, _ = _diff(healthy, state)
    assert notifications == []  # exactly once


def test_fleet_recovered_retries_until_a_send_is_accepted() -> None:
    state = _settle(_fleet([_row("momo", health="halted")]))
    healthy = _fleet([_row("momo")])
    notifications, state = _diff(healthy, state)
    assert [n["kind"] for n in notifications] == ["fleet_clear"]
    # 0 accepted -> unhealthy stays True -> the clear re-emits next cycle.
    assert state["unhealthy"] is True
    retry, _ = _diff(healthy, state)
    assert [n["kind"] for n in retry] == ["fleet_clear"]


# --- the loop's stale-skip (R1) ---


async def _run_loop_until(
    calls: list[Any], n: int, task: asyncio.Task[None]
) -> None:
    try:
        for _ in range(200):
            if len(calls) >= n:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"poller made only {len(calls)} calls, wanted >= {n}")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _fake_run_cli(monkeypatch: pytest.MonkeyPatch, stale: bool) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    async def fake(
        *args: str, ttl_s: float, timeout_s: float = 60.0, force_refresh: bool = False
    ) -> dict[str, Any]:
        calls.append(args)
        return {
            "ok": True,
            "data": _fleet([_row("momo", health="halted")]),
            "fetched_at": "x",
            "stale": stale,
        }

    monkeypatch.setattr(poller, "run_cli", fake)
    return calls


def _record_diffs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    diffed: list[dict[str, Any]] = []

    async def fake_diff(fleet: dict[str, Any]) -> None:
        diffed.append(fleet)

    monkeypatch.setattr(poller, "_diff_and_notify", fake_diff)
    return diffed


async def test_stale_fleet_result_skips_the_diff_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _fake_run_cli(monkeypatch, stale=True)
    diffed = _record_diffs(monkeypatch)
    task = asyncio.ensure_future(poller.poll_loop())
    await _run_loop_until(calls, 4, task)
    assert diffed == []  # R1: a stale envelope is never diffed


async def test_fresh_fleet_result_is_diffed_and_registry_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _fake_run_cli(monkeypatch, stale=False)
    diffed = _record_diffs(monkeypatch)
    task = asyncio.ensure_future(poller.poll_loop())
    await _run_loop_until(calls, 4, task)
    assert len(diffed) >= 1
    assert all(f["rows"][0]["strategy"] == "momo" for f in diffed)
    # One diff per fleet refresh only — the registry target never diffs.
    fleet_calls = [c for c in calls if c == ("fleet", "health")]
    assert len(diffed) <= len(fleet_calls)


async def test_diff_exception_does_not_kill_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _fake_run_cli(monkeypatch, stale=False)

    async def boom(fleet: dict[str, Any]) -> None:
        raise RuntimeError("diff blew up")

    monkeypatch.setattr(poller, "_diff_and_notify", boom)
    task = asyncio.ensure_future(poller.poll_loop())
    await _run_loop_until(calls, 6, task)  # several passes despite the failing diff
