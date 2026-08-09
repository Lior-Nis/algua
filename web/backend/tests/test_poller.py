"""Poller (slice D): prewarm targets, failure isolation, interval parsing, cancellation."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from backend import poller
from backend.algua_cli import CliError

FLEET = ("fleet", "health")
REGISTRY = ("registry", "list")


async def _cancel_and_reap(task: asyncio.Task[None]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch,
    raising: dict[tuple[str, ...], Exception] | None = None,
) -> list[dict[str, Any]]:
    """Monkeypatch poller.run_cli with a recorder; argvs in ``raising`` raise their value."""
    calls: list[dict[str, Any]] = []
    raising = raising or {}

    async def fake_run_cli(
        *args: str, ttl_s: float, timeout_s: float = 60.0, force_refresh: bool = False
    ) -> dict[str, Any]:
        calls.append({"args": args, "ttl_s": ttl_s, "force_refresh": force_refresh})
        if args in raising:
            raise raising[args]
        return {"ok": True, "data": {}, "fetched_at": "x", "stale": False}

    monkeypatch.setattr(poller, "run_cli", fake_run_cli)
    return calls


async def _wait_for_calls(calls: list[dict[str, Any]], n: int) -> None:
    for _ in range(200):
        if len(calls) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"poller made only {len(calls)} calls, wanted >= {n}")


async def test_prewarms_both_targets_with_force_refresh_and_endpoint_ttls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _install_recorder(monkeypatch)
    task = asyncio.ensure_future(poller.poll_loop())
    try:
        await _wait_for_calls(calls, 4)  # at least two full passes
    finally:
        await _cancel_and_reap(task)
    assert all(c["force_refresh"] is True for c in calls)
    ttls = {c["args"]: c["ttl_s"] for c in calls}
    assert ttls == {FLEET: 10.0, REGISTRY: 60.0}  # the SERVING endpoints' TTLs
    # Pass order: fleet then registry, every cycle.
    assert [c["args"] for c in calls[:4]] == [FLEET, REGISTRY, FLEET, REGISTRY]


async def test_cli_error_on_one_target_stops_neither_the_other_nor_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _install_recorder(
        monkeypatch, raising={FLEET: CliError("cli_timeout", "fleet health timed out")}
    )
    task = asyncio.ensure_future(poller.poll_loop())
    try:
        await _wait_for_calls(calls, 6)  # several cycles despite the failing target
    finally:
        await _cancel_and_reap(task)
    argvs = [c["args"] for c in calls]
    assert argvs.count(REGISTRY) >= 2  # the healthy target keeps prewarming
    assert argvs.count(FLEET) >= 2  # the failing target keeps being retried


async def test_unexpected_exception_does_not_kill_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _install_recorder(monkeypatch, raising={REGISTRY: RuntimeError("boom")})
    task = asyncio.ensure_future(poller.poll_loop())
    try:
        await _wait_for_calls(calls, 6)
    finally:
        await _cancel_and_reap(task)
    assert [c["args"] for c in calls[:4]] == [FLEET, REGISTRY, FLEET, REGISTRY]


async def test_cancellation_stops_the_loop_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "0.01")
    calls = _install_recorder(monkeypatch)
    task = asyncio.ensure_future(poller.poll_loop())
    await _wait_for_calls(calls, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


@pytest.mark.parametrize("raw", ["nope", "-5", "0", "inf", "nan", ""])
def test_poll_seconds_bad_value_falls_back_to_default_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", raw)
    with caplog.at_level("WARNING", logger="backend.poller"):
        assert poller.poll_seconds() == 600.0
    assert "ALGUA_WEB_POLL_SECONDS" in caplog.text


def test_poll_seconds_unset_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALGUA_WEB_POLL_SECONDS", raising=False)
    assert poller.poll_seconds() == 600.0


def test_poll_seconds_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALGUA_WEB_POLL_SECONDS", "30")
    assert poller.poll_seconds() == 30.0
