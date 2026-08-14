"""TTL cache, singleflight, stale-serving, and timeout classification."""

from __future__ import annotations

import asyncio
import json
import signal
from typing import Any

import pytest
from backend import algua_cli
from backend.algua_cli import CliError, run_cli


async def test_fresh_cache_hit_spawns_no_second_subprocess(fake_cli: Any) -> None:
    calls = fake_cli(json.dumps({"ok": True, "n": 1}))
    first = await run_cli("fleet", "health", ttl_s=60.0)
    second = await run_cli("fleet", "health", ttl_s=60.0)
    assert calls["count"] == 1
    assert second["data"] == first["data"]
    assert second["stale"] is False
    assert second["fetched_at"] == first["fetched_at"]


async def test_expired_ttl_refetches(fake_cli: Any) -> None:
    calls = fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=0.0)
    await run_cli("fleet", "health", ttl_s=0.0)
    assert calls["count"] == 2


async def test_singleflight_two_concurrent_callers_one_subprocess(fake_cli: Any) -> None:
    calls = fake_cli(json.dumps({"ok": True, "n": 1}), hang_s=0.05)
    first, second = await asyncio.gather(
        run_cli("fleet", "health", ttl_s=60.0),
        run_cli("fleet", "health", ttl_s=60.0),
    )
    assert calls["count"] == 1
    assert first["data"] == second["data"] == {"ok": True, "n": 1}


async def test_failure_with_cache_serves_stale_with_loud_metadata(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": True, "n": 1}))
    fresh = await run_cli("fleet", "health", ttl_s=0.0)
    fake_cli("not json at all")  # next fetch fails
    stale = await run_cli("fleet", "health", ttl_s=0.0)
    assert stale["ok"] is True
    assert stale["data"] == {"ok": True, "n": 1}
    assert stale["stale"] is True
    assert stale["cache_age_s"] >= 0.0
    assert stale["last_success_at"] == fresh["fetched_at"]
    assert stale["fetched_at"] == fresh["fetched_at"]
    assert stale["last_error_code"] == "bad_output"
    assert stale["last_error_at"] is not None


async def test_failure_without_cache_raises(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": False, "error": "boom", "code": "cli_error"}))
    with pytest.raises(CliError) as excinfo:
        await run_cli("fleet", "health", ttl_s=10.0)
    assert excinfo.value.code == "cli_error"


async def test_timeout_kills_the_process_group_not_just_the_child(
    fake_cli: Any, group_signals: list[tuple[int, int]]
) -> None:
    """start_new_session makes the child a group leader; timeout cleanup must signal the whole
    group (the uv-run fallback's real CLI process lives there), never just the direct child."""
    calls = fake_cli(json.dumps({"ok": True}), hang_s=5.0)
    with pytest.raises(CliError) as excinfo:
        await run_cli("fleet", "health", ttl_s=10.0, timeout_s=0.05)
    assert excinfo.value.code == "cli_timeout"
    proc = calls["procs"][0]
    assert (proc.pid, int(signal.SIGTERM)) in group_signals
    assert proc.terminated is False  # group signal, not direct-child terminate()


async def test_cancellation_reaps_the_process_group(
    fake_cli: Any, group_signals: list[tuple[int, int]]
) -> None:
    """Server shutdown / request cancellation mid-communicate must not leak the subprocess."""
    calls = fake_cli(json.dumps({"ok": True}), hang_s=5.0)
    task = asyncio.ensure_future(run_cli("fleet", "health", ttl_s=10.0))
    await asyncio.sleep(0.05)  # let it reach communicate()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    proc = calls["procs"][0]
    assert (proc.pid, int(signal.SIGTERM)) in group_signals


def _age_entry_past_skew(key: tuple[str, ...]) -> None:
    """Back-date a cache entry past the force_refresh just-fetched guard."""
    algua_cli._cache[key]["fetched_monotonic"] -= algua_cli._FORCE_REFRESH_SKEW_S + 1.0


async def test_force_refresh_bypasses_a_fresh_ttl(fake_cli: Any) -> None:
    calls = fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=60.0)
    _age_entry_past_skew(("fleet", "health"))  # still well inside the 60s TTL
    refreshed = await run_cli("fleet", "health", ttl_s=60.0, force_refresh=True)
    assert calls["count"] == 2  # fresh cache, but force_refresh re-ran the CLI
    assert refreshed["stale"] is False


async def test_force_refresh_skew_guard_dedups_back_to_back_refreshes(fake_cli: Any) -> None:
    """An entry fetched <2s ago (a refresh that just completed ahead of us) is
    served as fresh even under force_refresh — ONE subprocess for the pair."""
    calls = fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=10.0, force_refresh=True)
    await run_cli("fleet", "health", ttl_s=10.0, force_refresh=True)
    assert calls["count"] == 1


async def test_force_refresh_failure_with_cache_serves_stale(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=60.0)  # warm the cache
    _age_entry_past_skew(("fleet", "health"))
    fake_cli("not json at all")  # the forced refresh fails
    stale = await run_cli("fleet", "health", ttl_s=60.0, force_refresh=True)
    assert stale["ok"] is True
    assert stale["data"] == {"ok": True, "n": 1}
    assert stale["stale"] is True
    assert stale["last_error_code"] == "bad_output"


async def test_force_refresh_failure_without_cache_raises(fake_cli: Any) -> None:
    fake_cli("not json at all")
    with pytest.raises(CliError) as excinfo:
        await run_cli("fleet", "health", ttl_s=60.0, force_refresh=True)
    assert excinfo.value.code == "bad_output"


async def test_ttl_fresh_hit_after_a_failed_refresh_still_reads_stale(fake_cli: Any) -> None:
    """Staleness must not FLICKER: the failing request banners, and the TTL-fresh requests
    behind it must not silently re-render the same cached payload as fresh."""
    fake_cli(json.dumps({"ok": True, "n": 1}))
    fresh = await run_cli("fleet", "health", ttl_s=0.0)
    assert fresh["stale"] is False
    fake_cli("not json at all")
    await run_cli("fleet", "health", ttl_s=0.0)  # the refresh that fails
    hit = await run_cli("fleet", "health", ttl_s=60.0)  # well inside the TTL
    assert hit["stale"] is True
    assert hit["last_error_code"] == "bad_output"
    assert hit["data"] == {"ok": True, "n": 1}


async def test_a_successful_refresh_clears_the_sticky_stale_marker(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=0.0)
    fake_cli("not json at all")
    await run_cli("fleet", "health", ttl_s=0.0)
    fake_cli(json.dumps({"ok": True, "n": 2}))
    recovered = await run_cli("fleet", "health", ttl_s=0.0)
    assert recovered["stale"] is False
    assert "last_error_code" not in recovered
    assert (await run_cli("fleet", "health", ttl_s=60.0))["stale"] is False


async def test_spawn_failure_is_cli_error_and_stale_serves(
    fake_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn OSError (missing/non-executable binary) must classify as CliError so the
    stale-serving path still works — never an uncaught 500."""
    fake_cli(json.dumps({"ok": True, "n": 1}))
    await run_cli("fleet", "health", ttl_s=0.0)  # warm the cache

    async def broken_spawn(*cmd: str, **kwargs: Any) -> Any:
        raise OSError("permission denied")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", broken_spawn)
    stale = await run_cli("fleet", "health", ttl_s=0.0)
    assert stale["stale"] is True
    assert stale["last_error_code"] == "cli_spawn_failed"
    algua_cli._cache.clear()
    with pytest.raises(CliError) as excinfo:
        await run_cli("fleet", "health", ttl_s=0.0)
    assert excinfo.value.code == "cli_spawn_failed"
