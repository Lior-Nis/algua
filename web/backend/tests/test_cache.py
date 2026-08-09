"""TTL cache, singleflight, stale-serving, and timeout classification."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
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


async def test_timeout_classified_as_cli_timeout_and_process_terminated(fake_cli: Any) -> None:
    calls = fake_cli(json.dumps({"ok": True}), hang_s=5.0)
    with pytest.raises(CliError) as excinfo:
        await run_cli("fleet", "health", ttl_s=10.0, timeout_s=0.05)
    assert excinfo.value.code == "cli_timeout"
    assert calls["procs"][0].terminated is True
