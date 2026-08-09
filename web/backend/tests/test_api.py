"""API surface: httpx ASGITransport against the app, run_cli monkeypatched."""

from __future__ import annotations

from typing import Any

import backend.main as main_mod
import httpx
import pytest
from backend.algua_cli import CliError


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=main_mod.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_fleet_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    envelope = {
        "ok": True,
        "data": {"ok": False, "summary": {"alerting": 0}, "rows": []},
        "fetched_at": "2026-08-09T00:00:00+00:00",
        "stale": False,
    }
    seen: dict[str, Any] = {}

    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        seen["args"] = args
        seen["ttl_s"] = ttl_s
        return envelope

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    async with _client() as client:
        resp = await client.get("/api/fleet")
    assert resp.status_code == 200
    assert resp.json() == envelope
    assert seen["args"] == ("fleet", "health")
    assert seen["ttl_s"] == 10.0


async def test_fleet_cli_error_is_502_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
        raise CliError("cli_timeout", "algua fleet health exceeded 60.0s wall timeout")

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    async with _client() as client:
        resp = await client.get("/api/fleet")
    assert resp.status_code == 502
    assert resp.json() == {
        "ok": False,
        "error": "algua fleet health exceeded 60.0s wall timeout",
        "code": "cli_timeout",
    }


async def test_healthz() -> None:
    async with _client() as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
