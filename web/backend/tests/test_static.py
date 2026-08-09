"""Static serving (slice D): SPA deep links, cache headers, API 404 discipline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import backend.main as main_mod
import httpx
import pytest
from fastapi import FastAPI

INDEX_HTML = "<!doctype html><title>algua-monitor</title>"


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp frontend build the app is pointed at via ALGUA_WEB_DIST."""
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "icons").mkdir()
    (d / "index.html").write_text(INDEX_HTML)
    (d / "assets" / "app.AbC123.js").write_text("console.log('app')")
    (d / "sw.js").write_text("// service worker")
    (d / "manifest.webmanifest").write_text("{}")
    (d / "favicon.svg").write_text("<svg/>")
    (d / "icons" / "icon-192.png").write_bytes(b"\x89PNG")
    monkeypatch.setenv("ALGUA_WEB_DIST", str(d))
    return d


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.parametrize("path", ["/", "/funnel", "/s/x"])
async def test_deep_links_serve_index_with_no_cache(dist: Path, path: str) -> None:
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get(path)
    assert resp.status_code == 200
    assert resp.text == INDEX_HTML
    assert resp.headers["cache-control"] == "no-cache"


async def test_hashed_asset_served_with_immutable_cache_header(dist: Path) -> None:
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get("/assets/app.AbC123.js")
    assert resp.status_code == 200
    assert resp.text == "console.log('app')"
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize("path", ["/sw.js", "/manifest.webmanifest", "/favicon.svg"])
async def test_top_level_pwa_files_served_with_no_cache(dist: Path, path: str) -> None:
    # sw.js/manifest MUST be no-cache: a cached service worker would pin old builds.
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get(path)
    assert resp.status_code == 200
    assert resp.text != INDEX_HTML  # the file itself, not the SPA fallback
    assert resp.headers["cache-control"] == "no-cache"


async def test_icons_subdir_file_is_served(dist: Path) -> None:
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get("/icons/icon-192.png")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG"
    assert resp.headers["cache-control"] == "no-cache"


async def test_unknown_api_path_is_json_404_never_index(dist: Path) -> None:
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get("/api/unknown")
    assert resp.status_code == 404
    assert resp.json() == {"ok": False, "error": "not found", "code": "not_found"}


async def test_healthz_still_works_with_static_installed(dist: Path) -> None:
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


async def test_api_routes_keep_priority_over_the_catch_all(
    dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {"ok": True, "data": {"rows": []}, "fetched_at": "x", "stale": False}

    async def fake_run_cli(
        *args: str, ttl_s: float, timeout_s: float = 60.0, force_refresh: bool = False
    ) -> dict[str, Any]:
        return envelope

    monkeypatch.setattr(main_mod, "run_cli", fake_run_cli)
    app = main_mod.create_app()
    async with _client(app) as client:
        resp = await client.get("/api/fleet")
    assert resp.status_code == 200
    assert resp.json() == envelope


async def test_api_only_mode_when_no_build_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALGUA_WEB_DIST", str(tmp_path / "missing-dist"))
    app = main_mod.create_app()
    async with _client(app) as client:
        funnel = await client.get("/funnel")
        healthz = await client.get("/healthz")
    assert funnel.status_code == 404  # no SPA fallback in api-only mode
    assert healthz.status_code == 200
