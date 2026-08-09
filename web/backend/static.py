"""Static serving of the built frontend (slice D).

Explicit routes, NOT a root ``StaticFiles(html=True)`` mount: hashed
``/assets/*`` files get an immutable cache header, the unhashed top-level dist
files (``sw.js``, ``manifest.webmanifest``, icons) get ``no-cache`` so
service-worker updates propagate, unknown ``/api/*`` paths stay a JSON 404
(never index.html), and every other GET falls through to ``index.html`` —
which is what keeps deep links (``/funnel``, ``/s/x``) working.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# web/backend/static.py -> web/backend -> web; the vite build lands in web/frontend/dist.
_DEFAULT_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# Hashed asset filenames: content changes mint a new URL, so cache forever.
_CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
# Unhashed files (index.html, sw.js, manifest, icons): revalidate every time so a
# new build — especially a service-worker update — propagates immediately.
_CACHE_NO_CACHE = "no-cache"


def dist_dir() -> Path:
    """The frontend build dir: ALGUA_WEB_DIST override (tests, prod relocation) or default."""
    override = os.environ.get("ALGUA_WEB_DIST")
    return Path(override) if override else _DEFAULT_DIST


def _known_top_level(dist: Path) -> dict[str, Path]:
    """URL path -> file for the KNOWN top-level dist files (PWA files, favicons, icons/).

    Snapshotted at install time: only what the build actually emitted is servable,
    so the catch-all never maps arbitrary request paths onto the filesystem.
    """
    known = {p.name: p for p in dist.iterdir() if p.is_file()}
    icons = dist / "icons"
    if icons.is_dir():
        known.update({f"icons/{p.name}": p for p in icons.iterdir() if p.is_file()})
    return known


def install_static(app: FastAPI, dist: Path) -> None:
    """Register the static routes on ``app``. MUST be called after every API route:
    the SPA catch-all matches everything, so it has to be the LAST route registered."""
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.middleware("http")
        async def assets_cache_control(request: Request, call_next: Any) -> Response:
            # StaticFiles does not set Cache-Control; scope the immutable header to
            # the hashed-assets mount only.
            response: Response = await call_next(request)
            if request.url.path.startswith("/assets/") and response.status_code == 200:
                response.headers["Cache-Control"] = _CACHE_IMMUTABLE
            return response

    known = _known_top_level(dist)
    index = dist / "index.html"

    # GET + HEAD: proxies and health probes HEAD these routes (FileResponse handles
    # HEAD bodies itself once the method is allowed).
    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa_fallback(full_path: str) -> Response:
        # The API namespace must NEVER fall through to index.html: an unknown
        # /api/* path is a JSON 404 in the standard error envelope.
        if full_path == "api" or full_path.startswith("api/") or full_path == "healthz":
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": "not found", "code": "not_found"},
            )
        file = known.get(full_path)
        if file is not None:
            return FileResponse(file, headers={"Cache-Control": _CACHE_NO_CACHE})
        # Deep link (/funnel, /s/x, /) -> the SPA shell.
        return FileResponse(index, headers={"Cache-Control": _CACHE_NO_CACHE})
