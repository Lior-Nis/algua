"""algua-monitor: read-only FastAPI backend over the algua CLI seam (slice B).

Same-origin by design: no CORS middleware (dev uses the Vite proxy). Binds
127.0.0.1 ONLY — tailnet exposure goes through `tailscale serve`, never 0.0.0.0.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.algua_cli import CliError, run_cli

app = FastAPI(title="algua-monitor")


@app.exception_handler(CliError)
async def cli_error_handler(request: Request, exc: CliError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"ok": False, "error": exc.error, "code": exc.code},
    )


@app.get("/api/fleet")
async def fleet() -> dict[str, Any]:
    # 10s success TTL: global_halt must not lag behind the fleet view.
    return await run_cli("fleet", "health", ttl_s=10.0)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
