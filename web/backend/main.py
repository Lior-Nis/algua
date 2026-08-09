"""algua-monitor: read-only FastAPI backend over the algua CLI seam (slices B+C).

Same-origin by design: no CORS middleware (dev uses the Vite proxy). Binds
127.0.0.1 ONLY — tailnet exposure goes through `tailscale serve`, never 0.0.0.0.

Every user-controlled value that reaches the CLI is validated in backend.params
and passed in ``--flag=value`` form (a single argv element), so a crafted value
can never be parsed as a separate flag.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.algua_cli import CliError, run_cli
from backend.params import (
    InvalidParam,
    validate_action,
    validate_actor,
    validate_lane,
    validate_since,
    validate_strategy_name,
)

# `research idea stats` default --window-days (algua/cli/idea_cmd.py); the UI must
# label the window the stats cover.
IDEA_STATS_WINDOW_DAYS = 90

app = FastAPI(title="algua-monitor")


@app.exception_handler(CliError)
async def cli_error_handler(request: Request, exc: CliError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"ok": False, "error": exc.error, "code": exc.code},
    )


@app.exception_handler(InvalidParam)
async def invalid_param_handler(request: Request, exc: InvalidParam) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": exc.error, "code": "invalid_param"},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Same envelope as InvalidParam so a bad ?limit= renders like every other 422.
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in err.get('loc', ()))}: {err.get('msg', 'invalid')}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": detail, "code": "invalid_param"},
    )


@app.get("/api/fleet")
async def fleet() -> dict[str, Any]:
    # 10s success TTL: global_halt must not lag behind the fleet view.
    return await run_cli("fleet", "health", ttl_s=10.0)


@app.get("/api/strategies")
async def strategies() -> dict[str, Any]:
    # `registry list` emits a bare array; the seam wraps it as {"data": [...]} — pass through.
    return await run_cli("registry", "list", ttl_s=60.0)


@app.get("/api/strategy/{name}")
async def strategy_detail(name: str) -> Any:
    """Composed per-strategy drill-down: registry show + paper show + registry gates.

    registry show is authoritative: not_found -> 404, any other CliError -> 502.
    paper/gates degrade per-part: a CliError nulls that part and surfaces its code
    in ``part_errors`` (paper show returns an idle rollup for idea-stage strategies,
    so a null part is a REAL failure, render-worthy).
    """
    validate_strategy_name(name)
    registry, paper, gates = await asyncio.gather(
        run_cli("registry", "show", name, ttl_s=30.0),
        run_cli("paper", "show", name, ttl_s=30.0),
        run_cli("registry", "gates", name, ttl_s=30.0),
        return_exceptions=True,
    )
    if isinstance(registry, BaseException):
        if isinstance(registry, CliError) and registry.code == "not_found":
            return JSONResponse(
                status_code=404,
                content={"ok": False, "error": registry.error, "code": "not_found"},
            )
        raise registry
    parts: dict[str, Any] = {}
    part_errors: dict[str, str] = {}
    fetched_ats = [registry["fetched_at"]]
    stale = bool(registry["stale"])
    for part, result in (("paper", paper), ("gates", gates)):
        if isinstance(result, CliError):
            parts[part] = None
            part_errors[part] = result.code
        elif isinstance(result, BaseException):
            raise result
        else:
            parts[part] = result["data"]
            fetched_ats.append(result["fetched_at"])
            stale = stale or bool(result["stale"])
    body: dict[str, Any] = {
        "ok": True,
        "strategy": name,
        "registry": registry["data"],
        "paper": parts["paper"],
        "gates": parts["gates"],
        "fetched_at": min(fetched_ats),
        "stale": stale,
    }
    if part_errors:
        body["part_errors"] = part_errors
    return body


@app.get("/api/strategy/{name}/series")
async def strategy_series(
    name: str,
    since: str | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    # No-lane default returns BOTH lanes grouped (slice-A CLI contract) — pass through verbatim.
    validate_strategy_name(name)
    args = ["fleet", "series", name]
    if since is not None:
        args.append(f"--since={validate_since(since)}")
    if lane is not None:
        args.append(f"--lane={validate_lane(lane)}")
    return await run_cli(*args, ttl_s=60.0)


@app.get("/api/activity")
async def activity(
    strategy: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    args = ["audit", "log", f"--limit={limit}"]
    if strategy is not None:
        args.append(f"--strategy={validate_strategy_name(strategy)}")
    if actor is not None:
        args.append(f"--actor={validate_actor(actor)}")
    if action is not None:
        args.append(f"--action={validate_action(action)}")
    return await run_cli(*args, ttl_s=30.0)


@app.get("/api/ideas")
async def ideas() -> dict[str, Any]:
    idea_list, idea_stats = await asyncio.gather(
        run_cli("research", "idea", "list", ttl_s=300.0),
        run_cli("research", "idea", "stats", ttl_s=300.0),
    )
    return {
        "ok": True,
        "ideas": idea_list["data"],
        "stats": idea_stats["data"],
        "stats_window_days": IDEA_STATS_WINDOW_DAYS,
        "fetched_at": min(idea_list["fetched_at"], idea_stats["fetched_at"]),
        "stale": bool(idea_list["stale"]) or bool(idea_stats["stale"]),
    }


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
