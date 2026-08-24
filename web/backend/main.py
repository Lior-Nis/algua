"""algua-monitor: read-only FastAPI backend over the algua CLI seam (slices B+C+D).

Same-origin by design: no CORS middleware (dev uses the Vite proxy). Binds
127.0.0.1 ONLY — tailnet exposure goes through `tailscale serve`, never 0.0.0.0.

Every user-controlled value that reaches the CLI is validated in backend.params
and passed in ``--flag=value`` form (a single argv element), so a crafted value
can never be parsed as a separate flag.

Slice D adds a background cache-prewarm poller (backend.poller, wired via the
lifespan) and static serving of the built frontend (backend.static) — active
only when a frontend build exists at the DIST dir; otherwise the app runs
api-only. ``create_app()`` is the factory (tests build isolated instances);
``app`` is the module-level instance uvicorn targets.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend import push
from backend.algua_cli import CliError, run_cli
from backend.params import (
    InvalidParam,
    validate_action,
    validate_actor,
    validate_lane,
    validate_run_ids,
    validate_since,
    validate_strategy_name,
)
from backend.poller import poll_loop
from backend.static import dist_dir, install_static
from backend.triage import build_triage

logger = logging.getLogger(__name__)

# `research idea stats` default --window-days (algua/cli/idea_cmd.py); the UI must
# label the window the stats cover.
IDEA_STATS_WINDOW_DAYS = 90

# A real PushSubscription JSON is well under 1 KiB; anything over 16 KiB is abuse.
MAX_SUBSCRIBE_BODY_BYTES = 16 * 1024

# `registry gates` returns the newest N rows PER LEDGER (its own default is 20). Passed
# explicitly and echoed back as ``gates_limit`` so the UI can say "newest N" instead of
# presenting a truncated ledger as the complete history.
GATE_HISTORY_LIMIT = 50


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Background prewarm poller (slice D): started on startup, cancelled on shutdown.
    poll_task = asyncio.create_task(poll_loop())
    try:
        yield
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task


def create_app() -> FastAPI:
    app = FastAPI(title="algua-monitor", lifespan=_lifespan)

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

    @app.exception_handler(push.SubscriptionLimit)
    async def subscription_limit_handler(
        request: Request, exc: push.SubscriptionLimit
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "error": str(exc), "code": "subscription_limit"},
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

    @app.get("/api/ops")
    async def ops() -> dict[str, Any]:
        # 30s: loop liveness moves on a timer cadence (30-60 min), not tick-by-tick.
        return await run_cli("ops", "status", ttl_s=30.0)

    @app.get("/api/book")
    async def book() -> dict[str, Any]:
        # 30s: allocations change only on an intake/allocate/revoke, which are rare and human-paced.
        return await run_cli("book", "status", ttl_s=30.0)

    @app.get("/api/triage")
    async def triage() -> dict[str, Any]:
        """The Now screen: one ranked "needs you" list across fleet + loops + capital.

        Each part degrades independently — a failed part contributes nothing and is reported in
        ``sources`` rather than blanking the screen or, worse, rendering a false all-clear.
        """
        fleet_r, ops_r, book_r = await asyncio.gather(
            run_cli("fleet", "health", ttl_s=10.0),
            run_cli("ops", "status", ttl_s=30.0),
            run_cli("book", "status", ttl_s=30.0),
            return_exceptions=True,
        )
        parts: dict[str, Any] = {}
        fetched_ats: list[str] = []
        stale = False
        for name, result in (("fleet", fleet_r), ("ops", ops_r), ("book", book_r)):
            if isinstance(result, CliError):
                parts[name] = None
            elif isinstance(result, BaseException):
                raise result
            else:
                parts[name] = result["data"]
                fetched_ats.append(result["fetched_at"])
                stale = stale or bool(result["stale"])
        if not fetched_ats:
            # Every part failed: there is nothing to triage FROM, and rendering an empty
            # "all clear" would be a lie. Fail loudly instead.
            raise CliError("triage_unavailable", "no triage source could be read")
        return {
            "ok": True,
            **build_triage(parts["fleet"], parts["ops"], parts["book"]),
            "fetched_at": min(fetched_ats),
            "stale": stale,
        }

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
            run_cli("registry", "gates", name, f"--limit={GATE_HISTORY_LIMIT}", ttl_s=30.0),
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
            "gates_limit": GATE_HISTORY_LIMIT,
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
        # No-lane default returns BOTH lanes grouped (slice-A CLI contract) — pass
        # through verbatim.
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

    @app.get("/api/runs")
    async def runs_list(
        kind: str | None = None,
        strategy: str | None = None,
        family: str | None = None,
        sort: str | None = None,
        limit: Annotated[int, Query(ge=1)] = 100,
    ) -> dict[str, Any]:
        # `sort` is forwarded verbatim as a single argv element: the store's METRIC_COLUMNS
        # allow-list (algua/registry/store/runs.py) is the one gate for it, same as the CLI's
        # own caller. `kind`/`family` are likewise passed through — the store binds them as
        # parameterized SQL, not string-interpolated.
        args = ["runs", "list", f"--limit={limit}"]
        if kind is not None:
            args.append(f"--kind={kind}")
        if strategy is not None:
            args.append(f"--strategy={validate_strategy_name(strategy)}")
        if family is not None:
            args.append(f"--family={family}")
        if sort is not None:
            args.append(f"--sort={sort}")
        return await run_cli(*args, ttl_s=60.0)

    @app.get("/api/runs/series")
    async def runs_series(ids: str) -> dict[str, Any]:
        """Registered BEFORE `/api/runs/{run_id}` so the literal path segment "series" is never
        swallowed by that route's `{run_id}` matcher.

        `ids` (comma-separated run ids) is validated and capped BEFORE shelling out — see
        `validate_run_ids`.
        """
        unique_ids = validate_run_ids(ids)
        args = ["runs", "series", str(unique_ids[0])]
        args.extend(f"--run-id={run_id}" for run_id in unique_ids[1:])
        return await run_cli(*args, ttl_s=60.0)

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: int) -> dict[str, Any]:
        return await run_cli("runs", "show", str(run_id), ttl_s=30.0)

    @app.get("/api/push/key")
    async def push_key() -> dict[str, Any]:
        # First call generates + persists the VAPID keypair (disk + EC keygen) -> thread.
        return {"ok": True, "key": await asyncio.to_thread(push.vapid_public_key)}

    @app.post("/api/push/subscribe")
    async def push_subscribe(request: Request) -> dict[str, Any]:
        # Body cap BEFORE parsing: the Content-Length header is the cheap first
        # line; the incremental stream read is the real cap for chunked/lying
        # clients — it aborts as soon as the accumulated bytes exceed the limit,
        # never buffering an unbounded body.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None  # malformed header -> rely on the stream cap
            if declared is not None and declared > MAX_SUBSCRIBE_BODY_BYTES:
                raise InvalidParam("body too large")
        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_SUBSCRIBE_BODY_BYTES:
                raise InvalidParam("body too large")
            chunks.append(chunk)
        body = b"".join(chunks)
        try:
            sub = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise InvalidParam("body must be valid JSON") from None
        # add_subscription re-validates shape/size/limit (R12): InvalidParam -> 422
        # invalid_param, SubscriptionLimit -> 422 subscription_limit.
        await asyncio.to_thread(push.add_subscription, sub)
        return {"ok": True}

    @app.delete("/api/push/subscribe")
    async def push_unsubscribe(endpoint: str) -> dict[str, Any]:
        # Idempotent: removing an unknown endpoint is still {ok: true}.
        await asyncio.to_thread(push.remove_subscription, endpoint)
        return {"ok": True}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True}

    # Static serving LAST: the SPA catch-all matches everything, so every API route
    # above must already be registered. Active only when a frontend build exists.
    dist = dist_dir()
    if (dist / "index.html").is_file():
        logger.info("static mode: serving frontend build from %s", dist)
        install_static(app, dist)
    else:
        logger.info("api-only mode: no frontend build at %s (run `npm run build`)", dist)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
