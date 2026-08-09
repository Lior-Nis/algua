# algua-monitor — web monitor

A **read-only** web monitor for the algua fleet. This directory is a
**standalone uv project** (own `web/uv.lock`) — slice B is the backend: a small
FastAPI app whose only seam to the platform is running the `algua` CLI as a
subprocess and serving its JSON (with a short TTL cache, singleflight, and loud
stale-serving when the CLI fails).

## Run

```sh
# one-time: install web deps (creates web/.venv + web/uv.lock)
uv sync --project web

# run the backend (from the repo root)
uv run --project web python -m backend.main
# or:
uv run --project web uvicorn backend.main:app --host 127.0.0.1 --port 8787
```

Endpoints:

- `GET /api/fleet` — `algua fleet health` rollup (10s cache TTL; serves stale
  with `stale: true` + error metadata if the CLI fails).
- `GET /healthz` — backend liveness.

## Network rule

**Bind 127.0.0.1 ONLY. Never bind 0.0.0.0.** Tailnet access goes through
`tailscale serve` (which terminates on the loopback listener). There is no
auth layer in this app — loopback + tailscale is the security boundary.

## Preconditions

- **Prod path:** the backend prefers direct exec of `<repo>/.venv/bin/algua` —
  run root `uv sync` first so that binary exists.
- **Dev fallback:** if `<repo>/.venv/bin/algua` is missing, it falls back to
  `uv run --no-sync algua ...` (slower; logged once at startup).

## Dependency isolation

Web dependencies (fastapi, uvicorn, httpx, ...) must **NEVER** be added to the
root project: the root `uv.lock` is identity-bearing — its hash is part of the
`dependency_hash` stamped into gate artifacts, so touching it invalidates
promotion identity. All web deps live here, locked in `web/uv.lock`.
