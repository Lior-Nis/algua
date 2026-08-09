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

When a frontend build exists (`web/frontend/dist/index.html`, overridable via
`ALGUA_WEB_DIST`), the backend also serves it: hashed `/assets/*` with an
immutable cache header, `sw.js`/`manifest.webmanifest` with `no-cache` (so
service-worker updates propagate), and an `index.html` catch-all for deep
links (`/funnel`, `/s/x`). Without a build it runs api-only (logged at
startup). A background poller (interval `ALGUA_WEB_POLL_SECONDS`, default 600)
prewarms the fleet + strategies cache so an idle monitor still opens warm.

## Deploy (systemd + tailscale)

Assumes the repo at `/opt/algua` (matching the other units) and the env file at
`/etc/algua/algua.env` (see `deploy/systemd/README.md`). In order:

1. Root deps — provides `.venv/bin/algua`, the CLI the backend execs:

   ```sh
   cd /opt/algua && uv sync
   ```

2. Web backend deps — provides `web/.venv/bin/uvicorn`:

   ```sh
   uv sync --project web
   ```

3. Frontend build — the `dist/` the backend serves:

   ```sh
   cd web/frontend && npm ci && npm run build
   ```

4. Install and start the service (or run uvicorn directly for dev, see "Run"):

   ```sh
   sudo cp deploy/systemd/algua-web.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now algua-web
   ```

   The unit's DB-access note applies: the DB's parent dir must be writable by
   the service user (SQLite WAL sidecar `-shm`/`-wal` files + the idempotent
   `migrate()`).

5. Tailscale exposure — one-time: enable **MagicDNS** and **HTTPS
   certificates** for the tailnet in the admin console, then:

   ```sh
   tailscale serve --bg https:443 http://127.0.0.1:8787
   ```

6. Access the monitor ONLY via `https://<box>.<tailnet>.ts.net`. The service
   worker (and future push) need the ONE stable secure origin — hitting the
   bare tailnet IP is a different origin, which breaks the PWA install, its
   cache, and any notification subscription.

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
