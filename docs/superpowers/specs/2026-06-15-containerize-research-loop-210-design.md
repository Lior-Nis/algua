# Containerize the research-loop run environment (#210)

**Status:** design
**Date:** 2026-06-15
**Issue:** #210

## Goal

Package the agent's research-loop run environment as a container — **local dev/run
substrate first**, not cloud-deploy prep. Two payoffs:

1. **Reproducible + disposable local runs** — fan out many concurrent research-loop runs
   (the top-of-funnel bottleneck: backtest / walk-forward many strategies over deep history)
   without polluting the host or fighting venv/state drift.
2. **Free cloud lift later** — when a VPS/k8s deployment is eventually wanted, the runtime is
   already containerized, so it's a non-event.

There is no Docker setup today (no Dockerfile/compose). This adds it.

## Non-goals (explicitly out of scope)

- VPS / k8s / cloud provisioning (deferred until the loop produces survivors).
- The web-sourcing agent's OS-isolation (separate concern, #134 follow-up).
- An MLflow tracking-*server* container — runs use the file-based `mlruns` store
  (matches the current `mlflow_tracking_uri="mlruns"` default). YAGNI until there's a reason.
- Any change to `algua` application code, config, or the gate definition. This is pure
  packaging: a Dockerfile, a compose file, a `.dockerignore`, and one gate script.

## Key design decisions

### 0. The container is the EXPLORATION substrate, not the promotion authority

**This is the load-bearing boundary** (surfaced by GATE-1 review). The research-loop's
top-of-funnel work — `backtest`, `walk-forward`, `sweep` over deep history — is *exploration*:
it reads snapshots and writes scratch results, and it is safe to fan out across many isolated
per-run DBs. **Promotion is different.** `research promote` (and any command that burns the
single-use holdout reservation, #161/#193) depends on a *global* invariant: a given
strategy×snapshot×holdout-window may be evaluated against the holdout exactly once, ever. That
invariant lives in **one authoritative registry DB**.

If a disposable per-run DB were allowed to run `research promote`, then N concurrent runs would
each "legally" burn *their own* copy of the same holdout — silently defeating the
multiple-testing defense. So the operating rule this design encodes:

- **Per-run isolated DB → exploration only** (backtest / walk-forward / sweep, `data inspect`,
  `doctor`). These never burn the authoritative holdout.
- **Promotion / holdout-burning commands → the shared authoritative registry DB.** To run them,
  aim `ALGUA_DB_PATH` at the canonical DB (or run them on the host as today). This is a
  deliberate, explicit choice — not the disposable-run default.

This is enforced in two layers. **(a) A packaging-only footgun guard** in the container
entrypoint (`scripts/entrypoint.sh`) refuses `research promote` when `ALGUA_DB_PATH` points under
`/app/runs/` (an isolated per-run DB) unless `ALGUA_ALLOW_PROMOTE=1` is explicitly set. This is a
default-safe wall, not a security boundary — it catches the obvious mistake without any
application change. **(b) The full hard wall** — the CLI itself refusing a holdout-burn against a
non-authoritative DB regardless of how it's invoked — requires application code and is an explicit
**deferred follow-up**, out of scope for this pure-packaging change. The entrypoint guard is
deliberately minimal: it keys on the single documented agent holdout-burn command
(`research promote`), so it adds no broad subcommand-classification cruft.

### 1. Isolation model — shared read-only snapshots, per-run writable state

The expensive shared asset is the **deep-history snapshots** in `data_dir` (parquet bars +
provenance manifest): ingested once, read by every run. The state that actually needs per-run
isolation is the **registry DB** (`db_path`) and the **mlruns** store, where exploration
scratch state is written.

algua's config already separates these knobs (`ALGUA_DATA_DIR`, `ALGUA_DB_PATH`,
`ALGUA_MLFLOW_TRACKING_URI`), and `db_path` is independent of `data_dir` (it does not have to
live under it). So the seam is clean:

- **Shared, read-only:** `data_dir` snapshots — mounted `:ro`, never duplicated per run.
- **Shared, read-only:** `kb/` knowledge vault — mounted `:ro` (so `doctor` sees the real,
  non-stale KB rather than a frozen baked copy; KB-*writing* workflows like `report-experiments`
  are a separate writable-mount step, out of scope here).
- **Per-run, writable:** `db_path` and `mlruns` — pointed at a per-run subdir under `./runs/`.

This delivers the "reproducible + disposable" payoff with minimal Docker plumbing: drop the
per-run state subdir and the run is gone. (Rejected alternatives: per-run named volumes — fight
the shared-snapshot goal with more ceremony; single shared DB for everything — abandons
disposability and contends at the top-of-funnel bottleneck the issue is trying to widen.)

### 2. Per-run selection via `RUN_ID` + a launcher that validates and prepares it

A run names its state dir with a `RUN_ID` value, interpolated by compose at parse time into
`./runs/${RUN_ID:-default}/` (`algua.db`, `mlruns`). But raw `RUN_ID` interpolation has three
real holes (GATE-1): an unsanitized value (`../x`, `a/b`, spaces) can traverse outside `./runs/`;
sqlite/mlflow will **not** create the per-run parent dir, so `doctor` fails if it is absent; and
`RUN_ID` must come from the **host shell env at parse time**, not `docker compose run -e` (which
sets it only inside the container, after interpolation) — an easy footgun.

A thin launcher closes all three:

```
scripts/run.sh <RUN_ID> <algua args...>
```

It (1) validates `RUN_ID` against `^[A-Za-z0-9_.-]+$` and rejects `..`, refusing path traversal;
(2) `mkdir -p ./runs/$RUN_ID` so the per-run DB/mlruns parent exists before the container starts;
(3) refuses an existing run dir unless `ALGUA_REUSE=1` (collision guard — a stale dir silently
reused would violate "disposable/reproducible"); (4) exports `RUN_ID`, `UID`, and `GID` and
`exec`s `docker compose run --rm algua "$@"`. Direct `docker compose run` still works for the
`default` run; the launcher is the ergonomic, safe path for fan-out.

### 3. Single image, dev deps included

One stage on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. `uv sync --frozen` installs the
full dependency set including the `dev` group, so the **same image** runs both the research loop
and the parity gate. The dev tools (pytest/ruff/mypy/import-linter) are tiny next to the runtime
(vectorbt/numba/pandas/pyarrow/mlflow), so the size cost is negligible for a local-first
substrate. A runtime-only multi-stage target is trivial to split out later when cloud deploy
actually happens — deferred per the issue's "defer hosting" stance (YAGNI now).

Layer ordering: copy `pyproject.toml` + `uv.lock` and `uv sync` **before** copying the source
tree, so the dependency layer caches across source edits.

`ENTRYPOINT ["uv", "run", "algua"]` makes the CLI the container surface, exactly as the issue
asks ("the CLI (`uv run algua ...`) as the entrypoint surface").

### 4. Gate runs as a second compose service

Because the entrypoint is the `algua` CLI, the parity gate (pytest/ruff/mypy/lint-imports)
cannot run through it. compose therefore defines two services on the one image:

- `algua` — entrypoint = the CLI, for research-loop runs.
- `gate` — same image, `entrypoint: ["scripts/gate.sh"]`, runs the full gate.

Two clear named surfaces, no forgettable `--entrypoint` incantation, both sharing the one built
image. Matches the agent-first operating model where commands should be unambiguous.

### 5. Secrets

Provided at run time via `env_file: .env` (the file already exists; Alpaca paper creds live
there). **Never baked into the image** — `.env` is in `.dockerignore`, so it cannot be COPY'd in.

## Files

### `Dockerfile`

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Dependency layer — caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project source + editable install of the project itself.
COPY . .
RUN uv sync --frozen

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["doctor"]
```

`entrypoint.sh` wraps `uv run algua` with the promote footgun guard (§0); `CMD ["doctor"]` is the
default no-arg surface.

`--no-install-project` on the first sync installs only third-party deps (the project source is
not copied yet); the second sync, after `COPY . .`, installs the project itself. This is the
documented uv-in-Docker idiom and keeps the heavy dependency layer cached independently of
source changes. (`uv sync` *does* install the project on the second pass — `uv install` is not
the right tool here.)

**Platform / native deps:** the heavy deps (`pyarrow`, `vectorbt`, `numba`→`llvmlite`, `pandas`,
`mlflow`) ship manylinux **wheels for linux/amd64**, which is the declared supported build
platform and what verification runs on. On other platforms (e.g. arm64) a dep may fall back to a
source build; the fallback, if it ever bites, is to add `build-essential` to the image — noted,
not pre-added (YAGNI for the amd64 local-first target).

### `.dockerignore`

```
.git
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.import_linter_cache/
data/
runs/
mlruns/
artifacts/
*.db
*.db-wal
*.db-shm
.env
.claude/worktrees/
.superpowers/
```

Keeps the build context small and prevents host state (DBs, snapshots, secrets, caches) from
leaking into the image.

### `scripts/gate.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

The exact gate from `CLAUDE.md`. Executable (`chmod +x`). Used as the `gate` service entrypoint;
also runnable on the host.

### `scripts/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
# Footgun guard (NOT a security wall, see spec §0): refuse the documented holdout-burning
# promote on an isolated per-run DB unless explicitly opted into authoritative mode.
if [[ "${1:-}" == "research" && "${2:-}" == "promote" \
      && "${ALGUA_DB_PATH:-}" == /app/runs/* && "${ALGUA_ALLOW_PROMOTE:-0}" != "1" ]]; then
  echo "refusing 'research promote' on isolated per-run DB ($ALGUA_DB_PATH):" >&2
  echo "point ALGUA_DB_PATH at the authoritative DB, or set ALGUA_ALLOW_PROMOTE=1." >&2
  exit 4
fi
exec uv run algua "$@"
```

`chmod +x`. This is the image `ENTRYPOINT`.

### `scripts/run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?usage: scripts/run.sh <RUN_ID> <algua args...>}"; shift
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_ID" != *..* && ! "$RUN_ID" =~ ^\.+$ ]] \
  || { echo "invalid RUN_ID: $RUN_ID" >&2; exit 2; }
mkdir -p ./runs
dir="./runs/$RUN_ID"
if [[ "${ALGUA_REUSE:-0}" == "1" ]]; then
  mkdir -p "$dir"                         # explicit reuse
else
  mkdir "$dir" 2>/dev/null \              # atomic create — no TOCTOU on concurrent fan-out
    || { echo "run dir $dir exists; set ALGUA_REUSE=1 to reuse" >&2; exit 3; }
fi
export RUN_ID
export HOST_UID HOST_GID
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"   # bash UID is readonly; GID is not a builtin — derive both
exec docker compose run --rm algua "$@"
```

Validates `RUN_ID` (rejecting traversal and all-dots), pre-creates the per-run dir atomically
(sqlite/mlflow won't, and `mkdir` without `-p` is the TOCTOU-free collision guard), and exports
the env compose interpolates. `chmod +x`. **Note:** a launch that fails *after* the dir is created
leaves an empty run dir; re-run with `ALGUA_REUSE=1` or `rm -rf ./runs/<id>`.

### `docker-compose.yml`

```yaml
services:
  algua:
    build: .
    user: "${HOST_UID:-1000}:${HOST_GID:-1000}"   # host-owned ./runs, not root-owned
    env_file: .env
    environment:
      ALGUA_DATA_DIR: /snapshots
      ALGUA_DB_PATH: /app/runs/${RUN_ID:-default}/algua.db
      ALGUA_MLFLOW_TRACKING_URI: /app/runs/${RUN_ID:-default}/mlruns
    volumes:
      - ./data:/snapshots:ro          # shared deep-history snapshots, read-only
      - ./kb:/app/kb:ro               # shared knowledge vault, read-only (doctor reads it)
      - ./runs:/app/runs              # per-run writable state (DB + mlruns)

  gate:
    build: .
    entrypoint: ["./scripts/gate.sh"]
```

- The `algua` service mounts shared snapshots + KB read-only and a writable `./runs` tree;
  per-run paths resolve under `${RUN_ID:-default}`.
- `data_dir` is the shared snapshot store; `db_path` and the mlflow URI sit under the per-run
  dir, independent of `data_dir`.
- `kb/` is **mounted read-only** (not the baked copy) so `doctor`'s knowledge-base check reflects
  the real host vault and never goes stale against the image.
- `user:` runs the container as the host UID/GID (`HOST_UID`/`HOST_GID`, exported by
  `scripts/run.sh` via `id -u`/`id -g`) so per-run artifacts under `./runs` are host-owned, not
  root-owned. Default `1000:1000` for a bare `docker compose run`.
- The `gate` entrypoint is `./scripts/gate.sh` (explicit relative path; `WORKDIR` is `/app`). The
  `gate` service intentionally does **not** mount host `./kb` or `./data`: it is a **hermetic
  image-source parity** check — it tests exactly the tree baked into the image via `COPY . .`
  (which is the point of a build-parity gate), distinct from the `algua` service's runtime-mount
  topology. Repo tests monkeypatch KB/data paths, so they are unaffected by the mount difference.
- **Prerequisite:** `.env` must exist (`cp .env.example .env`) — `env_file: .env` makes compose
  fail fast if it is missing, even for `doctor`.

## Verification (the issue's gate)

Always **build fresh first** — the `gate` service runs whatever the image was last built with
(`COPY . .`), so parity with the working tree requires a rebuild, not a stale image (GATE-1):

1. `docker compose build` succeeds (declared platform: linux/amd64).
2. `docker compose run --rm gate` is green — pytest + ruff + mypy + lint-imports all pass
   inside the freshly built container (parity with the host gate).
3. `scripts/run.sh smoke doctor` exits 0 inside the container, with its per-run DB created under
   `./runs/smoke/` and host-owned (not root).
4. **Beyond `doctor`** (which alone can pass while the loop can't): `scripts/run.sh smoke data
   inspect --summary` reads the read-only `/snapshots` mount, and a command that writes MLflow
   (e.g. a tiny `walk-forward`/`backtest` if a snapshot is present, else a confirmed `mlruns/`
   dir write) proves the per-run `mlruns` is writable. Proportionate: no ingested-data dependency
   is assumed; if no snapshot exists locally, the `data inspect` + mlruns-write checks still run.
5. **Promote footgun guard:** `scripts/run.sh guardtest research promote some-strat ...` exits
   non-zero with the refusal message (per-run DB under `/app/runs/`); the same with
   `ALGUA_ALLOW_PROMOTE=1` passes the guard through to the CLI.
6. Host gate stays green (no application code changed): `uv run pytest -q && uv run ruff check .
   && uv run mypy algua && uv run lint-imports`.

## Risks / notes

- **`doctor` & the per-run DB dir**: sqlite creates the DB *file* but not its parent dir, so
  `scripts/run.sh` `mkdir -p ./runs/$RUN_ID` before the container starts. Verification step 3
  confirms end-to-end. No application change is in scope.
- **Read-only `data_dir`**: the research loop reads snapshots; it does not write them. Ingestion
  (which *does* write `data_dir`) is a separate, non-disposable prep step against a writable data
  mount — out of scope; a dedicated `ingest` compose profile is a noted **deferred follow-up**.
- **Holdout integrity across per-run DBs** (the GATE-1 load-bearing finding, see §0): per-run
  isolated DBs are exploration-only; promotion/holdout-burn must target the shared authoritative
  DB. Default + documented today; a hard CLI wall (refuse holdout-burn on a non-authoritative DB)
  is a **deferred follow-up** requiring app code.
- **Local filesystem only**: the registry DB uses SQLite WAL + a busy timeout (#164). That is
  sound on a normal local FS / bind mount, **not** on network filesystems (NFS) or some Docker
  Desktop file-sharing edge cases. Stated as a hard assumption of this local-first substrate.
- **Paper creds only**: the research `.env` should carry paper Alpaca creds, never live
  (`ALGUA_ALPACA_LIVE_*`) — a research/exploration container has no business holding live keys,
  and the live wall is enforced elsewhere regardless.
- **No Docker daemon in CI assumption**: the parity gate itself is the existing host gate; the
  container build/run verification is a local/manual step (the issue's gate), not wired into the
  repo's automated `pytest`. This keeps the change pure-packaging.

## Declined GATE-1 findings (with rationale)

- **Pre-cache calendar data at build (network at runtime)** — declined: `exchange_calendars`
  *computes* XNYS schedules in-process from rules; it does not fetch them over the network, so
  `doctor`'s calendar check needs no connectivity.
- **Use `uv install` instead of `uv sync` for the project** — declined: `uv sync` after
  `COPY . .` installs the project; the two-step `--no-install-project` → `uv sync` flow is the
  documented uv-in-Docker pattern. `uv install` is not the correct command.
- **Per-run named volumes** — declined: they fight the shared-snapshot goal with more Docker
  ceremony (see §1 rejected alternatives).
```
