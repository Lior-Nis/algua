# Containerize the research-loop run environment (#210) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-first, reproducible, disposable Docker substrate for fanning out research-loop runs — Dockerfile + compose + `.dockerignore` + three small shell scripts — with zero application-code change.

**Architecture:** One image (uv + Python 3.12, dev deps included) serves both the research loop and the parity gate. Snapshots and the KB are mounted read-only and shared; each run gets its own writable `db_path` + `mlruns` under `./runs/$RUN_ID/`. A launcher (`scripts/run.sh`) validates `RUN_ID` and prepares the per-run dir; the container entrypoint (`scripts/entrypoint.sh`) wraps the CLI with a footgun guard refusing `research promote` on an isolated per-run DB (the holdout-reuse defense); a second compose service runs the gate.

**Tech Stack:** Docker, docker compose, uv, bash, pytest (hermetic script tests via PATH-stubbed `uv`/`docker`).

**Spec:** `docs/superpowers/specs/2026-06-15-containerize-research-loop-210-design.md`

**Test layout:** all new pytest tests live under `tests/container/`. Each test computes the repo
root with `REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]`. The shell-script logic is
tested by invoking the scripts via `subprocess` with stubbed `uv`/`docker` on `PATH`, so **no
Docker daemon is needed for the pytest gate** — the daemon is only needed for the manual Docker
verification in Task 8.

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

### Task 1: `.dockerignore` (keep secrets + state out of the build context)

**Files:**
- Create: `.dockerignore`
- Test: `tests/container/test_dockerignore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/container/test_dockerignore.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dockerignore_excludes_secrets_and_state():
    lines = {
        ln.strip()
        for ln in (REPO_ROOT / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    # The CRITICAL finding: never bake secrets, host DBs, snapshots, or per-run state.
    for required in {".env", "data/", "runs/", "mlruns/", ".venv/", ".git", "*.db"}:
        assert required in lines, f"{required!r} must be in .dockerignore"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/container/test_dockerignore.py -v`
Expected: FAIL — `.dockerignore` does not exist (FileNotFoundError).

- [ ] **Step 3: Create `.dockerignore`**

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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/container/test_dockerignore.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .dockerignore tests/container/test_dockerignore.py
git commit -m "feat(210): .dockerignore — keep secrets/state/caches out of the build context"
```

---

### Task 2: `scripts/gate.sh` (the parity gate, runnable in-container and on host)

**Files:**
- Create: `scripts/gate.sh`
- Test: `tests/container/test_gate_script.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/container/test_gate_script.py
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "gate.sh"


def test_gate_script_is_executable():
    assert os.access(GATE, os.X_OK), "scripts/gate.sh must be executable"


def test_gate_script_runs_the_four_quality_commands():
    body = GATE.read_text()
    # Must match the gate defined in CLAUDE.md, in order, so the container can't drift from it.
    for cmd in (
        "uv run pytest -q",
        "uv run ruff check .",
        "uv run mypy algua",
        "uv run lint-imports",
    ):
        assert cmd in body, f"gate.sh must run: {cmd}"


def test_gate_script_fails_fast():
    assert "set -euo pipefail" in GATE.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/container/test_gate_script.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Create `scripts/gate.sh` and make it executable**

```bash
#!/usr/bin/env bash
set -euo pipefail
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

Then: `chmod +x scripts/gate.sh`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/container/test_gate_script.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/gate.sh tests/container/test_gate_script.py
git commit -m "feat(210): scripts/gate.sh — the CLAUDE.md parity gate as one script"
```

---

### Task 3: `scripts/entrypoint.sh` (the promote footgun guard)

This is the holdout-reuse defense (spec §0): refuse `research promote` when `ALGUA_DB_PATH`
points at an isolated per-run DB (`/app/runs/...`) unless `ALGUA_ALLOW_PROMOTE=1`. Everything else
passes straight through to `uv run algua`.

**Files:**
- Create: `scripts/entrypoint.sh`
- Test: `tests/container/test_entrypoint_guard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/container/test_entrypoint_guard.py
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint.sh"


@pytest.fixture
def stub_uv(tmp_path):
    """A fake `uv` on PATH that logs its args and exits 0, so we never run the real CLI."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv.log"
    stub = bindir / "uv"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 0\n')
    stub.chmod(0o755)
    return bindir, log


def _run(args, env_extra, stub_bindir):
    env = {**os.environ, "PATH": f"{stub_bindir}:{os.environ['PATH']}"}
    env.update(env_extra)
    return subprocess.run(
        [str(ENTRYPOINT), *args], env=env, capture_output=True, text=True
    )


def test_refuses_promote_on_per_run_db(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db"}, bindir)
    assert r.returncode == 4
    assert not log.exists(), "uv must NOT be invoked when the guard fires"
    assert "refusing 'research promote'" in r.stderr


def test_allows_promote_on_per_run_db_with_explicit_optin(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db", "ALGUA_ALLOW_PROMOTE": "1"}, bindir)
    assert r.returncode == 0
    assert "research promote s" in log.read_text()


def test_allows_promote_against_authoritative_db(stub_uv):
    bindir, log = stub_uv
    r = _run(["research", "promote", "s"],
             {"ALGUA_DB_PATH": "/data/algua.db"}, bindir)
    assert r.returncode == 0
    assert "research promote s" in log.read_text()


def test_non_promote_commands_pass_through(stub_uv):
    bindir, log = stub_uv
    r = _run(["doctor"], {"ALGUA_DB_PATH": "/app/runs/alpha/algua.db"}, bindir)
    assert r.returncode == 0
    assert "doctor" in log.read_text()


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT, os.X_OK)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/container/test_entrypoint_guard.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Create `scripts/entrypoint.sh` and make it executable**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Footgun guard (NOT a security wall, see spec §0): refuse the documented holdout-burning
# promote on an isolated per-run DB unless explicitly opted into authoritative mode.
# The real hard wall belongs in the CLI and is a deferred follow-up.
if [[ "${1:-}" == "research" && "${2:-}" == "promote" \
      && "${ALGUA_DB_PATH:-}" == /app/runs/* && "${ALGUA_ALLOW_PROMOTE:-0}" != "1" ]]; then
  echo "refusing 'research promote' on isolated per-run DB (${ALGUA_DB_PATH:-unset}):" >&2
  echo "point ALGUA_DB_PATH at the authoritative DB, or set ALGUA_ALLOW_PROMOTE=1." >&2
  exit 4
fi
exec uv run algua "$@"
```

Then: `chmod +x scripts/entrypoint.sh`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/container/test_entrypoint_guard.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/entrypoint.sh tests/container/test_entrypoint_guard.py
git commit -m "feat(210): entrypoint promote guard — block holdout-burn on isolated per-run DB"
```

---

### Task 4: `scripts/run.sh` (the RUN_ID launcher)

Validates `RUN_ID` (rejecting path traversal and all-dots), atomically creates the per-run dir
(TOCTOU-free collision guard), exports the env compose interpolates (`RUN_ID`, `HOST_UID`,
`HOST_GID`), then `exec`s the `algua` service.

**Files:**
- Create: `scripts/run.sh`
- Test: `tests/container/test_run_launcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/container/test_run_launcher.py
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "scripts" / "run.sh"


@pytest.fixture
def stub_docker(tmp_path):
    """A fake `docker` on PATH so the happy path doesn't need a daemon."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "docker.log"
    stub = bindir / "docker"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 0\n')
    stub.chmod(0o755)
    return bindir, log


def _run(args, cwd, stub_bindir, env_extra=None):
    env = {**os.environ, "PATH": f"{stub_bindir}:{os.environ['PATH']}"}
    env.update(env_extra or {})
    return subprocess.run(
        [str(RUN), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize("bad", ["a/b", "..", ".", "has space", "../escape", ""])
def test_rejects_invalid_run_id(bad, tmp_path, stub_docker):
    bindir, log = stub_docker
    r = _run([bad, "doctor"], tmp_path, bindir)
    assert r.returncode == 2
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())
    assert not log.exists(), "docker must not run for an invalid RUN_ID"


def test_collision_without_reuse_is_refused(tmp_path, stub_docker):
    bindir, log = stub_docker
    (tmp_path / "runs" / "alpha").mkdir(parents=True)
    r = _run(["alpha", "doctor"], tmp_path, bindir)
    assert r.returncode == 3
    assert not log.exists()


def test_valid_new_run_creates_dir_and_execs_docker(tmp_path, stub_docker):
    bindir, log = stub_docker
    r = _run(["alpha", "doctor"], tmp_path, bindir)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "runs" / "alpha").is_dir()
    assert "compose run --rm algua doctor" in log.read_text()


def test_reuse_flag_allows_existing_dir(tmp_path, stub_docker):
    bindir, log = stub_docker
    (tmp_path / "runs" / "alpha").mkdir(parents=True)
    r = _run(["alpha", "doctor"], tmp_path, bindir, {"ALGUA_REUSE": "1"})
    assert r.returncode == 0, r.stderr
    assert "compose run --rm algua doctor" in log.read_text()


def test_run_launcher_is_executable():
    assert os.access(RUN, os.X_OK)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/container/test_run_launcher.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Create `scripts/run.sh` and make it executable**

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?usage: scripts/run.sh <RUN_ID> <algua args...>}"; shift
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_ID" != *..* && ! "$RUN_ID" =~ ^\.+$ ]] \
  || { echo "invalid RUN_ID: $RUN_ID" >&2; exit 2; }
mkdir -p ./runs
dir="./runs/$RUN_ID"
if [[ "${ALGUA_REUSE:-0}" == "1" ]]; then
  mkdir -p "$dir"                          # explicit reuse
else
  mkdir "$dir" 2>/dev/null \
    || { echo "run dir $dir exists; set ALGUA_REUSE=1 to reuse" >&2; exit 3; }
fi
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"   # bash UID is readonly; GID is not a builtin — derive both
export RUN_ID HOST_UID HOST_GID
exec docker compose run --rm algua "$@"
```

Then: `chmod +x scripts/run.sh`

Note the empty-string `RUN_ID` case: `"${1:?...}"` makes bash exit non-zero (status 2 from
`set -e` on the parameter-expansion error) before validation — the test asserts `returncode == 2`,
which holds. (If a future bash treats it differently, the regex `+` quantifier also rejects empty.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/container/test_run_launcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run.sh tests/container/test_run_launcher.py
git commit -m "feat(210): scripts/run.sh — RUN_ID launcher (validate, atomic per-run dir, host UID/GID)"
```

---

### Task 5: `Dockerfile` (single image, uv + Python 3.12, dev deps; non-root-safe runtime)

No pytest here — the Dockerfile is validated by `docker build` in Task 8. Build deps are
linux/amd64 wheels (declared platform). `UV_NO_SYNC=1` + a writable cache dir keep `uv run` from
trying to write the root-owned `.venv`/cache when the container runs as the host (non-root) user.

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# uv runtime hygiene: the image is fully synced at build time, so `uv run` must NOT try to
# re-sync or write the root-owned .venv when the container runs as a non-root host user.
ENV UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_COMPILE_BYTECODE=1

# Dependency layer — caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project source + install of the project itself.
COPY . .
RUN uv sync --frozen

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["doctor"]
```

- [ ] **Step 2: Sanity-check it builds (manual; needs Docker)**

Run: `docker build -t algua:dev .`
Expected: build succeeds; final image tagged `algua:dev`.

(If the build is run in an environment without a Docker daemon, defer this to Task 8 and note it.)

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat(210): Dockerfile — uv+py3.12 single image, dev deps, non-root-safe uv runtime"
```

---

### Task 6: `docker-compose.yml` (two services: `algua` + `gate`)

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `docker-compose.yml`**

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

Notes:
- `gate` deliberately mounts nothing — it is a **hermetic image-source parity** check (it tests
  the tree baked via `COPY . .`). Repo tests monkeypatch KB/data paths, so the mount difference
  is irrelevant to them.
- `${RUN_ID:-default}` resolves at compose parse time from the host env (set by `scripts/run.sh`).
- `env_file: .env` makes compose fail fast if `.env` is missing — `cp .env.example .env` first.

- [ ] **Step 2: Validate compose config (manual; needs Docker)**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (compose file parses; defaults `RUN_ID=default`, `HOST_UID/GID=1000`).

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(210): docker-compose — algua (per-run, :ro snapshots/kb) + gate services"
```

---

### Task 7: README container section + `.env` prerequisite

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Containerized runs" section to `README.md`**

Append after the existing Quickstart:

````markdown
## Containerized research-loop runs (local-first)

Reproducible, disposable runs that don't pollute the host. One image runs both the loop and the
gate. Prerequisite: `cp .env.example .env` (compose loads it; paper creds only — never live).

```bash
# Build the image.
docker compose build

# A disposable run gets its own DB + mlruns under ./runs/<RUN_ID>/; snapshots + kb are shared :ro.
scripts/run.sh alpha doctor
scripts/run.sh alpha backtest ...        # explore: backtest / walk-forward / sweep

# Re-use an existing run dir on purpose:
ALGUA_REUSE=1 scripts/run.sh alpha ...

# Run the same quality gate the host runs, inside the freshly built image:
docker compose build && docker compose run --rm gate
```

**Exploration vs promotion (important):** disposable per-run DBs are for *exploration* only
(backtest / walk-forward / sweep). `research promote` burns the single-use holdout, which must be
tracked in the **one shared authoritative DB** — running it against a throwaway per-run DB would
silently reuse the holdout across runs. The container refuses `research promote` on a per-run DB
unless you explicitly point `ALGUA_DB_PATH` at the authoritative DB (or set `ALGUA_ALLOW_PROMOTE=1`).

**Assumptions / limits:** linux/amd64; a local filesystem (SQLite WAL is not safe on NFS); data
*ingest* (which writes snapshots) is a separate step, not part of the read-only disposable-run path.
````

- [ ] **Step 2: Verify it renders / no broken fences**

Run: `uv run python -c "import pathlib; t=pathlib.Path('README.md').read_text(); assert t.count('\`\`\`') % 2 == 0, 'unbalanced code fences'; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(210): README — containerized runs, .env prereq, exploration-vs-promote boundary"
```

---

### Task 8: Full verification (host gate + manual Docker gate)

**Files:** none (verification only).

- [ ] **Step 1: Host gate green (no app code changed)**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (including the new `tests/container/` tests).

- [ ] **Step 2: Docker build (needs daemon)**

Run: `cp .env.example .env 2>/dev/null; docker compose build`
Expected: build succeeds.

- [ ] **Step 3: In-container parity gate**

Run: `docker compose run --rm gate`
Expected: pytest + ruff + mypy + lint-imports all pass inside the container.

- [ ] **Step 4: `doctor` in the per-run container**

Run: `scripts/run.sh smoke doctor`
Expected: exits 0; `./runs/smoke/` created and **host-owned** (`ls -ld runs/smoke` shows your
user, not root).

- [ ] **Step 5: Promote footgun guard fires**

Run: `scripts/run.sh guardtest research promote any-strat`
Expected: non-zero exit + "refusing 'research promote' on isolated per-run DB" on stderr.

- [ ] **Step 6: Snapshot read + mlruns write reachable**

Run: `scripts/run.sh smoke data inspect --summary`
Expected: reads the read-only `/snapshots` mount and emits JSON (empty summary is fine if no
snapshots are ingested locally — the point is the mount + CLI work). The `doctor`/`data` runs also
prove `./runs/smoke/mlruns` is writable by the non-root container user.

- [ ] **Step 7: Final commit (if any verification fixups were needed)**

```bash
git add -A
git commit -m "chore(210): container verification fixups"   # only if Steps 2-6 required changes
```

---

## Self-review notes (author)

- **Spec coverage:** Dockerfile (T5), `.dockerignore` (T1), docker-compose with shared-:ro
  snapshots/kb + per-run writable DB/mlruns (T6), `RUN_ID` launcher (T4), promote guard / §0
  holdout defense (T3), gate parity service (T2/T6), secrets via env_file never baked (T1/T6),
  README + boundaries (T7), full verification incl. the issue's `docker build` + `doctor` gate
  (T8). Deferred items (ingest profile, hard CLI promote wall) are documented, not built.
- **No app code touched** — the import-linter / mypy contracts are unaffected; all logic lives in
  shell scripts tested hermetically.
- **Names are consistent:** `HOST_UID`/`HOST_GID` (launcher exports ↔ compose `user:`),
  `ALGUA_ALLOW_PROMOTE` / `ALGUA_REUSE` (scripts ↔ README ↔ tests), `/app/runs/...` guard prefix
  (entrypoint ↔ compose `ALGUA_DB_PATH`).
