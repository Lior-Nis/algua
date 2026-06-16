FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Runtime OS tools the platform genuinely needs (not just for tests):
#   - git: the provenance code_hash is a git tree hash; without git every backtest stamps a
#     NULL code_hash, breaking the promotion/live-gate dependency identity. The operator
#     launchers also drive git worktrees.
#   - openssh-client: the live-gate go-live signature path verifies with `ssh-keygen -Y`.
# Pinned to a single cached layer before dependency install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime hygiene: the image is built (synced + bytecode-compiled) as root, but the
# container runs as the host UID/GID (so per-run ./runs artifacts are host-owned). A non-root
# user cannot write the root-owned .venv or the build-time caches, so every cache a runtime
# import wants to write must live in a world-writable location:
#   - UV_NO_SYNC: `uv run` must not try to re-sync the already-complete venv.
#   - UV_NO_CACHE: uv uses an ephemeral temp dir instead of a root-owned cache dir.
#   - HOME=/tmp: tools that default caches under $HOME get a writable home.
#   - NUMBA_CACHE_DIR: vectorbt's @njit(cache=True) writes compiled caches here, NOT into the
#     root-owned site-packages (which raises "no locator available" for a non-root user).
#   - MPLCONFIGDIR: matplotlib (pulled in transitively) writes its config cache here.
ENV UV_NO_SYNC=1 \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/mpl

# Dependency layer — caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project source + install of the project itself.
COPY . .
RUN uv sync --frozen

# Provenance needs a git HEAD to stamp code_hash. The build context excludes the real .git (to
# keep it small), so seed a deterministic single-commit repo of the baked tree. The committed
# .gitignore keeps .venv/caches out of the commit. The resulting SHA identifies THIS image's
# source tree; promotion still re-stamps against the host's authoritative repo/DB. Fixed
# identity + dates => identical source yields an identical code_hash across rebuilds.
RUN git init -q \
 && git add -A \
 && GIT_AUTHOR_DATE="2020-01-01T00:00:00 +0000" GIT_COMMITTER_DATE="2020-01-01T00:00:00 +0000" \
    git -c user.email=build@algua.local -c user.name=algua-build \
        commit -q -m "algua containerized build snapshot" \
 && git config --system --add safe.directory /app
# /app/.git is built root-owned, but the container runs as the host UID; without this, git's
# dubious-ownership guard makes `git rev-parse HEAD` fail for the non-root user and code_hash
# falls back to NULL. System config applies regardless of $HOME/user.

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["doctor"]
