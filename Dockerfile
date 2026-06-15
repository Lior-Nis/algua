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
