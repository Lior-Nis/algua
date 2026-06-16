#!/usr/bin/env bash
set -euo pipefail
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
